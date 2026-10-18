/*
 * engine.js — game rules, board symmetry and Q-table lookup.
 * Pure logic: no DOM access. Mirrors rl/game.py and rl/symmetry.py.
 *
 * Exposes one global namespace: window.TTT.engine
 */
(function () {
  "use strict";

  const EMPTY = 0;
  const X = 1;
  const O = -1;
  const CELLS = 9;

  const LINES = Object.freeze([
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
  ]);

  // ---------------------------------------------------------------- rules

  /** @returns {{w: number, line: number[]} | null}  w: 1 X wins, -1 O wins, 0 draw */
  function winnerInfo(board) {
    for (const [a, b, c] of LINES) {
      const s = board[a] + board[b] + board[c];
      if (s === 3) return { w: X, line: [a, b, c] };
      if (s === -3) return { w: O, line: [a, b, c] };
    }
    return board.includes(EMPTY) ? null : { w: 0, line: [] };
  }

  function legalMoves(board) {
    const moves = [];
    for (let i = 0; i < CELLS; i++) if (board[i] === EMPTY) moves.push(i);
    return moves;
  }

  /** Strict check used before every state change. */
  function isValidMove(board, cell) {
    return Number.isInteger(cell) && cell >= 0 && cell < CELLS && board[cell] === EMPTY;
  }

  function emptyBoard() {
    return new Array(CELLS).fill(EMPTY);
  }

  // ------------------------------------------------------------- symmetry

  const rotate = (p) => [6, 3, 0, 7, 4, 1, 8, 5, 2].map((i) => p[i]);
  const mirror = (p) => [2, 1, 0, 5, 4, 3, 8, 7, 6].map((i) => p[i]);

  // PERMS[t][i] = original cell that lands on cell i after transform t
  const PERMS = (() => {
    const out = [];
    let p = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    for (let k = 0; k < 4; k++) {
      out.push(p);
      out.push(mirror(p));
      p = rotate(p);
    }
    return Object.freeze(out);
  })();
  // INV[t][a] = where original cell a ends up after transform t
  const INV = Object.freeze(PERMS.map((perm) => [...Array(CELLS).keys()].map((a) => perm.indexOf(a))));

  function lexLess(a, b) {
    for (let i = 0; i < CELLS; i++) if (a[i] !== b[i]) return a[i] < b[i];
    return false;
  }

  /** Smallest of the 8 symmetric images, plus which transform produced it. */
  function canonical(board) {
    let best = null;
    let bestT = 0;
    for (let t = 0; t < 8; t++) {
      const img = PERMS[t].map((i) => board[i]);
      if (best === null || lexLess(img, best)) {
        best = img;
        bestT = t;
      }
    }
    return { key: best.join(","), t: bestT };
  }

  // ------------------------------------------------------- data validation

  const KEY_PATTERN = /^(-1|0|1)(,(-1|0|1)){8}$/;

  /**
   * Validate the Q-table object before using it. Anything unexpected is
   * rejected instead of being trusted, so a corrupted or tampered data file
   * cannot inject markup/script or crash the page.
   * @returns {{ok: true, data: object} | {ok: false, error: string}}
   */
  function validateData(raw) {
    if (raw === null || typeof raw !== "object") return { ok: false, error: "Q-table data is missing." };
    const { meta, Q } = raw;
    if (!meta || typeof meta !== "object" || !Q || typeof Q !== "object") {
      return { ok: false, error: "Q-table data has the wrong structure." };
    }
    const states = Number(meta.states);
    const accuracy = Number(meta.accuracy);
    const gamma = Number(meta.gamma);
    if (![states, accuracy, gamma].every(Number.isFinite)) {
      return { ok: false, error: "Q-table metadata is invalid." };
    }

    // Copy into a prototype-less object so keys like "__proto__" can't do harm.
    const table = Object.create(null);
    for (const key of Object.keys(Q)) {
      const row = Q[key];
      if (!KEY_PATTERN.test(key)) return { ok: false, error: "Q-table contains an invalid state key." };
      if (!Array.isArray(row) || row.length !== CELLS || !row.every((v) => typeof v === "number" && Number.isFinite(v))) {
        return { ok: false, error: "Q-table contains an invalid row." };
      }
      table[key] = Object.freeze(row.slice());
    }
    return {
      ok: true,
      data: Object.freeze({
        meta: Object.freeze({ states, accuracy, gamma }),
        Q: Object.freeze(table),
      }),
    };
  }

  // ---------------------------------------------------------------- agent

  /** Q-value of every legal move, from the point of view of `player`. */
  function qValues(data, board, player) {
    const relative = board.map((c) => c * player); // mover's stones = +1
    const { key, t } = canonical(relative);
    const row = data.Q[key];
    const out = new Map();
    for (const a of legalMoves(board)) out.set(a, row ? row[INV[t][a]] : 0);
    return out;
  }

  /** Epsilon-greedy choice; ties between equally good moves are broken at random. */
  function chooseMove(data, board, player, epsilon) {
    const moves = legalMoves(board);
    if (moves.length === 0) return null;
    if (Math.random() < epsilon) {
      return { cell: moves[Math.floor(Math.random() * moves.length)], random: true };
    }
    const qs = qValues(data, board, player);
    const best = Math.max(...qs.values());
    const top = moves.filter((a) => qs.get(a) >= best - 1e-9);
    return { cell: top[Math.floor(Math.random() * top.length)], random: false };
  }

  window.TTT = window.TTT || {};
  window.TTT.engine = Object.freeze({
    EMPTY, X, O, CELLS,
    winnerInfo, legalMoves, isValidMove, emptyBoard,
    canonical, validateData, qValues, chooseMove,
  });
})();