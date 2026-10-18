 /*
 * app.js — user interface. Depends on engine.js and data/qtable.js.
 *
 * Safety rules followed in this file:
 *   - never use innerHTML / insertAdjacentHTML / document.write:
 *     all text goes through textContent, all elements through createElement
 *   - every value coming from the page (clicked cell, chosen side, slider)
 *     is parsed and checked against a whitelist / range before use
 *   - the Q-table is validated by engine.validateData() before use
 */
(function () {
  "use strict";

  const E = window.TTT.engine;
  const SYMBOL = new Map([[E.X, "X"], [E.O, "O"]]);
  const AI_DELAY_MS = 380;

  // ------------------------------------------------------------ DOM helpers

  const $ = (id) => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing element #${id}`);
    return el;
  };

  /** Create an element with optional class and text — never parses HTML. */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  const fmtQ = (q) => `${q >= 0 ? "+" : "−"}${Math.abs(q).toFixed(2)}`;

  // ------------------------------------------------------------------ state

  const state = {
    data: null,
    board: E.emptyBoard(),
    toMove: E.X,
    human: E.X,
    over: false,
    last: -1,
    busy: false,
    eps: 0,
    showQ: true,
    score: { you: 0, draw: 0, ai: 0 },
    timer: null,
  };

  const ui = {};

  // ------------------------------------------------------------- rendering

  function buildBoard() {
    ui.cells = [];
    for (let i = 0; i < E.CELLS; i++) {
      const btn = el("button", "cell");
      btn.type = "button";
      btn.dataset.cell = String(i);
      btn.setAttribute("aria-label", `cell ${i}`);
      ui.board.appendChild(btn);
      ui.cells.push(btn);
    }
  }

  function renderCell(btn, i, info, qs, best) {
    const v = state.board[i];
    btn.className = "cell";
    btn.classList.toggle("cell--x", v === E.X);
    btn.classList.toggle("cell--o", v === E.O);
    btn.classList.toggle("cell--last", i === state.last);
    btn.classList.toggle("cell--win", Boolean(info && info.line.includes(i)));
    btn.disabled = state.over || state.busy || v !== E.EMPTY || state.toMove !== state.human;
    btn.style.removeProperty("--q-strength");
    btn.replaceChildren();

    if (v !== E.EMPTY) {
      btn.textContent = SYMBOL.get(v);
      return;
    }
    if (qs && qs.has(i)) {
      const q = qs.get(i);
      btn.classList.add(q >= 0 ? "cell--good" : "cell--bad");
      btn.style.setProperty("--q-strength", (0.12 + 0.55 * Math.min(1, Math.abs(q))).toFixed(2));
      const label = el("span", "q", fmtQ(q));
      if (Math.abs(q - best) < 1e-9) label.classList.add("q--best");
      btn.appendChild(label);
    }
  }

  function render() {
    const info = E.winnerInfo(state.board);
    const qs = !state.over && state.showQ ? E.qValues(state.data, state.board, state.toMove) : null;
    const best = qs ? Math.max(...qs.values()) : null;
    ui.cells.forEach((btn, i) => renderCell(btn, i, info, qs, best));

    let status;
    if (info) status = info.w === 0 ? "Draw." : info.w === state.human ? "You win! 🎉" : "AI wins.";
    else status = state.toMove === state.human ? `Your move (${SYMBOL.get(state.human)})` : "AI is thinking…";
    ui.status.textContent = status;

    ui.legend.textContent = !qs ? "" : state.toMove === state.human
      ? "Values are from YOUR side. ★ = the move the agent thinks is best for you."
      : "Values are from the AI's side.";

    ui.scoreYou.textContent = String(state.score.you);
    ui.scoreDraw.textContent = String(state.score.draw);
    ui.scoreAI.textContent = String(state.score.ai);
  }

  function renderAIDecision(qs, chosen, wasRandom) {
    const rows = [...qs.entries()].sort((a, b) => b[1] - a[1]).map(([cell, q]) => {
      const row = el("div", "bar");
      const name = el(cell === chosen ? "strong" : "span", null, `cell ${cell}`);
      const track = el("div", "bar__track");
      const fill = el("div", `bar__fill ${q >= 0 ? "bar__fill--good" : "bar__fill--bad"}`);
      const width = Math.min(1, Math.abs(q)) * 50;
      fill.style.left = `${q >= 0 ? 50 : 50 - width}%`;
      fill.style.width = `${width}%`;
      track.append(el("div", "bar__mid"), fill);
      row.append(name, track, el("span", "bar__value", fmtQ(q)));
      return row;
    });
    if (wasRandom) rows.push(el("p", "muted", `Played cell ${chosen} at random (exploration).`));
    ui.aiBars.replaceChildren(...rows);
  }

  function renderMeta() {
    const { states, accuracy, gamma } = state.data.meta;
    const items = [
      ["Table rows (canonical states)", String(states)],
      ["Optimal-move accuracy", `${(accuracy * 100).toFixed(1)}%`],
      ["Discount γ", String(gamma)],
    ];
    ui.meta.replaceChildren(...items.flatMap(([k, v]) => [el("dt", null, k), el("dd", null, v)]));
  }

  function showError(message) {
    ui.status.replaceChildren(el("span", "error", message));
    ui.cells.forEach((b) => { b.disabled = true; });
  }

  // ------------------------------------------------------------ game logic

  function finishIfOver() {
    const info = E.winnerInfo(state.board);
    if (!info) return false;
    state.over = true;
    if (info.w === 0) state.score.draw++;
    else if (info.w === state.human) state.score.you++;
    else state.score.ai++;
    return true;
  }

  function place(cell, player) {
    if (!E.isValidMove(state.board, cell)) return false;
    state.board[cell] = player;
    state.last = cell;
    state.toMove = -player;
    return true;
  }

  function aiTurn() {
    if (state.over || state.toMove === state.human) return;
    state.busy = true;
    render();
    state.timer = setTimeout(() => {
      state.timer = null;
      const player = state.toMove;
      const qs = E.qValues(state.data, state.board, player);
      const move = E.chooseMove(state.data, state.board, player, state.eps);
      if (move && place(move.cell, player)) {
        renderAIDecision(qs, move.cell, move.random);
        finishIfOver();
      }
      state.busy = false;
      render();
    }, AI_DELAY_MS);
  }

  function onBoardClick(event) {
    const btn = event.target.closest("button.cell");
    if (!btn || !ui.board.contains(btn)) return;
    const cell = Number.parseInt(btn.dataset.cell, 10);
    if (state.over || state.busy || state.toMove !== state.human) return;
    if (!place(cell, state.human)) return;
    if (!finishIfOver()) aiTurn();
    render();
  }

  function newGame() {
    if (state.timer !== null) clearTimeout(state.timer);
    Object.assign(state, {
      board: E.emptyBoard(), toMove: E.X, over: false, last: -1, busy: false, timer: null,
    });
    ui.aiBars.replaceChildren(el("p", "muted", "The AI hasn't moved yet."));
    render();
    aiTurn();
  }

  function onSideClick(event) {
    const btn = event.target.closest("button[data-side]");
    if (!btn) return;
    const side = Number.parseInt(btn.dataset.side, 10);
    if (side !== E.X && side !== E.O) return; // whitelist
    state.human = side;
    for (const b of ui.side.querySelectorAll("button[data-side]")) {
      const on = b === btn;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-pressed", String(on));
    }
    newGame();
  }

  function onEpsilonInput() {
    const pct = Number.parseInt(ui.eps.value, 10);
    const safe = Number.isFinite(pct) ? Math.min(100, Math.max(0, pct)) : 0;
    state.eps = safe / 100;
    ui.epsValue.textContent = `${safe}%`;
  }

  // ------------------------------------------------------------------ init

  function init() {
    Object.assign(ui, {
      board: $("board"), status: $("status"), legend: $("legend"), side: $("side"),
      newGame: $("new-game"), showQ: $("show-q"), eps: $("eps"), epsValue: $("eps-value"),
      scoreYou: $("score-you"), scoreDraw: $("score-draw"), scoreAI: $("score-ai"),
      aiBars: $("ai-bars"), meta: $("meta"),
    });
    buildBoard();

    const result = E.validateData(window.TTT_QTABLE);
    if (!result.ok) {
      showError(`${result.error} Run "python -m rl export" to (re)create web/data/qtable.js.`);
      return;
    }
    state.data = result.data;
    renderMeta();

    ui.board.addEventListener("click", onBoardClick);
    ui.side.addEventListener("click", onSideClick);
    ui.newGame.addEventListener("click", newGame);
    ui.showQ.addEventListener("change", () => { state.showQ = ui.showQ.checked; render(); });
    ui.eps.addEventListener("input", onEpsilonInput);
    onEpsilonInput();
    newGame();
  }

  // exposed for automated tests only
  window.TTT.qValuesForTest = (board, player) =>
    state.data ? Object.fromEntries(E.qValues(state.data, board, player)) : null;

  document.addEventListener("DOMContentLoaded", init);
})();