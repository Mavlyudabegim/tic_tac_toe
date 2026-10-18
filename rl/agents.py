"""Players: tabular Q-learning agent, random player, perfect minimax player."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

from .game import Board, legal_moves, play, winner
from .minimax import optimal_moves
from .symmetry import action_to_canonical, canonical


class RandomAgent:
    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def act(self, board: Board, player: int) -> int:
        return self.rng.choice(legal_moves(board))


class MinimaxAgent:
    """Perfect player; picks uniformly among optimal moves so games vary."""
    name = "minimax"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def act(self, board: Board, player: int) -> int:
        return self.rng.choice(optimal_moves(board))


class QAgent:
    """Tabular Q-learning for a two-player zero-sum game (negamax form).

    * Mover's perspective: board * player, so "my pieces" are always +1.
      One table plays both X and O.
    * Negamax target: after I move it's my opponent's turn, and their best
      value is the negative of mine:
          r                              if the game ended (win=+1, draw=0)
          -gamma * max_a' Q(s', a')      otherwise (s' = opponent's view)
    * Symmetry: states reduced to one of 8 rotations/reflections.
    * Draw = 0: a positive draw reward makes the agent settle for draws.
    """
    name = "q-learning"

    def __init__(self, alpha: float = 0.5, gamma: float = 0.9,
                 use_symmetry: bool = True, seed: int | None = None):
        self.alpha = alpha
        self.gamma = gamma
        self.use_symmetry = use_symmetry
        self.Q: dict[Board, list[float]] = {}
        self.rng = random.Random(seed)

    # ---------- table access ----------
    def _key(self, board: Board, player: int) -> tuple[Board, int]:
        rel = tuple(c * player for c in board)        # mover's pieces = +1
        return canonical(rel, self.use_symmetry)

    def _row(self, key: Board) -> list[float]:
        row = self.Q.get(key)
        if row is None:
            row = self.Q[key] = [0.0] * 9
        return row

    def q_values(self, board: Board, player: int) -> dict[int, float]:
        """Q-value of every legal move, in the *original* board's coordinates."""
        key, t = self._key(board, player)
        row = self.Q.get(key, [0.0] * 9)
        return {a: row[action_to_canonical(a, t)] for a in legal_moves(board)}

    def max_q(self, board: Board, player: int) -> float:
        return max(self.q_values(board, player).values())

    # ---------- acting ----------
    def greedy_moves(self, board: Board, player: int, tol: float = 1e-9) -> list[int]:
        qs = self.q_values(board, player)
        best = max(qs.values())
        return [a for a, q in qs.items() if q >= best - tol]

    def act(self, board: Board, player: int, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return self.rng.choice(legal_moves(board))
        return self.rng.choice(self.greedy_moves(board, player))

    # ---------- learning ----------
    def update(self, board: Board, player: int, action: int, alpha: float | None = None) -> float:
        """One Q-learning step for `player` playing `action` on `board`. Returns |TD error|."""
        alpha = self.alpha if alpha is None else alpha
        nxt = play(board, action, player)
        w = winner(nxt)
        if w is not None:
            target = 1.0 if w == player else 0.0   # a move can only win or draw
        else:
            target = -self.gamma * self.max_q(nxt, -player)

        key, t = self._key(board, player)
        row = self._row(key)
        ca = action_to_canonical(action, t)
        td = target - row[ca]
        row[ca] += alpha * td
        return abs(td)

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "alpha": self.alpha, "gamma": self.gamma,
                         "use_symmetry": self.use_symmetry}, f)

    @classmethod
    def load(cls, path: str | Path) -> "QAgent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        agent = cls(d["alpha"], d["gamma"], d["use_symmetry"])
        agent.Q = d["Q"]
        return agent