"""Exact game-theoretic values via memoised negamax.

Used only as a *referee* for evaluation (and as a perfect sparring partner);
the Q-learning agent never sees it during training.
"""
from __future__ import annotations

from functools import lru_cache

from .game import Board, legal_moves, play, to_move, winner


@lru_cache(maxsize=None)
def value(board: Board) -> int:
    """Value for the player to move: +1 win, 0 draw, -1 loss (perfect play)."""
    w = winner(board)
    if w is not None:
        # the game is over, so the player "to move" did not make the last move
        return 0 if w == 0 else -1
    p = to_move(board)
    return max(-value(play(board, a, p)) for a in legal_moves(board))


@lru_cache(maxsize=None)
def optimal_moves(board: Board) -> tuple[int, ...]:
    """All moves that keep the best achievable outcome for the player to move."""
    p = to_move(board)
    scores = {a: -value(play(board, a, p)) for a in legal_moves(board)}
    best = max(scores.values())
    return tuple(a for a, s in scores.items() if s == best)