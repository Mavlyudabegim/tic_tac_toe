"""Tic-Tac-Toe rules.

A board is a tuple of 9 ints:  1 = X,  -1 = O,  0 = empty.
Cells are indexed

    0 | 1 | 2
    3 | 4 | 5
    6 | 7 | 8

X (=1) always moves first.
"""
from __future__ import annotations

from typing import Optional, Tuple

Board = Tuple[int, ...]

EMPTY_BOARD: Board = (0,) * 9

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


def winner(board: Board) -> Optional[int]:
    """Return 1 (X wins), -1 (O wins), 0 (draw) or None (game still running)."""
    for a, b, c in LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0 if 0 not in board else None


def legal_moves(board: Board) -> list[int]:
    return [i for i, c in enumerate(board) if c == 0]


def to_move(board: Board) -> int:
    """Whose turn it is (1 or -1), derived from the piece count."""
    return 1 if sum(1 for c in board if c != 0) % 2 == 0 else -1


def play(board: Board, action: int, player: int) -> Board:
    if board[action] != 0:
        raise ValueError(f"cell {action} is already taken")
    b = list(board)
    b[action] = player
    return tuple(b)


def render(board: Board, show_indices: bool = True) -> str:
    sym = {1: "X", -1: "O"}
    cells = [sym.get(c, str(i) if show_indices else " ") for i, c in enumerate(board)]
    rows = [" " + " | ".join(cells[r * 3:(r + 1) * 3]) for r in range(3)]
    return "\n---+---+---\n".join(rows)


def all_reachable_states() -> set[Board]:
    """Every position reachable from the empty board under legal play (5,478)."""
    seen: set[Board] = set()
    stack = [EMPTY_BOARD]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        if winner(b) is None:
            p = to_move(b)
            stack.extend(play(b, a, p) for a in legal_moves(b))
    return seen