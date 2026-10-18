"""Board symmetries (the 8 rotations/reflections of the square).

Using them collapses ~4,500 non-terminal positions into ~630 canonical ones,
so every game teaches the agent about up to 8 positions at once.
"""
from __future__ import annotations

from .game import Board


def _rot(p):   # rotate 90 degrees clockwise: new[i] = old[_rot[i]]
    return [p[i] for i in (6, 3, 0, 7, 4, 1, 8, 5, 2)]


def _flip(p):  # mirror left-right
    return [p[i] for i in (2, 1, 0, 5, 4, 3, 8, 7, 6)]


def _build():
    ident = list(range(9))
    perms, p = [], ident
    for _ in range(4):
        perms.append(p)
        perms.append(_flip(p))
        p = _rot(p)
    return [tuple(x) for x in perms]


# PERMS[t][i] = index in the original board that lands on cell i after transform t
PERMS: list[tuple[int, ...]] = _build()
# INV[t][a] = where original cell a ends up after transform t
INV: list[tuple[int, ...]] = [
    tuple(perm.index(a) for a in range(9)) for perm in PERMS
]


def transform(board: Board, t: int) -> Board:
    return tuple(board[i] for i in PERMS[t])


def canonical(board: Board, use_symmetry: bool = True) -> tuple[Board, int]:
    """Return (canonical board, transform id) — the lexicographically smallest image."""
    if not use_symmetry:
        return board, 0
    best, best_t = None, 0
    for t in range(8):
        b = transform(board, t)
        if best is None or b < best:
            best, best_t = b, t
    return best, best_t


def action_to_canonical(action: int, t: int) -> int:
    return INV[t][action]


def action_from_canonical(action: int, t: int) -> int:
    return PERMS[t][action]