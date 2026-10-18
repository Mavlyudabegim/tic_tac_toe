"""Honest evaluation: play vs random and vs a perfect player, and check every position."""
from __future__ import annotations

from functools import lru_cache

from .game import Board, EMPTY_BOARD, play, to_move, winner
from .minimax import optimal_moves


class _Greedy:
    """Wrap a QAgent so it always plays greedily (epsilon = 0)."""
    def __init__(self, agent):
        self.agent = agent

    def act(self, board, player):
        return self.agent.act(board, player, epsilon=0.0)


def play_game(x_player, o_player) -> int:
    """Return 1 if X wins, -1 if O wins, 0 for a draw."""
    board, player = EMPTY_BOARD, 1
    while True:
        mover = x_player if player == 1 else o_player
        board = play(board, mover.act(board, player), player)
        w = winner(board)
        if w is not None:
            return w
        player = -player


def match(agent, opponent, n_games: int, agent_side: int) -> dict[str, int]:
    """Play n games with the (greedy) Q-agent on `agent_side` (1=X, -1=O)."""
    me = _Greedy(agent) if hasattr(agent, "Q") else agent
    res = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(n_games):
        w = play_game(me, opponent) if agent_side == 1 else play_game(opponent, me)
        res["draw" if w == 0 else "win" if w == agent_side else "loss"] += 1
    return res


@lru_cache(maxsize=1)
def _decision_states() -> tuple[Board, ...]:
    from .game import all_reachable_states
    return tuple(sorted(b for b in all_reachable_states() if winner(b) is None))


def optimal_move_accuracy(agent) -> float:
    """Share of all 4,520 non-terminal positions where *every* move the agent
    would pick greedily is game-theoretically optimal. 1.0 = perfect policy."""
    states = _decision_states()
    good = 0
    for b in states:
        best = set(optimal_moves(b))
        if set(agent.greedy_moves(b, to_move(b))) <= best:
            good += 1
    return good / len(states)


def full_report(agent, opponents: dict, n_games: int = 500) -> dict:
    report = {"optimal_move_accuracy": optimal_move_accuracy(agent)}
    for name, opp in opponents.items():
        for side, label in ((1, "X"), (-1, "O")):
            report[f"vs_{name}_as_{label}"] = match(agent, opp, n_games, side)
    return report