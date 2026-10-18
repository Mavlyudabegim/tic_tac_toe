"""Self-play training loop with epsilon / learning-rate schedules."""
from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass

from .agents import MinimaxAgent, QAgent, RandomAgent
from .evaluate import match, optimal_move_accuracy
from .game import EMPTY_BOARD, play, winner


@dataclass
class TrainConfig:
    episodes: int = 50_000
    alpha_start: float = 0.5
    alpha_end: float = 0.05
    gamma: float = 0.9
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7      # epsilon reaches eps_end after this share of training
    use_symmetry: bool = True
    backward: bool = True            # update moves in reverse order after each game
    eval_every: int = 2_500
    eval_games: int = 200
    seed: int = 0


def _linear(start: float, end: float, frac: float) -> float:
    return start + (end - start) * min(max(frac, 0.0), 1.0)


def self_play_episode(agent: QAgent, epsilon: float, alpha: float, backward: bool = True) -> int:
    """One game where the agent plays both sides.

    backward=False: update right after every move (classic online Q-learning).
    backward=True : store the game, then update from the LAST move to the first,
                    so the final reward flows back through the whole game at once.
    The update is off-policy (target uses max), so exploratory moves don't
    poison the values.
    """
    board, player, moves = EMPTY_BOARD, 1, []
    while True:
        a = agent.act(board, player, epsilon)
        if backward:
            moves.append((board, player, a))
        else:
            agent.update(board, player, a, alpha)
        board = play(board, a, player)
        w = winner(board)
        if w is not None:
            break
        player = -player
    for b, p, a in reversed(moves):
        agent.update(b, p, a, alpha)
    return w


def train(cfg: TrainConfig, verbose: bool = True) -> tuple[QAgent, list[dict]]:
    random.seed(cfg.seed)
    agent = QAgent(cfg.alpha_start, cfg.gamma, cfg.use_symmetry, seed=cfg.seed)
    rnd, perfect = RandomAgent(random.Random(cfg.seed + 1)), MinimaxAgent(random.Random(cfg.seed + 2))
    history: list[dict] = []
    t0 = time.time()

    def evaluate(ep: int, eps: float, alpha: float):
        row = {
            "episode": ep, "epsilon": round(eps, 4), "alpha": round(alpha, 4),
            "states": len(agent.Q),
            "accuracy": optimal_move_accuracy(agent),
        }
        for name, opp in (("random", rnd), ("minimax", perfect)):
            for side, lab in ((1, "X"), (-1, "O")):
                r = match(agent, opp, cfg.eval_games, side)
                row[f"{name}_{lab}_win"] = r["win"] / cfg.eval_games
                row[f"{name}_{lab}_loss"] = r["loss"] / cfg.eval_games
        history.append(row)
        if verbose:
            print(f"ep {ep:>7,} | eps {eps:.3f} | acc {row['accuracy']:.3f} | "
                  f"vs random  X win {row['random_X_win']:.0%} loss {row['random_X_loss']:.0%}, "
                  f"O win {row['random_O_win']:.0%} loss {row['random_O_loss']:.0%} | "
                  f"vs minimax loss X {row['minimax_X_loss']:.0%} O {row['minimax_O_loss']:.0%} | "
                  f"{time.time() - t0:.0f}s")

    evaluate(0, cfg.eps_start, cfg.alpha_start)
    for ep in range(1, cfg.episodes + 1):
        frac = ep / cfg.episodes
        eps = _linear(cfg.eps_start, cfg.eps_end, frac / cfg.eps_decay_frac)
        alpha = _linear(cfg.alpha_start, cfg.alpha_end, frac)
        self_play_episode(agent, eps, alpha, cfg.backward)
        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            evaluate(ep, eps, alpha)
    return agent, history


def save_history(history: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)