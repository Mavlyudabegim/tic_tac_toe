"""Command line:  python -m rl {train,evaluate,play} ..."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .agents import MinimaxAgent, QAgent, RandomAgent
from .evaluate import full_report
from .game import EMPTY_BOARD, legal_moves, play, render, winner
from .train import TrainConfig, save_history, train


def cmd_train(args):
    cfg = TrainConfig(episodes=args.episodes, gamma=args.gamma, use_symmetry=not args.no_symmetry,
                      backward=not args.online, eval_every=args.eval_every, seed=args.seed)
    agent, history = train(cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agent.save(out / "qtable.pkl")
    save_history(history, out / "history.csv")
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    try:
        from .plots import plot_history
        plot_history(history, out / "learning_curve.png")
    except ImportError:
        pass
    print(f"\nSaved model and logs to {out}/ ({len(agent.Q)} canonical states in Q-table)")


    # Keep the web page in sync with the newly trained model.
    if args.no_export:
        return
    if not cfg.use_symmetry:
        print("Skipped web export: the web UI needs a model trained with symmetry.")
        return
    from .export import export_web
    export_web(out / "qtable.pkl", args.web_out)
    print(f"Updated {args.web_out} - refresh web/index.html to play against this model.")


def cmd_evaluate(args):
    agent = QAgent.load(args.model)
    rep = full_report(agent, {"random": RandomAgent(random.Random(1)),
                              "minimax": MinimaxAgent(random.Random(2))}, args.games)
    print(f"Optimal-move accuracy: {rep.pop('optimal_move_accuracy'):.2%}  (all 4,520 positions)")
    for k, r in rep.items():
        print(f"{k:<20} win {r['win']:>4}  draw {r['draw']:>4}  loss {r['loss']:>4}")


def cmd_play(args):
    agent = QAgent.load(args.model)
    human = 1 if args.first else -1
    board, player = EMPTY_BOARD, 1
    print("You are", "X (first)" if human == 1 else "O (second)", "— enter a cell number 0-8.\n")
    while winner(board) is None:
        if player == human:
            print(render(board))
            try:
                a = int(input("\nYour move: "))
            except (ValueError, EOFError):
                continue
            if a not in legal_moves(board):
                print("Illegal move.")
                continue
        else:
            qs = agent.q_values(board, player)
            a = agent.act(board, player)
            print("AI Q-values:", {k: round(v, 2) for k, v in sorted(qs.items())}, "-> plays", a)
        board = play(board, a, player)
        player = -player
    print(render(board, show_indices=False))
    w = winner(board)
    print("\nDraw!" if w == 0 else "\nYou win!" if w == human else "\nAI wins!")

def cmd_export(args):
    from .export import export_web
    meta = export_web(args.model, args.out)
    print(f"Wrote {args.out}  ({meta['states']} states, optimal-move accuracy {meta['accuracy']:.2%})")
    print("Open web/index.html in your browser to play.")

def main(argv=None):
    p = argparse.ArgumentParser(prog="rl", description="Tic-Tac-Toe Q-learning")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train by self-play")
    t.add_argument("--episodes", type=int, default=50_000)
    t.add_argument("--gamma", type=float, default=0.9)
    t.add_argument("--eval-every", type=int, default=2_500)
    t.add_argument("--no-symmetry", action="store_true", help="ablation: learn every rotation separately")
    t.add_argument("--online", action="store_true", help="ablation: update after each move instead of backward after the game")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out-dir", default="runs/default")
    t.add_argument("--web-out", default="web/data/qtable.js", help="where to write the web UI data")
    t.add_argument("--no-export", action="store_true", help="do not update the web UI data after training")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved model")
    e.add_argument("--model", default="runs/default/qtable.pkl")
    e.add_argument("--games", type=int, default=1000)
    e.set_defaults(func=cmd_evaluate)

    g = sub.add_parser("play", help="play against a saved model in the terminal")
    g.add_argument("--model", default="runs/default/qtable.pkl")
    g.add_argument("--first", action="store_true", help="you play X and move first")
    g.set_defaults(func=cmd_play)

    x = sub.add_parser("export", help="write the Q-table for the web UI (web/data/qtable.js)")
    x.add_argument("--model", default="runs/default/qtable.pkl")
    x.add_argument("--out", default="web/data/qtable.js")
    x.set_defaults(func=cmd_export)
    
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()