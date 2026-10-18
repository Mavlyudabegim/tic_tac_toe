"""Q-learning WITH vs WITHOUT symmetry: same settings, several seeds, one plot.

Run from the project folder:
    python experiments/compare_symmetry.py
Output:
    results/symmetry_comparison.png
    results/symmetry_comparison.csv
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.train import TrainConfig, train  # noqa: E402

EPISODES = 60_000
EVAL_EVERY = 2_000
SEEDS = [0, 1, 2]

runs = {}
for use_sym in (True, False):
    label = "with symmetry" if use_sym else "without symmetry"
    runs[label] = []
    for seed in SEEDS:
        print(f"training {label}, seed {seed} ...")
        cfg = TrainConfig(episodes=EPISODES, eval_every=EVAL_EVERY, eval_games=200,
                          use_symmetry=use_sym, seed=seed)
        agent, hist = train(cfg, verbose=False)
        runs[label].append(hist)
        print(f"   final accuracy {hist[-1]['accuracy']:.3f}, table rows {len(agent.Q)}")


def mean_curve(hists, key):
    return [statistics.mean(h[i][key] for h in hists) for i in range(len(hists[0]))]


def first_episode_reaching(hist, key, level):
    for row in hist:
        if row[key] >= level:
            return row["episode"]
    return None


# ---------- plot ----------
colors = {"with symmetry": "#2563eb", "without symmetry": "#ea580c"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
panels = [
    ("accuracy", "Optimal-move accuracy", None),
    ("minimax_loss", "Loss rate vs perfect player (avg of X and O)", None),
    ("states", "Q-table rows", None),
]
for label, hists in runs.items():
    ep = [r["episode"] for r in hists[0]]
    for h in hists:
        for r in h:
            r["minimax_loss"] = (r["minimax_X_loss"] + r["minimax_O_loss"]) / 2
    for ax, (key, title, _) in zip(axes, panels):
        for h in hists:  # each seed, faint
            ax.plot(ep, [r[key] for r in h], color=colors[label], alpha=0.2, lw=1)
        ax.plot(ep, mean_curve(hists, key), color=colors[label], lw=2.5, label=label)
        ax.set_title(title)
for ax in axes:
    ax.set_xlabel("training games")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(0, 1.02)
axes[1].set_ylim(-0.02, 1.02)
axes[0].legend(frameon=False)
fig.suptitle(f"Q-learning with vs without symmetry ({len(SEEDS)} seeds; thick = mean)", fontsize=13)
fig.tight_layout()
(ROOT / "results").mkdir(exist_ok=True)
fig.savefig(ROOT / "results" / "symmetry_comparison.png", dpi=130)

# ---------- summary table ----------
print("\n" + f"{'':<18}{'games to 99% acc':>18}{'games to 0 losses':>20}{'final acc':>11}{'rows':>7}")
with open(ROOT / "results" / "symmetry_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "seed", "games_to_99pct_accuracy", "games_to_zero_losses", "final_accuracy", "table_rows"])
    for label, hists in runs.items():
        g99 = [first_episode_reaching(h, "accuracy", 0.99) for h in hists]
        g0 = [next((r["episode"] for r in h if r["minimax_loss"] == 0), None) for h in hists]
        for s, h, a, b in zip(SEEDS, hists, g99, g0):
            w.writerow([label, s, a, b, h[-1]["accuracy"], h[-1]["states"]])
        avg = lambda xs: f"{statistics.mean(x for x in xs if x is not None):,.0f}"
        print(f"{label:<18}{avg(g99):>18}{avg(g0):>20}{statistics.mean(h[-1]['accuracy'] for h in hists):>11.3f}"
              f"{hists[0][-1]['states']:>7}")
print(f"\nSaved {ROOT / 'results' / 'symmetry_comparison.png'}")