"""Learning-curve figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(history: list[dict], path, title: str = "Q-learning self-play") -> None:
    ep = [h["episode"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ep, [h["accuracy"] for h in history], color="#2563eb", lw=2)
    ax.set_title("Optimal-move accuracy (all 4,520 positions)")
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.plot(ep, [h["random_X_win"] for h in history], label="win as X", color="#16a34a", lw=2)
    ax.plot(ep, [h["random_O_win"] for h in history], label="win as O", color="#16a34a", lw=2, ls="--")
    ax.plot(ep, [h["random_X_loss"] for h in history], label="loss as X", color="#dc2626", lw=2)
    ax.plot(ep, [h["random_O_loss"] for h in history], label="loss as O", color="#dc2626", lw=2, ls="--")
    ax.set_title("vs Random player")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.plot(ep, [h["minimax_X_loss"] for h in history], label="loss as X", color="#dc2626", lw=2)
    ax.plot(ep, [h["minimax_O_loss"] for h in history], label="loss as O", color="#dc2626", lw=2, ls="--")
    ax.set_title("vs Perfect (minimax) player — losses")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.set_xlabel("training games")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)