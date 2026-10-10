"""Builds the loss-curve and fixed-noise evolution figures for one or more runs.

Usage: python src/plot_run.py base [hinge sn ...]
Writes artifacts/figures/<run>_losses.png and <run>_evolution.png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
FIGS = ROOT / "artifacts" / "figures"
SMOOTH = 50  # iterations in the rolling mean; raw curves are too noisy to read


def plot_losses(name: str) -> None:
    df = pd.read_csv(RUNS / name / "losses.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for col, label in [("loss_d", "Pérdida D"), ("loss_g", "Pérdida G")]:
        ax1.plot(df["iter"], df[col], alpha=0.15)
        ax1.plot(df["iter"], df[col].rolling(SMOOTH).mean(), label=f"{label} (media móvil {SMOOTH})",
                 color=ax1.lines[-1].get_color())
    ax1.set_ylabel("pérdida")
    ax1.legend()
    ax1.set_title(f"Corrida '{name}'")
    ax2.plot(df["iter"], df["d_real"].rolling(SMOOTH).mean(), label="D(x) real")
    ax2.plot(df["iter"], df["d_fake"].rolling(SMOOTH).mean(), label="D(G(z)) falso")
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("prob. media de D")
    ax2.set_xlabel("iteración")
    ax2.legend()
    # Epoch ticks on top make it easy to match curves with snapshots.
    per_epoch = df.groupby("epoch")["iter"].max()
    top = ax1.secondary_xaxis("top")
    marks = [e for e in per_epoch.index if e in (1, 25, 50, 100, 150, 200, 300)]
    top.set_xticks([per_epoch[e] for e in marks], [f"ép. {e}" for e in marks])
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}_losses.png", dpi=130)
    plt.close(fig)


def plot_evolution(name: str, max_cols: int = 8) -> None:
    snaps = sorted((RUNS / name / "snapshots").glob("epoch_*.png"))
    if len(snaps) > max_cols:
        # Keep first and last, spread the rest evenly.
        idx = [round(i * (len(snaps) - 1) / (max_cols - 1)) for i in range(max_cols)]
        snaps = [snaps[i] for i in idx]
    fig, axes = plt.subplots(1, len(snaps), figsize=(2.2 * len(snaps), 2.6))
    for ax, path in zip(axes, snaps):
        ax.imshow(Image.open(path))
        ax.set_title(f"época {int(path.stem.split('_')[1])}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Rejilla de ruido fijo (mismos 16 z) — '{name}'")
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}_evolution.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    for run in sys.argv[1:]:
        plot_losses(run)
        plot_evolution(run)
        print(f"Figures for '{run}' in {FIGS}")
