"""Plot every held-out seed and paired differences from saved replication data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("momentum", "momentum_restart", "momentum_damp", "momentum_periodic", "adamw")
LABELS = ("Momentum", "+ restart", "+ state damping", "+ periodic damping", "AdamW")
COLORS = ("#626b73", "#9872ad", "#248a9c", "#d89426", "#293f69")


def main():
    output = ROOT/"output/damping_replication"
    runs = json.loads((output/"runs.json").read_text())
    paired = json.loads((output/"paired.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        group = sorted([r for r in runs if r["method"] == method], key=lambda r: r["seed"])
        values = [r["final_loss"] for r in group]
        ax.scatter(i+np.linspace(-.13, .13, len(group)), values, color=color, s=30, alpha=.8)
        ax.plot([i-.25, i+.25], [np.mean(values)]*2, color=color, linewidth=2.8)
    ax.set_xticks(range(5), LABELS, rotation=30, ha="right")
    ax.set_ylabel("Final test cross-entropy (lower is better)")
    ax.set_title("Ten seeds; dots = runs, line = mean")
    for i, reference in enumerate(("momentum", "momentum_periodic")):
        row = next(p for p in paired if p["method"] == "momentum_damp" and p["reference"] == reference)
        color = (COLORS[2], COLORS[3])[i]
        values = [p["difference"] for p in row["seed_differences"]]
        bx.scatter(values, i+np.linspace(-.12, .12, len(values)), color=color, alpha=.6, s=27)
        bx.plot([row["bootstrap_95_low"], row["bootstrap_95_high"]], [i, i], color=color, linewidth=3)
        bx.scatter([row["mean_difference"]], [i], marker="D", color=color, s=70, zorder=5)
    bx.axvline(0, color="#333333", linestyle="--", linewidth=1)
    bx.set_yticks((0, 1), ("State damping\nminus Momentum", "State damping\nminus periodic"))
    bx.set_ylim(-.6, 1.6)
    bx.set_xlabel("Paired loss difference; negative favors state damping")
    bx.set_title("Mean and paired bootstrap 95% interval")
    fig.suptitle("Damping replication: seeds 400--409, 400 real updates each", fontsize=14)
    fig.savefig(output/"damping_replication.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
