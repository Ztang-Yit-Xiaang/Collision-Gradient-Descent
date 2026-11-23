"""Rebuild diagnostic figures from saved results without rerunning experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("momentum", "momentum_restart", "momentum_damp", "momentum_random", "momentum_adaptive")
LABELS = ("Momentum", "+ restart", "+ damping", "+ random pulse", "+ adaptive")
COLORS = ("#626b73", "#9872ad", "#248a9c", "#d89426", "#c75c52")


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "bold"})
    path = ROOT/"output/momentum_routing"
    aggregate = json.loads((path/"aggregate.json").read_text())
    cases = ("bad_well", "good_well", "saddle", "quadratic10", "rosenbrock10", "wells20")
    labels = ("Bad well", "Good well", "Exact saddle", "Quadratic 10D", "Rosenbrock 10D", "Wells 20D")
    methods = METHODS+("adam",)
    table = {(r["case"], r["method"]): r for r in aggregate}
    values = np.array([[table[c, m]["median_relative_final_gap"] for c in cases] for m in methods])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 8.5), gridspec_kw={"height_ratios": (1.45, 1)}, layout="constrained")
    im = ax.imshow(np.log10(np.maximum(values, 1e-12)), cmap="viridis", vmin=-12, vmax=3, aspect="auto")
    ax.set_xticks(range(6), labels); ax.set_yticks(range(6), LABELS+("Adam (wd=0)",))
    ax.set_title("Final relative loss gap: median of 10 seeds (lower is better)")
    for i in range(6):
        for j in range(6):
            label = "<1e-12" if values[i, j] < 1e-12 else f"{values[i,j]:.3g}"
            ax.text(j, i, label, ha="center", va="center", color="white" if values[i, j] < .01 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10 gap (display floor 1e-12)", shrink=.85)
    x = np.arange(3); width = .25
    for i, (method, color) in enumerate(zip(("momentum", "momentum_random", "momentum_adaptive"), (COLORS[0], COLORS[3], COLORS[4]))):
        group = [table[c, method] for c in cases[:3]]
        positions = x+(i-1)*width
        bx.bar(positions, [r["target_hits"] for r in group], width, color=color, alpha=.25)
        bx.bar(positions, [r["final_target_hits"] for r in group], width, color=color, label=method.replace("momentum_", "+ ").title())
    bx.set_xticks(x, labels[:3]); bx.set_yticks(range(0, 11, 2)); bx.set_ylim(0, 11)
    bx.set_ylabel("Successful runs / 10"); bx.legend(ncols=3, frameon=False)
    bx.set_title("Solid = final success; faded = reached target at any time")
    fig.suptitle("Equal 1,500 actual queries; no virtual trajectories or extra probes", fontsize=14)
    fig.savefig(path/"synthetic_routing.png", dpi=180); plt.close(fig)

    path = ROOT/"output/nanogpt_routing"
    runs = json.loads((path/"runs.json").read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": (1.6, 1)}, layout="constrained")
    methods = METHODS+("adamw",); labels = LABELS+("AdamW",); colors = COLORS+("#293f69",)
    for i, (method, color) in enumerate(zip(methods, colors)):
        group = [r for r in runs if r["method"] == method]
        y = [r["final_loss"] for r in group]
        ax.scatter(i+np.linspace(-.1, .1, len(y)), y, color=color, s=35, zorder=3)
        ax.plot([i-.23, i+.23], [np.mean(y)]*2, color=color, linewidth=2.5)
        bx.bar(i, sum(r["dampings"] for r in group), color=color)
    for panel in (ax, bx):
        panel.set_xticks(range(6), labels, rotation=35, ha="right")
    ax.set_ylabel("Final held-out cross-entropy"); ax.set_title("Dots: seeds 300--302; line: mean")
    bx.set_ylabel("Total actions over three seeds"); bx.set_title("Damping/restart activated; pulses = 0")
    fig.suptitle("NanoGPT: 400 real updates for every method; no diagnostic forwards", fontsize=13)
    fig.savefig(path/"nanogpt_routing.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
