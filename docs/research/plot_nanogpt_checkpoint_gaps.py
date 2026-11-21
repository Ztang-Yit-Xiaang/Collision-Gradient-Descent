"""Plot the saved paired checkpoint results used in the report."""

import argparse
import csv
from math import sqrt
from pathlib import Path
from statistics import mean, variance

from matplotlib import rc_context
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/final_nanogpt_reselection_evaluation/evaluations.csv"
FIGURE = ROOT / "docs/figures/nanogpt_checkpoint_gaps.pdf"
METRICS = (
    ("fixed_index_gap", "Fixed-index gap", "#245D78"),
    ("reselection_gap", "Reselection gap", "#9A512F"),
    ("selected_gap", "Selected test gap", "#3B7261"),
)
MARKERS = ("o", "s", "^", "v", "D", "X")


def signed_label(value):
    return f"{value:+.5f}".replace("-", "−")


def draw_gap(ax, rows, key, color):
    values = [float(row[key]) for row in rows]
    average, sd = mean(values), sqrt(variance(values))
    ax.axhline(0, color="#E8EDF0", linewidth=0.55, zorder=0)
    ax.vlines(0, -0.022, 0.022, color="#8E989C", linewidth=0.8, linestyles=(0, (3, 2)))
    ax.add_patch(Circle((average, 0), sd, facecolor=to_rgba(color, 0.13),
                        edgecolor=color, linewidth=1))
    ax.scatter(average, 0, s=34, marker="+", color=color, linewidth=1, zorder=4)

    observations = sorted(zip(values, rows, MARKERS, strict=True), key=lambda item: item[0])
    for side, band in ((1, observations[::2]), (-1, observations[1::2])):
        middle = min(0.041, max(0.009, band[1][0]))
        if key == "reselection_gap":
            middle += 0.010
        label_spacing = 0.032
        for i, (value, row, marker) in enumerate(band):
            point_y = side * (0.004 + 0.004 * i)
            ax.scatter(value, point_y, s=16, marker=marker, color=color,
                       edgecolors="white", linewidths=0.35, zorder=4)
            label_height = 0.023
            if key == "reselection_gap" and i == 1:
                label_height += 0.009
            ax.annotate(f"{row['seed']}\n{signed_label(value)}",
                        xy=(value, point_y),
                        xytext=(middle + (i - 1) * label_spacing, side * label_height),
                        ha="center", va="bottom" if side > 0 else "top",
                        fontsize=9.1875, linespacing=0.95, color="#263842",
                        arrowprops={"arrowstyle": "-", "color": to_rgba(color, 0.70),
                                    "linewidth": 0.52, "shrinkA": 1.6, "shrinkB": 2.0},
                        zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(-0.029, 0.079), ylim=(-0.049, 0.049))
    ax.set_yticks([])
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", length=2.5, pad=2, labelsize=9.1875)
    return average, sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="Optional PNG for visual review.")
    args = parser.parse_args()
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    style = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9.1875,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
    with rc_context(style):
        fig = Figure(figsize=(6.48, 2.39))
        for left, (key, label, color) in zip((0.035, 0.357, 0.679), METRICS, strict=True):
            ax = fig.add_axes((left, 0.155, 0.286, 0.71))
            ax.spines["left"].set_visible(False)
            average, sd = draw_gap(ax, rows, key, color)
            center = left + 0.143
            fig.text(center, 0.973, label, ha="center", va="top",
                     fontweight="bold", fontsize=11.025)
            fig.text(center, 0.882, f"Mean {signed_label(average)}   SD {sd:.5f}",
                     ha="center", va="center", color=color, fontsize=8.3)
        fig.text(0.5, 0.023, "Test-loss difference", ha="center", va="bottom")
        fig.savefig(FIGURE)
        if args.preview is not None:
            fig.savefig(args.preview, dpi=240)


if __name__ == "__main__":
    main()
