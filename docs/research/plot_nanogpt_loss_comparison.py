"""Plot training histories and final scores from the saved NanoGPT comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/final_nanogpt_exchange"
FIGURE = ROOT / "docs/figures/nanogpt_loss_comparison.pdf"
METHODS = (
    ("exchange", "Exchange", "#234F70", "-"),
    ("independent", "Independent", "#426958", "--"),
    ("adamw", "AdamW", "#985427", "-."),
)
BLOCK_SIZE = 20


def load_metrics():
    config = json.loads((SOURCE / "config.json").read_text())
    histories, validation, test = {}, {}, {}
    for key, _, _, _ in METHODS:
        histories[key], validation[key], test[key] = [], [], []
        for seed in config["seeds"]:
            data = json.loads((SOURCE / f"{seed}_{key}.json").read_text())
            outcome = data["outcome"]
            runs = sorted(data["runs"], key=lambda run: run["replica"])
            selected = next(run for run in runs if run["selected"])
            history = np.array([run["training_history"] for run in runs])[:, :, 0]
            expected_budget = config["gradient_budget_per_method"]
            if history.size != expected_budget or outcome["gradient_evaluations"] != expected_budget:
                raise ValueError(f"Unexpected training budget for seed {seed}, {key}.")
            histories[key].append(history)
            validation[key].append(selected["validation_loss"])
            test[key].append(selected["test_loss"])
    return config, {k: np.array(v) for k, v in histories.items()}, validation, test


def panel_title(ax, panel, title):
    ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold", pad=5)


def plot_training(ax, histories, budget):
    panel_title(ax, "a", "Training loss")
    budget_axis = np.arange(BLOCK_SIZE, budget + 1, BLOCK_SIZE)
    for key, label, color, style in METHODS:
        steps_per_block = BLOCK_SIZE // histories[key].shape[1]
        by_block = histories[key].reshape(
            histories[key].shape[0], histories[key].shape[1], -1, steps_per_block
        )
        ax.plot(budget_axis, by_block.mean(axis=(0, 1, 3)), linestyle=style,
                color=color, linewidth=1.25, label=label)
    ax.set(xlabel="Total gradient evaluations", ylabel="Training loss",
           xlim=(12, budget + 8))
    ax.set_xticks(np.arange(100, budget + 1, 100))
    ax.legend(loc="upper right", fontsize=7.5)


def plot_training_gap(ax, histories, test, schedule):
    panel_title(ax, "b", "Paired training loss gap")
    differences = np.subtract(histories["exchange"], histories["independent"])
    steps = np.arange(1, differences.shape[2] + 1)
    higher = np.greater(test["exchange"], test["independent"])
    higher_mean = differences[higher].mean(axis=(0, 1))
    ax.axhline(0, color="#9CA5AC", linewidth=0.65)
    for step in schedule:
        ax.axvline(step, color="#9CA5AC", alpha=0.75, linestyle=":", linewidth=0.8)
    ax.plot(steps, differences.mean(axis=(0, 1)), color="#647584", linestyle="--",
            linewidth=0.94, label=f"All groups ($n={len(higher)}$)")
    ax.plot(steps, higher_mean, color=METHODS[0][2],
            linewidth=1.05, label=f"Higher final test loss ($n={higher.sum()}$)")
    first_loss_after_swapped_update = schedule[0] + 2
    ax.scatter(first_loss_after_swapped_update, higher_mean[first_loss_after_swapped_update - 1],
               s=14, color=METHODS[0][2], zorder=4)
    ax.set(xlabel="Local Momentum SGD step", ylabel="Exchange minus independent",
           xlim=(1, steps[-1]))
    ax.set_ylim(bottom=min(-0.043, ax.get_ylim()[0]))
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.legend(loc="lower right", fontsize=7.1, facecolor="white", frameon=True,
              framealpha=0.92, edgecolor="none")


def plot_final(ax, panel, title, scores):
    panel_title(ax, panel, title)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E1E7EB", linewidth=0.55)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#899398")
    for position, (key, _, color, _) in enumerate(METHODS):
        values = np.array(scores[key])
        average = values.mean()
        sample_sd = values.std(ddof=1)
        fill = tuple(0.81 + 0.19 * channel for channel in to_rgb(color))
        ax.bar(position, average, width=0.58, color=fill,
               edgecolor=color, linewidth=0.65)
        ax.errorbar(position, average, yerr=sample_sd, fmt="none", ecolor=color,
                    linewidth=1.05, capsize=3, capthick=0.95)
        ax.text(position, 2.26, f"{average:.3f}", ha="center", va="top",
                fontweight="bold", fontsize=10, color=color)
        ax.text(position, 1.92, f"±{sample_sd:.3f}", ha="center", va="top",
                fontsize=8, color=color)
    ax.set(xlim=(-0.53, 2.53), ylim=(0, 3.12))
    ax.set_xticks(range(len(METHODS)), [label for _, label, _, _ in METHODS])
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="Optional PNG for visual review.")
    args = parser.parse_args()
    config, histories, validation, test = load_metrics()
    style = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.handlelength": 2,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
    }
    with rc_context(style):
        fig = Figure(figsize=(6.48, 3.72))
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.09,
                            hspace=0.55, wspace=0.38)
        plot_training(axes[0, 0], histories, config["gradient_budget_per_method"])
        plot_training_gap(axes[0, 1], histories, test, config["schedule"])
        plot_final(axes[1, 0], "c", "Final validation loss", validation)
        plot_final(axes[1, 1], "d", "Final test loss", test)
        fig.savefig(FIGURE)
        if args.preview is not None:
            fig.savefig(args.preview, dpi=240)


if __name__ == "__main__":
    main()
