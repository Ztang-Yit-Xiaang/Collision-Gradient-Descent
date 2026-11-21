"""Typeset the saved bounce measurements for the final report."""

import json
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/collision_stability_map"
DESTINATION = ROOT / "docs/figures/bounce_stability.pdf"


def main():
    matrices = json.loads((SOURCE / "return_matrices.json").read_text())
    runs = json.loads((SOURCE / "rollout_summary.json").read_text())
    style = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9.1875,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
    with rc_context(style):
        fig = Figure(figsize=(6.48, 2.04))
        left, right = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.235,
                            top=0.84, wspace=0.37)
        for kappa, color, linestyle in zip(
            (1, 4, 16), ("#234F70", "#985427", "#426958"), ("-", "--", ":"),
            strict=True,
        ):
            rows = sorted(
                (row for row in matrices if row["kappa"] == kappa
                 and row["epsilon"] == 1e-7),
                key=lambda row: row["product"],
            )
            assert len(rows) == 10
            left.plot([row["product"] for row in rows],
                      [row["spectral_radius"] for row in rows],
                      color=color, linestyle=linestyle, marker="o", markersize=2.5,
                      linewidth=1.1, label=rf"$\kappa={kappa}$")
        left.axhline(1.01, linestyle="--", color="#899398", linewidth=0.6)
        left.set(xlabel=r"Curvature times height, $\kappa H$",
                 ylabel="Spectral radius")
        left.legend(fontsize=7.5, loc="upper left", handlelength=1.8,
                    borderaxespad=0.2, labelspacing=0.2)
        representative = sorted(
            (run for run in runs if run["kappa"] == 4 and run["initial_x"] > 0),
            key=lambda run: run["product"],
        )
        assert len(representative) == 4
        for run, color, linestyle in zip(
            representative, ("#234F70", "#985427", "#426958", "#9A4760"),
            ("-", "--", "-.", ":"), strict=True,
        ):
            trace = json.loads((SOURCE / "return_traces" / f"{run['run_id']}.json").read_text())
            right.plot([row["bounce"] for row in trace["rows"]],
                       [max(row["phase_norm"] / abs(run["initial_x"]), 1e-12)
                        for row in trace["rows"]],
                       color=color, linestyle=linestyle, linewidth=1.05,
                       label=rf"$\kappa H={run['product']:g}$")
        right.set(yscale="log", xlabel="Natural returns",
                  ylabel="Perturbation amplification")
        right.legend(fontsize=7.1, loc="upper left", handlelength=1.5,
                     borderaxespad=0.2, labelspacing=0.15)
        for axis, title in ((left, "a  Return spectrum"), (right, "b  Perturbation growth")):
            axis.set_title(title, loc="left", fontweight="bold", fontsize=11.025, pad=5)
            axis.tick_params(length=2.8, pad=2)
        fig.savefig(DESTINATION)


if __name__ == "__main__":
    main()
