"""Fixed-duration surface audit; extra evaluations are not a free speedup."""

from dataclasses import asdict
from pathlib import Path
import numpy as np

from .active import ActiveConfig, MomentumSGD
from .gravity import GravityConfig, GravityContact
from .gravity_experiment import _initials
from .highdim import _csv, _json, make_case
from .objectives import CountedObjective
from .runner import run


def run_duration_audit(output, *, dimensions=(2, 100000), steps=1000, h=0.025):
    """Same physical duration, initial energy, and potential on the surface.

    This supplements the capped benchmark: sustained contact can exhaust its
    gradient cap early. Here calls are unlimited but all actual costs are saved.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = GravityConfig(h=h, beta=0.995**(h/0.05))
    _json(output / "config.json", {"gravity": asdict(cfg), "dimensions": list(dimensions),
                                   "steps": steps, "seed": 0, "case": "coupled_wells",
                                   "budget": None, "comparison": "fixed physical duration, not equal compute"})
    summaries, curves = [], {}
    for d in dimensions:
        objective = make_case("coupled_wells", d)
        plain, surface, matched = _initials(objective, "coupled_wells", 0, "surface", cfg)
        denominator = objective.value(plain.x)-objective.minimum
        for method_name in ("momentum_energy_match", "gravity_contact"):
            if method_name == "gravity_contact":
                method, initial = GravityContact(cfg), surface
            else:
                method = MomentumSGD(ActiveConfig(h=h, beta=cfg.beta, gradient_scale=cfg.gravity*cfg.height_scale))
                initial = matched
            result = run(method, CountedObjective(objective), initial, steps, store_states=False)
            for row in result.rows:
                row["relative_best_gap"] = (row["best_value"]-objective.minimum)/denominator
            _csv(output / f"d{d}__{method_name}.csv.gz", result.rows)
            curves[(d, method_name)] = result.rows
            entry = {"dimension": d, "method": method_name, "stop_reason": result.stop_reason,
                     "initial_energy": result.rows[0]["energy"],
                     "completed_time": result.rows[-1]["time"],
                     "relative_best_gap": (result.best_value-objective.minimum)/denominator,
                     "left_fraction": float(np.mean(objective.coordinates(result.best_x)<0)),
                     "max_normalized_residual": max(abs(r["cumulative_residual"]) for r in result.rows)/(1+abs(result.rows[0]["energy"])),
                     "contacts": len(result.contacts),
                     "sliding_time": sum(r.get("sliding_time", 0) for r in result.rows),
                     "wall_seconds": result.wall_seconds, **result.counts}
            summaries.append(entry)
            _json(output / "summary.json", {"runs": summaries, "note": "Fixed physical duration; actual evaluation costs differ."})
            print(f"Duration audit d={d:,}, {method_name}: {result.stop_reason}, "
                  f"gap={entry['relative_best_gap']:.6g}, gradients={entry['gradient_calls']}", flush=True)
    _csv(output / "runs.csv", summaries)
    _plot_duration(output, curves, dimensions)
    return summaries


def _plot_duration(output, curves, dimensions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(dimensions), figsize=(5*len(dimensions), 6),
                             squeeze=False, constrained_layout=True)
    for j, d in enumerate(dimensions):
        for method, color, label in [("momentum_energy_match", "#8b5cb2", "Momentum, energy matched"),
                                     ("gravity_contact", "#14816a", "Pure gravity")]:
            rows = curves[(d, method)]
            for i, horizontal in enumerate(["time", "gradient_calls"]):
                axes[i, j].semilogy([r[horizontal] for r in rows],
                                    [max(r["relative_best_gap"], 1e-12) for r in rows],
                                    label=label, color=color)
        axes[0, j].set_title(f"Coupled wells, d={d:,}; seed 0")
        axes[0, j].set_xlabel("Physical time")
        axes[1, j].set_xlabel("Gradient calls (path costs separate)")
        for ax in axes[:, j]:
            ax.set_ylabel("Relative best gap")
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Fixed-duration surface audit; equal initial energy, different computation")
    fig.savefig(output / "duration_comparison.png", dpi=170)
    plt.close(fig)
