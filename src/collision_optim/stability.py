"""Step refinement at fixed physical duration and damping, not a cost ranking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .active import ActiveConfig, ActiveReflection, MomentumSGD
from .core import State
from .highdim import _csv, _json, initialize_case, make_case
from .hybrid import HybridConfig, HybridContact
from .objectives import CountedObjective
from .runner import run


def run_stability(output, *, dimension=100000, seed=0, duration=25.0,
                  h_values=(0.05, 0.025, 0.0125)):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive")
    if not h_values or any(not np.isfinite(h) or h <= 0 for h in h_values):
        raise ValueError("All step sizes must be positive")
    if any(abs(round(duration/h)*h-duration) > 1e-9 for h in h_values):
        raise ValueError("duration must be an integer multiple of every step size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    objective = make_case("rosenbrock", dimension)
    x, v = initialize_case("rosenbrock", objective, seed)
    f0 = objective.value(x)
    curves, rows = {}, []
    configurations = []
    for h in h_values:
        # Hold continuous damping rate, physical patience duration, and the
        # approximate tolerated loss-increase *rate* fixed as h is refined.
        beta = 0.995**(h/0.05)
        patience = max(1, round(3*0.05/h))
        cfg = ActiveConfig(h=h, beta=beta, gradient_scale=0.5,
                           patience=patience, epsilon_f=0.001*h/0.05)
        hybrid_cfg = HybridConfig(h=h, beta=beta, mixing=0.5)
        configurations.append({"active": asdict(cfg), "hybrid": asdict(hybrid_cfg),
                               "steps": round(duration/h)})
        for name in ["momentum_sgd", "active_reflection", "hybrid_contact"]:
            if name == "hybrid_contact":
                method = HybridContact(hybrid_cfg)
                initial = State(x, v, f0+0.2, -0.1)
            else:
                method = ActiveReflection(cfg) if name == "active_reflection" else MomentumSGD(cfg)
                initial = State(x, v)
            result = run(method, CountedObjective(objective), initial, round(duration/h), store_states=False)
            energy0 = result.rows[0]["energy"]
            _csv(output / f"{name}__h{h:g}.csv.gz", result.rows)
            curves[(name, h)] = np.array([[row["time"], row["energy"]/energy0] for row in result.rows])
            rows.append({
                "method": name, "h": h, "beta": beta, "patience": patience,
                "epsilon_f": cfg.epsilon_f, "requested_duration": duration,
                "completed_duration": result.rows[-1]["time"], "completed_steps": len(result.rows)-1,
                "stop_reason": result.stop_reason, "best_value": result.best_value,
                "relative_best_gap": result.best_value/f0,
                "max_energy_ratio": max(row["energy"]/energy0 for row in result.rows),
                "contacts": len(result.contacts), **result.counts,
            })
            print(f"Stability {name}, h={h:g}, stop={result.stop_reason}, time={result.rows[-1]['time']:g}", flush=True)
            _csv(output / "runs.csv", rows)
    report = {
        "dimension": dimension, "seed": seed, "case": "rosenbrock",
        "configurations": configurations, "runs": rows,
        "interpretation": [
            "This holds physical duration and damping rate fixed; smaller h uses more oracle calls, so it is not an equal-cost optimizer comparison.",
            "Active patience is scaled with h to retain a comparable physical observation window; epsilon_f is scaled as a rate tolerance.",
            "Finite-time stability at smaller h does not establish convergence or superiority on neural networks.",
        ],
    }
    _json(output / "summary.json", report)
    _plot(output, curves, h_values)
    return report


def _plot(output, curves, h_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, name in zip(axes, ["momentum_sgd", "active_reflection", "hybrid_contact"]):
        for h in h_values:
            curve = curves[(name, h)]
            ax.plot(curve[:, 0], np.log10(np.maximum(curve[:, 1], 1e-300)), label=f"h={h:g}")
        ax.set(title=name.replace("_", " "), xlabel="Physical time", ylabel="log10(H / initial H)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Fixed-duration step refinement; curves stop before invalid states", fontsize=11)
    fig.savefig(output / "step_refinement.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stability"))
    parser.add_argument("--dimension", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=25)
    parser.add_argument("--h-values", nargs="+", type=float, default=[0.05, 0.025, 0.0125])
    args = parser.parse_args(argv)
    run_stability(args.output, dimension=args.dimension, seed=args.seed,
                  duration=args.duration, h_values=args.h_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

