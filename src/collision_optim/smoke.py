"""Reproducible synthetic smoke experiments, not a tuned optimizer leaderboard."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .active import ActiveConfig, ActiveReflection, MomentumRestart, MomentumSGD
from .core import State, elastic_pair_collision, specular_reflection
from .hybrid import HybridConfig, HybridContact
from .objectives import Budget, CountedObjective, Quadratic, TiltedDoubleWell
from .runner import run


def geometry_checks(samples=1000):
    rng = np.random.default_rng(5239)
    errors = {"reflection_energy": 0.0, "reflection_normal": 0.0,
              "reflection_tangent": 0.0, "pair_momentum": 0.0, "pair_energy": 0.0}
    for _ in range(samples):
        normal = rng.normal(size=5)
        normal /= np.linalg.norm(normal)
        a, b = rng.normal(size=(2, 5))
        reflected = specular_reflection(a, normal)
        left, right = elastic_pair_collision(a, b, normal)
        values = {
            "reflection_energy": abs(0.5 * (reflected @ reflected - a @ a)),
            "reflection_normal": abs((reflected + a) @ normal),
            "reflection_tangent": np.linalg.norm((reflected - a) - ((reflected - a) @ normal) * normal),
            "pair_momentum": np.linalg.norm(left + right - a - b),
            "pair_energy": abs(0.5 * (left @ left + right @ right - a @ a - b @ b)),
        }
        for key, value in values.items():
            errors[key] = max(errors[key], float(value))
    planes = []
    for slope in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        reflected = specular_reflection([0, -1], [-slope, 1])
        planes.append({"slope": slope, "incoming": [0, -1], "outgoing": reflected.tolist()})
    return {"samples": samples, "maximum_absolute_errors": errors,
            "passed": all(value < 1e-10 for value in errors.values()), "planes": planes}


def conservative_checks(mixing=0.5):
    cases = {}
    for name, initial, duration in [
        ("free_motion", State([1.0], [0.0], z=2.0, vz=0.0), 1.0),
        ("with_contacts", State([0.2], [0.3], z=0.25, vz=-0.2), 4.0),
    ]:
        records = []
        for h in [0.04, 0.02, 0.01]:
            result = run(HybridContact(HybridConfig(h=h, beta=1, mixing=mixing)), CountedObjective(Quadratic()),
                         initial, round(duration / h))
            energy0 = result.rows[0]["energy"]
            error = max(abs(row["energy"] - energy0) for row in result.rows) / (1 + abs(energy0))
            records.append({"h": h, "mixing": mixing, "max_normalized_energy_error": error,
                            "contacts": len(result.contacts), "stop_reason": result.stop_reason})
        cases[name] = records
    return cases


def _write_csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiments(output, *, seeds=10, steps=300, h=0.05, beta=0.995, mixing=0.5,
                    gap=0.2, vz=-0.1, value_budget=None, gradient_budget=None):
    if seeds < 1 or steps < 1 or gap < 0 or not np.isfinite(gap) or not np.isfinite(vz):
        raise ValueError("Require positive seeds/steps and a finite nonnegative gap.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces"
    traces.mkdir(exist_ok=True)
    active_cfg = ActiveConfig(h=h, beta=beta, gradient_scale=mixing)
    hybrid_cfg = HybridConfig(h=h, beta=beta, mixing=mixing)
    budget = Budget(values=value_budget if value_budget is not None else 4 * steps + 1,
                    gradients=gradient_budget if gradient_budget is not None else steps + 1,
                    paths=4 * steps + 1)
    config = {"seeds": seeds, "steps": steps, "base_seed": 5239,
              "active": asdict(active_cfg), "hybrid": asdict(hybrid_cfg),
              "initial_gap": gap, "initial_vz": vz, "budget_per_run": asdict(budget),
              "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}
    _json_write(output / "config.json", config)
    geometry, conservative = geometry_checks(), conservative_checks(mixing=mixing)
    summaries, runs = [], {}
    for case_name, objective in [("quadratic", Quadratic()), ("tilted_double_well", TiltedDoubleWell())]:
        for seed in range(seeds):
            rng = np.random.default_rng(5239 + seed)
            x0 = np.array([1.0 + rng.normal(0, 0.03)]) if case_name == "quadratic" else np.array([0.95 + rng.normal(0, 0.03)])
            v0 = np.array([0.3 + rng.normal(0, 0.05)]) if case_name == "quadratic" else np.array([-0.6 + rng.normal(0, 0.08)])
            initial_value = objective.value(x0)
            extra_reserve = 0.5 * vz**2 + (1 - mixing) * gap
            control_v = np.sign(v0) * np.sqrt(v0**2 + 2 * extra_reserve)
            factories = {
                "momentum_sgd": lambda: MomentumSGD(active_cfg),
                "momentum_restart": lambda: MomentumRestart(active_cfg),
                "active_reflection": lambda: ActiveReflection(active_cfg),
                "hybrid_contact": lambda: HybridContact(hybrid_cfg),
                "momentum_extra_reserve": lambda: MomentumSGD(active_cfg),
            }
            for name, factory in factories.items():
                method = factory()
                initial = State(x0, control_v if name == "momentum_extra_reserve" else v0)
                if name == "hybrid_contact":
                    initial = State(x0, v0, initial_value + gap, vz)
                result = run(method, CountedObjective(objective, budget), initial, steps)
                runs[(case_name, name, seed)] = result
                stem = f"{case_name}__{name}__seed{seed:02d}"
                _write_csv(traces / f"{stem}.csv", result.rows)
                if name == "hybrid_contact":
                    _json_write(traces / f"{stem}__contacts.json", result.contacts)
                success = any(row["x"][0] < -0.5 and row["value"] < initial_value for row in result.rows) if case_name == "tilted_double_well" else result.best_value < 1e-3
                summaries.append({
                    "case": case_name, "method": name, "seed": seed,
                    "initial_x": x0.tolist(), "initial_v": initial.v.tolist(),
                    "initial_gap": gap if name == "hybrid_contact" else None,
                    "initial_vz": vz if name == "hybrid_contact" else None,
                    "completed_steps": len(result.rows) - 1,
                    "stop_reason": result.stop_reason,
                    "best_value": result.best_value, "best_x": result.best_x.tolist(),
                    "final_value": result.rows[-1]["value"], "success": success,
                    "contacts": len(result.contacts),
                    "active_interventions": sum(row["intervention"] in {"reflect", "restart"} for row in result.rows),
                    "wall_seconds": result.wall_seconds, **result.counts,
                })
    _write_csv(output / "runs.csv", summaries)
    aggregate = []
    for case in ["quadratic", "tilted_double_well"]:
        for name in factories:
            selected = [s for s in summaries if s["case"] == case and s["method"] == name]
            aggregate.append({
                "case": case, "method": name, "runs": len(selected),
                "successes": sum(s["success"] for s in selected),
                "median_best_value": float(np.median([s["best_value"] for s in selected])),
                "median_value_calls": float(np.median([s["value_calls"] for s in selected])),
                "median_gradient_calls": float(np.median([s["gradient_calls"] for s in selected])),
                "median_path_calls": float(np.median([s["path_calls"] for s in selected])),
                "median_wall_seconds": float(np.median([s["wall_seconds"] for s in selected])),
                "contact_errors": sum(s["stop_reason"].startswith("contact_error") for s in selected),
                "total_contacts": sum(s["contacts"] for s in selected),
                "total_active_interventions": sum(s["active_interventions"] for s in selected),
            })
    no_penetration = all(row.get("gap", 0) >= -hybrid_cfg.gap_tolerance
                         for (case, name, seed), result in runs.items() if name == "hybrid_contact"
                         for row in result.rows)
    no_contact_errors = all(not s["stop_reason"].startswith("contact_error") for s in summaries)
    convergence = all(records[-1]["max_normalized_energy_error"] < records[0]["max_normalized_energy_error"]
                      and all(record["stop_reason"] == "completed" for record in records)
                      for records in conservative.values())
    checks = {"isolated_impulses": geometry["passed"], "no_saved_penetration": no_penetration,
              "no_contact_errors_in_benchmark": no_contact_errors,
              "conservative_error_reduces_with_h": convergence,
              "active_interventions_exercised": any(s["active_interventions"] > 0
                                                    for s in summaries if s["method"] == "active_reflection"),
              "hybrid_contacts_exercised": any(s["contacts"] > 0
                                              for s in summaries if s["method"] == "hybrid_contact")}
    report = {
        "checks": checks, "geometry": geometry, "conservative": conservative,
        "aggregate": aggregate,
        "limitations": [
            "Untuned single-particle synthetic smoke experiment; no NanoGPT performance claim.",
            "Hybrid contact detection uses separately counted analytic polynomial path access; it is not a neural-network black-box detector.",
            "The extra-reserve control only reallocates initial vertical kinetic energy and gravitational clearance reserve; it does not make the two different potentials or state spaces identical.",
            "Noisy paired diagnostics are supported by ActiveReflection but these optimization runs use deterministic full gradients.",
            "Grazing/sustained contacts fail explicitly; no hidden projection or speed deletion is applied.",
        ],
    }
    _json_write(output / "summary.json", report)
    _plots(output, runs, conservative, seeds)
    return report


def _plots(output, runs, conservative, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"momentum_sgd": "#2563a6", "momentum_restart": "#777777",
              "active_reflection": "#df791b", "hybrid_contact": "#21836c",
              "momentum_extra_reserve": "#8a5ca5"}
    labels = {"momentum_sgd": "Momentum SGD", "momentum_restart": "Matched restart",
              "active_reflection": "Active reflection", "hybrid_contact": "Hybrid contact",
              "momentum_extra_reserve": "SGD + initial reserve"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for horizontal, filename, axis_label in [
        ("gradient_calls", "optimization_smoke.png", "Gradient evaluations"),
        ("value_calls", "objective_budget_smoke.png", "Objective evaluations"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
        for ax, case, title in zip(axes, ["quadratic", "tilted_double_well"], ["Quadratic", "Tilted double well"]):
            for name in colors:
                paths = [runs[(case, name, seed)].rows for seed in range(seeds)]
                # Do not extend a failed or budget-stopped trajectory past its last completed state.
                start = max(path[0][horizontal] for path in paths)
                end = min(path[-1][horizontal] for path in paths)
                grid = np.linspace(start, max(start, end), 150)
                values = np.array([np.interp(grid, [r[horizontal] for r in path], [r["best_value"] for r in path])
                                   for path in paths])
                median = np.median(values, axis=0)
                lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
                ax.plot(grid, median, label=labels[name], color=colors[name], lw=1.6)
                ax.fill_between(grid, lower, upper, color=colors[name], alpha=0.10)
            ax.set(title=title, xlabel=axis_label, ylabel="Best objective f(x)")
            ax.grid(alpha=0.18)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle("Synthetic smoke only: median and interquartile range; shared horizontal force", fontsize=11)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    example = runs[("tilted_double_well", "hybrid_contact", 0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    # Include actual contact points between saved macro states so the drawn
    # polyline follows the simulated drifts rather than cutting across bounces.
    path = [[example.rows[0]["x"][0], example.rows[0]["z"]]]
    events_by_step = {}
    for event in example.contacts:
        events_by_step.setdefault(event["step"], []).append(event)
    for row in example.rows[1:]:
        path.extend(event["point"] for event in events_by_step.get(row["step"], []))
        path.append([row["x"][0], row["z"]])
    x, z = np.asarray(path).T
    lo, hi = min(-1.3, x.min() - 0.15), max(1.3, x.max() + 0.15)
    xx = np.linspace(lo, hi, 400)
    ff = np.array([TiltedDoubleWell().value(np.array([a])) for a in xx])
    axes[0].plot(xx, ff, color="#555555", label="Wall z=f(x)")
    axes[0].plot(x, z, color=colors["hybrid_contact"], lw=1.1, label="Particle trajectory")
    axes[0].scatter(x[0], z[0], marker="o", s=30, color="#2563a6", label="Start", zorder=4)
    if example.contacts:
        points = np.array([event["point"] for event in example.contacts])
        axes[0].scatter(points[:, 0], points[:, 1], s=12, color="#df791b", label="Contacts", zorder=3)
    axes[0].set(xlabel="Parameter x", ylabel="Height z", title="Hybrid geometry, seed 0")
    axes[0].legend(fontsize=8, frameon=False)
    for case, entries in conservative.items():
        axes[1].plot([r["h"] for r in entries], [r["max_normalized_energy_error"] for r in entries],
                     marker="o", label=case.replace("_", " "))
    axes[1].set(xlabel="Step size h", ylabel="Maximum normalized energy error", title="Undamped energy diagnostic")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(output / "hybrid_diagnostics.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/smoke"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.995)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--gap", type=float, default=0.2)
    parser.add_argument("--vz", type=float, default=-0.1)
    parser.add_argument("--value-budget", type=int)
    parser.add_argument("--gradient-budget", type=int)
    args = parser.parse_args(argv)
    report = run_experiments(args.output, seeds=args.seeds, steps=args.steps, h=args.h,
                             beta=args.beta, mixing=args.mixing, gap=args.gap, vz=args.vz,
                             value_budget=args.value_budget, gradient_budget=args.gradient_budget)
    print(json.dumps({"output": str(args.output.resolve()), "checks": report["checks"],
                      "aggregate": report["aggregate"]}, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

