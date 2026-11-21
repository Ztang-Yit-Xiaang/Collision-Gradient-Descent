"""High-dimensional, matrix-free diagnostics with bounded trajectory storage."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .active import ActiveConfig, ActiveReflection, MomentumRestart, MomentumSGD
from .core import State
from .highdim_objectives import CoupledDoubleWell, CoupledRosenbrock, DiagonalEllipsoid
from .hybrid import HybridConfig, HybridContact
from .objectives import Budget, CountedObjective, TiltedDoubleWell
from .runner import run


METHODS = ("momentum_sgd", "momentum_restart", "active_reflection", "hybrid_contact", "momentum_extra_reserve")
CASES = ("ellipsoid", "independent_wells", "coupled_wells", "rosenbrock")


def _json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _failed(reason):
    return reason.startswith(("contact_error", "numerical_error"))


def _read_checkpoint(output):
    checkpoint = output / "checkpoint.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())["runs"]
    csv_path = output / "runs.csv"
    if not csv_path.exists():
        return []
    # Compatibility with the first batch, whose CSV was saved after each group.
    integers = {"dimension", "seed", "completed_steps", "contacts", "active_interventions",
                "value_calls", "gradient_calls", "path_calls"}
    strings = {"case", "method", "stop_reason"}
    with csv_path.open() as stream:
        records = []
        for row in csv.DictReader(stream):
            converted = {}
            for key, value in row.items():
                if key in strings:
                    converted[key] = value
                elif value == "":
                    converted[key] = None
                elif key == "success":
                    converted[key] = value.lower() == "true"
                elif key in integers:
                    converted[key] = int(value)
                else:
                    converted[key] = float(value)
            records.append(converted)
    return records


def make_case(name, dimension):
    if name == "ellipsoid":
        return DiagonalEllipsoid(dimension)
    if name == "independent_wells":
        return CoupledDoubleWell(dimension, coupling=0.0)
    if name == "coupled_wells":
        return CoupledDoubleWell(dimension, coupling=0.2)
    if name == "rosenbrock":
        return CoupledRosenbrock(dimension)
    raise ValueError(f"Unknown case: {name}")


def initialize_case(name, objective, seed):
    rng = np.random.default_rng(5239 + seed)
    d = objective.dimension
    if name in {"independent_wells", "coupled_wells"}:
        r = 0.95 + rng.normal(0, 0.06, d)
        x = objective.from_coordinates(r)
        v = objective.from_coordinates(-1.0 + rng.normal(0, 0.2, d))
        v *= 0.6 / np.linalg.norm(v)
    else:
        if name == "ellipsoid":
            r = rng.normal(size=d)
            r *= np.sqrt(d) / np.linalg.norm(r)
        else:
            r = np.where(np.arange(d) % 2 == 0, -1.2, 1.0) + rng.normal(0, 0.03, d)
        x = objective.from_coordinates(r)
        g = objective.gradient(x)
        v = 0.3 * g / np.linalg.norm(g)
    return x, v


def audit_original_double_well():
    """Reproduce the original seed-0 example and inspect the first intervention."""
    objective = TiltedDoubleWell()
    rng = np.random.default_rng(5239)
    x = np.array([0.95 + rng.normal(0, 0.03)])
    v = np.array([-0.6 + rng.normal(0, 0.08)])
    cfg = ActiveConfig(h=0.05, beta=0.995, gradient_scale=0.5)
    active = run(ActiveReflection(cfg), CountedObjective(objective), State(x, v), 300)
    hybrid = run(HybridContact(HybridConfig(h=0.05, beta=0.995, mixing=0.5)),
                 CountedObjective(objective), State(x, v, objective.value(x)+0.2, -0.1), 300)
    first_active = next(row for row in active.rows if row["intervention"] == "reflect")
    first_contact = hybrid.contacts[0]
    saddle = sorted(np.roots([1, 0, -1, objective.tilt]).real)[1]
    barrier = objective.value(np.array([saddle]))
    return {
        "seed": 0, "x0": x.tolist(), "v0": v.tolist(),
        "active_first_step": first_active["step"], "active_velocity_after": first_active["v"],
        "hybrid_first_contact": first_contact,
        "active_initial_energy": active.rows[0]["energy"],
        "hybrid_initial_energy": hybrid.rows[0]["energy"],
        "saddle_coordinate": float(saddle),
        "active_saddle_potential": float(0.5*barrier),
        "hybrid_minimum_potential_at_saddle": float(barrier),
        "interpretation": [
            "The active rule reverses the uphill velocity on step 3, interrupting the attempted barrier crossing.",
            "The first geometric contact removes much of the leftward horizontal speed and redirects energy vertically.",
            "Matching horizontal force does not make the two models' effective barriers or initial energy budgets identical.",
            "This single-particle test does not evaluate the original population momentum-exchange hypothesis.",
        ],
    }


def run_highdim(output, *, dimensions=(100, 1000, 10000, 100000), cases=CASES,
                seeds=3, steps=500, h=0.05, beta=0.995, mixing=0.5, resume=False):
    dimensions, cases = list(dimensions), list(cases)
    if not dimensions or any(not isinstance(d, int) or d < 2 for d in dimensions):
        raise ValueError("All dimensions must be integers >=2")
    if not cases or any(name not in CASES for name in cases):
        raise ValueError("Invalid objective cases")
    if not isinstance(seeds, int) or not isinstance(steps, int) or min(seeds, steps) < 1:
        raise ValueError("seeds and steps must be positive integers")
    active_cfg = ActiveConfig(h=h, beta=beta, gradient_scale=mixing)
    hybrid_cfg = HybridConfig(h=h, beta=beta, mixing=mixing)
    gap, vz = 0.2, -0.1
    budget = Budget(values=4*steps+1, gradients=steps+1, paths=4*steps+1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    config = {
        "dimensions": dimensions, "cases": cases, "seeds": seeds, "steps": steps,
        "active": asdict(active_cfg), "hybrid": asdict(hybrid_cfg), "budget": asdict(budget),
        "initial_gap": gap, "initial_vz": vz,
        "normalization": "r=sqrt(d)*Qx; mean local nonlinear energy; fixed total initial horizontal kinetic energy",
        "rotation": "3 seeded Householder reflections, O(d) storage and work",
        "trajectory_storage": "scalar rows only; no per-step d-vectors retained",
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]},
    }
    if resume and (output / "config.json").exists():
        old = json.loads((output / "config.json").read_text())
        for key in ["dimensions", "cases", "seeds", "steps", "active", "hybrid", "budget", "initial_gap", "initial_vz"]:
            if old.get(key) != config[key]:
                raise ValueError(f"Cannot resume with changed {key}")
    summaries = _read_checkpoint(output) if resume else []
    compact_paths = {}
    reused = len(summaries)
    for entry in summaries:
        key = (entry["case"], entry["dimension"], entry["method"], entry["seed"])
        stem = f"{key[0]}__d{key[1]}__{key[2]}__seed{key[3]:02d}"
        with gzip.open(output / "traces" / f"{stem}.csv.gz", "rt") as stream:
            compact_paths[key] = np.array([[float(row["gradient_calls"]), float(row["relative_best_gap"])]
                                           for row in csv.DictReader(stream)])
    config["resume_reused_runs"] = reused
    _json(output / "config.json", config)
    if not resume or not (output / "original_double_well_audit.json").exists():
        _json(output / "original_double_well_audit.json", audit_original_double_well())
    impulse_max = max((entry["max_impulse_energy_error"] for entry in summaries), default=0.0)
    for name in cases:
        for d in dimensions:
            objective = make_case(name, d)
            for seed in range(seeds):
                x0, v0 = initialize_case(name, objective, seed)
                f0 = objective.value(x0)
                initial_gradient_norm = float(np.linalg.norm(objective.gradient(x0)))
                reserve = 0.5*vz*vz + (1-mixing)*gap
                reserve_v = v0*np.sqrt((v0@v0 + 2*reserve)/(v0@v0))
                for method_name in METHODS:
                    if (name, d, method_name, seed) in compact_paths:
                        continue
                    if method_name == "hybrid_contact":
                        method = HybridContact(hybrid_cfg)
                        initial = State(x0, v0, f0+gap, vz)
                    else:
                        constructor = {"active_reflection": ActiveReflection,
                                       "momentum_restart": MomentumRestart}.get(method_name, MomentumSGD)
                        method = constructor(active_cfg)
                        initial = State(x0, reserve_v if method_name == "momentum_extra_reserve" else v0)
                    result = run(method, CountedObjective(objective, budget), initial, steps, store_states=False)
                    denominator = max(1e-12, f0-objective.minimum)
                    relative_gap = (result.best_value-objective.minimum)/denominator
                    failed = _failed(result.stop_reason)
                    best_r = objective.coordinates(result.best_x)
                    left_fraction = float(np.mean(best_r < 0)) if "wells" in name else None
                    success = relative_gap <= 0.1
                    if left_fraction is not None:
                        success = success and left_fraction >= 0.9
                    if failed:
                        success = False
                    impulses = [abs(event["energy_error"]) for event in result.contacts]
                    impulse_max = max([impulse_max] + impulses)
                    stem = f"{name}__d{d}__{method_name}__seed{seed:02d}"
                    for row in result.rows:
                        row["relative_best_gap"] = (row["best_value"]-objective.minimum)/denominator
                    _csv(output / "traces" / f"{stem}.csv.gz", result.rows)
                    # Retain only these O(steps) scalar curves for plotting all runs.
                    compact_paths[(name, d, method_name, seed)] = np.array([
                        [row["gradient_calls"], row["relative_best_gap"]] for row in result.rows])
                    summary = {
                        "case": name, "dimension": d, "method": method_name, "seed": seed,
                        "initial_value": f0, "known_minimum": objective.minimum,
                        "initial_x_norm": float(np.linalg.norm(x0)),
                        "initial_horizontal_kinetic": float(0.5*(initial.v@initial.v)),
                        "initial_gradient_norm": initial_gradient_norm,
                        "best_value": result.best_value, "relative_best_gap": relative_gap,
                        "best_left_fraction": left_fraction,
                        "success": bool(success), "completed_steps": len(result.rows)-1,
                        "stop_reason": result.stop_reason,
                        "contacts": len(result.contacts),
                        "active_interventions": sum(row["intervention"] in {"reflect", "restart"} for row in result.rows),
                        "contact_horizontal_energy_transfer": float(sum(event["horizontal_energy_transfer"] for event in result.contacts)),
                        "mean_contact_normal_vertical": float(np.mean([event["normal_vertical"] for event in result.contacts])) if result.contacts else None,
                        "minimum_saved_surface_gap": min((row.get("gap", gap) for row in result.rows)) if method_name == "hybrid_contact" else None,
                        "max_impulse_energy_error": max(impulses, default=0.0),
                        "normalized_cumulative_integration_residual": abs(result.rows[-1]["cumulative_residual"])/(1+abs(result.rows[0]["energy"])),
                        "wall_seconds_including_diagnostics": result.wall_seconds, **result.counts,
                    }
                    summaries.append(summary)
                    _json(output / "checkpoint.json", {"runs": summaries})
                    _csv(output / "runs.csv", summaries)
            selected = [s for s in summaries if s["case"] == name and s["dimension"] == d]
            errors = sum(_failed(s["stop_reason"]) for s in selected)
            print(f"Finished {name}, d={d:,}, {len(selected)} runs, failures={errors}", flush=True)
            _csv(output / "runs.csv", summaries)

    aggregates = []
    for name in cases:
        for d in dimensions:
            for method in METHODS:
                selected = [s for s in summaries if s["case"] == name and s["dimension"] == d and s["method"] == method]
                valid = [s for s in selected if not _failed(s["stop_reason"])]
                aggregates.append({
                    "case": name, "dimension": d, "method": method, "runs": len(selected),
                    "valid_runs": len(valid),
                    "contact_errors": sum(s["stop_reason"].startswith("contact_error") for s in selected),
                    "numerical_errors": sum(s["stop_reason"].startswith("numerical_error") for s in selected),
                    "successes": sum(s["success"] for s in selected),
                    "median_relative_gap": float(np.median([s["relative_best_gap"] for s in valid])) if valid else None,
                    "median_initial_gradient_norm": float(np.median([s["initial_gradient_norm"] for s in selected])),
                    "median_left_fraction": float(np.median([s["best_left_fraction"] for s in valid])) if valid and "wells" in name else None,
                    "median_gradient_calls": float(np.median([s["gradient_calls"] for s in selected])),
                    "median_value_calls": float(np.median([s["value_calls"] for s in selected])),
                    "median_path_calls": float(np.median([s["path_calls"] for s in selected])),
                    "median_wall_seconds": float(np.median([s["wall_seconds_including_diagnostics"] for s in selected])),
                    "total_contacts": sum(s["contacts"] for s in selected),
                    "total_active_interventions": sum(s["active_interventions"] for s in selected),
                })
    checks = {
        "no_contact_failures": all(not s["stop_reason"].startswith("contact_error") for s in summaries),
        "no_numerical_failures": all(not s["stop_reason"].startswith("numerical_error") for s in summaries),
        "no_saved_penetration": all(s["minimum_saved_surface_gap"] is None or s["minimum_saved_surface_gap"] >= -hybrid_cfg.gap_tolerance for s in summaries),
        "elastic_impulses_preserve_energy": impulse_max < 1e-9,
        "finite_outcomes": all(np.isfinite(s["relative_best_gap"]) for s in summaries),
    }
    report = {"checks": checks, "total_runs": len(summaries), "maximum_dimension": max(dimensions),
              "max_impulse_energy_error": impulse_max, "aggregate": aggregates, "resume_reused_runs": reused,
              "notes": [
                  "Algorithms and tolerances were fixed across cases/dimensions; this is not a tuned performance ranking.",
                  "Known minima allow normalized optimality gaps. Well success additionally requires >=90% negative well coordinates.",
                  "This tests single-particle reflection. Population momentum exchange remains outside this experiment.",
                  "Rotation alone does not make a separable problem intrinsically harder for rotation-equivariant algorithms; neighbor coupling supplies genuine interactions.",
                  "Analytic polynomial path queries remain extra oracle access, separately counted; wall times cannot predict NanoGPT costs.",
                  "Only three seeds by default; reported bands describe those seeds, not statistical confidence.",
                  "Numerically failed runs are explicitly counted and excluded from final-gap medians; dashed curves stop at the last valid saved states.",
              ]}
    _csv(output / "aggregate.csv", aggregates)
    _json(output / "summary.json", report)
    failed_keys = {(s["case"], s["dimension"], s["method"], s["seed"]) for s in summaries if _failed(s["stop_reason"])}
    _plots(output, compact_paths, dimensions, cases, seeds, failed_keys=failed_keys)
    return report


def _plots(output, paths, dimensions, cases, seeds, failed_keys=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    failed_keys = failed_keys or set()
    colors = {"momentum_sgd": "#2563a6", "momentum_restart": "#777777",
              "active_reflection": "#df791b", "hybrid_contact": "#21836c",
              "momentum_extra_reserve": "#8a5ca5"}
    labels = {"momentum_sgd": "Momentum SGD", "momentum_restart": "Matched restart",
              "active_reflection": "Active reflection", "hybrid_contact": "Hybrid contact",
              "momentum_extra_reserve": "SGD + initial reserve"}
    fig, axes = plt.subplots(len(cases), len(dimensions), figsize=(4.2*len(dimensions), 2.6*len(cases)),
                             squeeze=False, constrained_layout=True)
    for i, name in enumerate(cases):
        for j, d in enumerate(dimensions):
            ax = axes[i, j]
            for method in METHODS:
                samples = [paths[(name, d, method, seed)] for seed in range(seeds)]
                end = min(path[-1, 0] for path in samples)
                grid = np.linspace(0, end, 120)
                values = np.array([np.interp(grid, path[:, 0], path[:, 1]) for path in samples])
                med = np.maximum(np.median(values, axis=0), 1e-14)
                lo, hi = np.maximum(np.quantile(values, [0.25, 0.75], axis=0), 1e-14)
                failed_count = sum((name, d, method, seed) in failed_keys for seed in range(seeds))
                ax.semilogy(grid, med, color=colors[method], label=labels[method], lw=1.3,
                            linestyle="--" if failed_count else "-")
                ax.fill_between(grid, lo, hi, color=colors[method], alpha=0.1)
            ax.set_title(f"{name.replace('_', ' ')}, d={d:,}", fontsize=10)
            failures = [(labels[method], sum((name, d, method, seed) in failed_keys for seed in range(seeds))) for method in METHODS]
            failure_text = "; ".join(f"{label}: {count}/{seeds} failed" for label, count in failures if count)
            if failure_text:
                ax.text(0.02, 0.02, failure_text, transform=ax.transAxes, fontsize=6.5, va="bottom", wrap=True)
            ax.set_xlabel("Gradient evaluations", fontsize=9)
            if j == 0:
                ax.set_ylabel("Relative best optimality gap", fontsize=9)
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"Dimension-scaled synthetic diagnostics: median/IQR, {seeds} seeds; fixed algorithm settings", fontsize=12)
    fig.savefig(output / "highdim_comparison.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/highdim"))
    parser.add_argument("--dimensions", nargs="+", type=int, default=[100, 1000, 10000, 100000])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.995)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs only when configuration matches")
    args = parser.parse_args(argv)
    result = run_highdim(args.output, dimensions=args.dimensions, cases=args.cases, seeds=args.seeds,
                         steps=args.steps, h=args.h, beta=args.beta, mixing=args.mixing, resume=args.resume)
    print(json.dumps({"output": str(args.output.resolve()), "total_runs": result["total_runs"],
                      "checks": result["checks"]}, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

