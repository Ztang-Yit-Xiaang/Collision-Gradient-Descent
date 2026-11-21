"""Reproducible pure-gravity comparison with explicit energy and cost controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import sys

import numpy as np

from .active import ActiveConfig, ActiveReflection, MomentumSGD
from .core import State
from .gravity import GravityConfig, GravityContact
from .highdim import _csv, _failed, _json, initialize_case, make_case
from .hybrid import HybridConfig, HybridContact
from .objectives import Affine, Budget, CountedObjective, Quadratic
from .runner import run


METHODS = ("momentum_sgd", "active_reflection", "hybrid_contact",
           "gravity_contact", "momentum_energy_match")
CASES = ("ellipsoid", "coupled_wells", "rosenbrock")
MODES = ("airborne", "surface")


def physical_checks(output):
    """Fixed plane, free-flight and curved-contact experiments, with raw traces."""
    records = []
    for name in ("flight", "flat_bounce", "plane_slide", "curved_slide"):
        for h in (0.1, 0.05, 0.025):
            cfg = GravityConfig(h=h, beta=1)
            method = GravityContact(cfg)
            objective = Affine([0.6]) if name == "plane_slide" else Quadratic()
            if name == "flat_bounce":
                objective = Affine([0.])
                initial = State([0.], [0.2], z=0.2, vz=-0.1)
            elif name == "flight":
                initial = State([0.2], [0.3], z=10., vz=0.1)
            else:
                oracle = CountedObjective(objective)
                initial = method.initialize(oracle, [0.8], [0.1], gap=0, on_surface=True)
            duration = 2.0
            result = run(method, CountedObjective(objective), initial, round(duration/h))
            energy0 = result.rows[0]["energy"]
            maximum_error = max(abs(row["energy"]-energy0) for row in result.rows)
            record = {"case": name, "h": h, "stop_reason": result.stop_reason,
                      "maximum_energy_error": maximum_error,
                      "contacts": len(result.contacts),
                      "sliding_time": sum(row.get("sliding_time", 0) for row in result.rows),
                      **result.counts}
            if name == "plane_slide":
                a = -cfg.gravity*0.6/(1+0.6**2)
                record["position_error"] = abs(result.final_state.x[0]-(0.8+0.1*duration+0.5*a*duration**2))
            records.append(record)
            _csv(output / "traces" / f"physics__{name}__h{h}.csv.gz", result.rows)
    _json(output / "physics.json", records)
    return records


def _initials(objective, name, seed, mode, cfg):
    x, v = initialize_case(name, objective, seed)
    f0 = objective.value(x)
    if mode == "airborne":
        vx, vz, gap = v, -0.1, 0.2
    else:
        # Match total kinetic energy, including the vertical tangent component,
        # across dimensions instead of letting a steep slope add initial energy.
        slope_speed = cfg.height_scale*float(objective.gradient(x) @ v)
        vx = v*np.sqrt((v @ v)/(v @ v+slope_speed**2))
        vz = cfg.height_scale*float(objective.gradient(x) @ vx)
        gap = 0.0
    augmented = State(x, vx, cfg.height_scale*f0+gap, vz)
    reserve = 0.5*vz*vz+cfg.gravity*gap
    matched_v = vx*np.sqrt((vx @ vx+2*reserve)/(vx @ vx))
    return State(x, vx), augmented, State(x, matched_v)


def run_gravity(output, *, dimensions=(2, 1000, 100000), cases=CASES,
                modes=MODES, seeds=3, steps=500, h=0.025, gradient_budget=501):
    dimensions, cases, modes = list(dimensions), list(cases), list(modes)
    if not dimensions or any(not isinstance(d, int) or d < 2 for d in dimensions):
        raise ValueError("Dimensions must be integers >=2.")
    if not cases or any(case not in CASES for case in cases):
        raise ValueError("Unknown gravity experiment case.")
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError("Unknown initial contact mode.")
    if any(not isinstance(n, int) or n < 1 for n in (seeds, steps, gradient_budget)):
        raise ValueError("Seeds, steps and gradient budget must be positive integers.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    beta = 0.995**(h/0.05)
    cfg = GravityConfig(h=h, beta=beta)
    active_cfg = ActiveConfig(h=h, beta=beta, gradient_scale=cfg.gravity*cfg.height_scale,
                              patience=max(1, round(0.15/h)), epsilon_f=0.001*h/0.05)
    hybrid_cfg = HybridConfig(h=h, beta=beta, mixing=0.5)
    budget = Budget(values=10*steps+1, gradients=gradient_budget, paths=10*steps+1)
    _json(output / "config.json", {
        "dimensions": dimensions, "cases": cases, "modes": modes, "seeds": seeds,
        "steps": steps, "gravity": asdict(cfg), "active": asdict(active_cfg),
        "hybrid": asdict(hybrid_cfg), "budget": asdict(budget),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "notes": ["Common observation-time and evaluation caps; actual costs and elapsed physical times differ.",
                  "Energy-match control has the same U=g*ell*f and total initial mechanical energy as pure gravity.",
                  "Horizontal-initial-state control has less initial energy when vertical speed/clearance is nonzero.",
                  "Legacy hybrid has U=f on the surface, so its effective barrier differs.",
                  "The surface cohort starts tangentially, with total initial speed fixed across dimensions.",
                  "Analytic path queries are counted separately, not treated as free neural-network access."]})
    physics = physical_checks(output)
    rows, curves = [], {}
    for mode in modes:
        for name in cases:
            for dimension in dimensions:
                objective = make_case(name, dimension)
                for seed in range(seeds):
                    plain, augmented, matched = _initials(objective, name, seed, mode, cfg)
                    f0 = objective.value(plain.x)
                    denominator = max(1e-12, f0-objective.minimum)
                    for method_name in METHODS:
                        if method_name == "gravity_contact":
                            method, initial = GravityContact(cfg), augmented
                        elif method_name == "hybrid_contact":
                            method, initial = HybridContact(hybrid_cfg), augmented
                        elif method_name == "active_reflection":
                            method, initial = ActiveReflection(active_cfg), plain
                        else:
                            method = MomentumSGD(active_cfg)
                            initial = matched if method_name == "momentum_energy_match" else plain
                        result = run(method, CountedObjective(objective, budget), initial, steps, store_states=False)
                        for row in result.rows:
                            row["relative_best_gap"] = (row["best_value"]-objective.minimum)/denominator
                        stem = f"{mode}__{name}__d{dimension}__{method_name}__s{seed:02d}"
                        _csv(output / "traces" / f"{stem}.csv.gz", result.rows)
                        _csv(output / "traces" / f"{stem}__contacts.csv.gz", result.contacts)
                        curves[(mode, name, dimension, method_name, seed)] = np.array([
                            [r["time"], r["gradient_calls"], r["relative_best_gap"]] for r in result.rows])
                        energy_scale = 1+abs(result.rows[0]["energy"])
                        entry = {
                            "mode": mode, "case": name, "dimension": dimension,
                            "method": method_name, "seed": seed,
                            "stop_reason": result.stop_reason, "failed": _failed(result.stop_reason),
                            "completed_steps": len(result.rows)-1,
                            "completed_time": result.rows[-1]["time"],
                            "initial_value": f0, "initial_energy": result.rows[0]["energy"],
                            "initial_horizontal_kinetic": float(0.5*(initial.v @ initial.v)),
                            "initial_vertical_kinetic": float(0.5*initial.vz**2) if initial.vz is not None else 0.,
                            "best_value": result.best_value,
                            "relative_best_gap": (result.best_value-objective.minimum)/denominator,
                            "left_fraction": float(np.mean(objective.coordinates(result.best_x)<0)) if "wells" in name else None,
                            "contacts": len(result.contacts),
                            "sliding_time": sum(r.get("sliding_time", 0) for r in result.rows),
                            "sliding_substeps": sum(r.get("sliding_substeps", 0) for r in result.rows),
                            "detachments": sum(r.get("detachments", 0) for r in result.rows),
                            "minimum_gap": min((r.get("gap", 0) for r in result.rows)),
                            "max_impulse_energy_error": max((abs(e["energy_error"]) for e in result.contacts), default=0.),
                            "max_normalized_residual": max(abs(r["cumulative_residual"]) for r in result.rows)/energy_scale,
                            "constraint_correction_energy": sum(r.get("constraint_correction_energy", 0.) for r in result.rows),
                            "wall_seconds": result.wall_seconds, **result.counts,
                        }
                        rows.append(entry)
                        _csv(output / "runs.csv", rows)
                selected = [r for r in rows if r["mode"] == mode and r["case"] == name and r["dimension"] == dimension]
                print(f"Gravity comparison {mode}, {name}, d={dimension:,}: {len(selected)} runs, "
                      f"{sum(r['failed'] for r in selected)} failures", flush=True)
    aggregate = []
    for mode in modes:
        for name in cases:
            for dimension in dimensions:
                for method in METHODS:
                    selected = [r for r in rows if (r["mode"], r["case"], r["dimension"], r["method"]) == (mode, name, dimension, method)]
                    valid = [r for r in selected if not r["failed"]]
                    aggregate.append({
                        "mode": mode, "case": name, "dimension": dimension, "method": method,
                        "runs": len(selected), "valid_runs": len(valid),
                        "failures": sum(r["failed"] for r in selected),
                        "median_relative_gap": float(np.median([r["relative_best_gap"] for r in valid])) if valid else None,
                        "median_completed_time": float(np.median([r["completed_time"] for r in selected])),
                        "median_gradient_calls": float(np.median([r["gradient_calls"] for r in selected])),
                        "median_value_calls": float(np.median([r["value_calls"] for r in selected])),
                        "median_path_calls": float(np.median([r["path_calls"] for r in selected])),
                        "median_wall_seconds": float(np.median([r["wall_seconds"] for r in selected])),
                        "total_contacts": sum(r["contacts"] for r in selected),
                        "total_sliding_time": sum(r["sliding_time"] for r in selected),
                    })
    gravity_rows = [r for r in rows if r["method"] == "gravity_contact"]
    checks = {
        "physical_checks_complete": all(r["stop_reason"] == "completed" for r in physics),
        "physical_energy_error_below_1e_7": all(r["maximum_energy_error"] < 1e-7 for r in physics),
        "gravity_no_failures": all(not r["failed"] for r in gravity_rows),
        "gravity_no_saved_penetration": all(r["minimum_gap"] >= -cfg.gap_tolerance for r in gravity_rows),
        "gravity_elastic_impulses": all(r["max_impulse_energy_error"] < 1e-9 for r in gravity_rows),
        "gravity_normalized_residual_below_1e_6": all(r["max_normalized_residual"] < 1e-6 for r in gravity_rows),
    }
    report = {"total_runs": len(rows), "checks": checks, "aggregate": aggregate,
              "physics": physics, "notes": [
                  "Unit-mass single-particle experiment; population momentum exchange is not implemented here.",
                  "Medians exclude numerical/contact failures, which remain explicitly counted.",
                  "Budget stops retain partial trajectories and actual costs; they do not imply convergence.",
                  "These are untuned synthetic experiments with three seeds, not NanoGPT results."]}
    _csv(output / "aggregate.csv", aggregate)
    _json(output / "summary.json", report)
    _plot(output, curves, dimensions, cases, modes, seeds, rows)
    return report


def _plot(output, curves, dimensions, cases, modes, seeds, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Momentum", "Active reflection", "Legacy hybrid", "Pure gravity", "Momentum, energy matched"]
    colors = ["#2b6cb0", "#dd7c22", "#999999", "#14816a", "#8b5cb2"]
    for mode in modes:
        for axis_index, axis_name in [(0, "time"), (1, "gradients")]:
            fig, axes = plt.subplots(len(cases), len(dimensions), figsize=(4.5*len(dimensions), 3.2*len(cases)),
                                     squeeze=False)
            fig.subplots_adjust(left=.14 if len(dimensions) == 1 else .08,
                                right=.99, bottom=.12, top=.92, wspace=.30, hspace=.58)
            for i, case in enumerate(cases):
                for j, d in enumerate(dimensions):
                    ax = axes[i, j]
                    failure_notes = []
                    for method, label, color in zip(METHODS, labels, colors):
                        selected = [r for r in summaries if (r["mode"], r["case"], r["dimension"], r["method"]) == (mode, case, d, method)]
                        failed = sum(r["failed"] for r in selected)
                        if failed:
                            failure_notes.append(f"{label}: {failed}/{seeds} failed")
                        paths = [curves[(mode, case, d, method, r["seed"])] for r in selected if not r["failed"]]
                        if not paths:
                            continue
                        end = min(p[-1, axis_index] for p in paths)
                        grid = np.linspace(0, end, 150)
                        values = np.array([np.interp(grid, p[:, axis_index], p[:, 2]) for p in paths])
                        med = np.maximum(np.median(values, axis=0), 1e-12)
                        lo, hi = np.maximum(np.quantile(values, [.25, .75], axis=0), 1e-12)
                        ax.semilogy(grid, med, color=color, label=label,
                                    linestyle="--" if failed else "-", lw=1.4)
                        ax.fill_between(grid, lo, hi, color=color, alpha=.1)
                    ax.set_title(f"{case.replace('_', ' ')}, d={d:,}", fontsize=10)
                    ax.set_xlabel("Physical time" if axis_index == 0 else "Gradient calls (path costs separate)", fontsize=8)
                    if j == 0:
                        ax.set_ylabel("Relative best gap", fontsize=9)
                    ax.grid(alpha=.15)
                    ax.tick_params(labelsize=8)
                    if failure_notes:
                        ax.text(.02, .02, "; ".join(failure_notes)+"; valid-run bands",
                                transform=ax.transAxes, fontsize=7, va="bottom")
                    ax.spines[["top", "right"]].set_visible(False)
            handles, texts = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, texts, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False, fontsize=9)
            fig.suptitle(f"Pure-gravity diagnostic, {mode} start; {seeds} seeds, median/IQR", fontsize=12)
            fig.savefig(output / f"{mode}__{axis_name}.png", dpi=160)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/gravity"))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 1000, 100000])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--h", type=float, default=0.025)
    parser.add_argument("--gradient-budget", type=int, default=501)
    args = parser.parse_args(argv)
    report = run_gravity(args.output, dimensions=args.dimensions, cases=args.cases, modes=args.modes,
                         seeds=args.seeds, steps=args.steps, h=args.h, gradient_budget=args.gradient_budget)
    print({"output": str(args.output.resolve()), "total_runs": report["total_runs"], "checks": report["checks"]})
    return 0 if all(report["checks"].values()) else 1
