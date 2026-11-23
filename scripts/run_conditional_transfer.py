"""E25 matched initialization replay with a loss-only threshold."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile

import numpy as np

from _source_paths import source_path
from collision_optim.curvature_energy import (
    CONDITIONS, IndependentEnergyPopulation, RotatedEnergyWells, initial_state,
    parameter_cloud,
)
from collision_optim.energy_generalization import controlled_velocities, generalization_score
from collision_optim.event_experiment import save_trace
from collision_optim.event_replay import pack
from collision_optim.gravity import GravityConfig
from collision_optim.initial_budget import allocate_initial, angular_allocation
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.strategic_experiment import _csv, _json

SPEC = Path("docs/superpowers/specs/2026-11-10-conditional-transfer-design.md")
PARENT = Path("output/collision_initial_budget")
SEEDS = {"preflight": [1690], "formal": list(range(1700, 1720))}
THRESHOLDS = (1, 3, 10, 30, 100, 300, 1000)
MODES = ("keep", "transfer")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss_gap_score(values):
    a, b, c, d = sorted(values)
    return (c - a) / max(d - b, 1e-12)


def source_policy(score, threshold):
    return "transfer" if score >= threshold else "keep"


def develop():
    context_path = PARENT / "contexts.json"
    runs_path = PARENT / "runs.json"
    contexts = [c for c in read(context_path) if c["stiffness"] == 8]
    assert {(c["seed"], c["condition"]) for c in contexts} == {
        (seed, condition) for seed in range(1600, 1620) for condition in CONDITIONS
    }
    assert len(contexts) == 40
    branches = {
        (r["context_id"], r["velocity_mode"]): r
        for r in read(runs_path)
        if r["stiffness"] == 8 and r["height_mode"] == "fixed_ref"
    }
    assert len(branches) == 80
    cases = []
    for c in sorted(contexts, key=lambda item: (item["seed"], item["condition"])):
        cases.append({
            "context_id": c["context_id"], "condition": c["condition"], "seed": c["seed"],
            "values": c["values"], "score": loss_gap_score(c["values"]),
            "source_runs": {mode: branches[c["context_id"], mode]["run_id"] for mode in MODES},
        })

    trials = []
    for threshold in THRESHOLDS:
        good_deviations = bad_hits = bad_transfers = good_transfers = 0
        for case in cases:
            mode = source_policy(case["score"], threshold)
            result = branches[case["context_id"], mode]
            if case["condition"] == "good_present":
                good_deviations += result["particle0_harmed"] or result["failed"]
                good_transfers += mode == "transfer"
            else:
                bad_hits += result["final_hit"] and not result["failed"]
                bad_transfers += mode == "transfer"
        trials.append({
            "threshold": threshold, "good_deviations_or_failed": good_deviations,
            "bad_final_hits": bad_hits, "bad_transfers": bad_transfers,
            "good_transfers": good_transfers,
            "ranking_key": [good_deviations, -bad_hits, bad_transfers, threshold],
        })
    selected = min(trials, key=lambda trial: trial["ranking_key"])
    return {
        "threshold": selected["threshold"], "trials": trials, "contexts": cases,
        "source_sha256": {str(p): sha(p) for p in (context_path, runs_path, SPEC)},
        "new_trajectories": 0, "new_oracle_calls": 0, "independent_seed_groups": 20,
    }


def snapshot(output):
    paths = {*Path("src").rglob("*.py"), *Path(".").glob("*.py"), *Path("scripts").glob("*.py"), SPEC,
             Path("pyproject.toml"), Path("AGENTS.md")}
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        manifest = {}
        for path in sorted(paths):
            data = path.read_bytes()
            archive.writestr(str(path), data)
            manifest[str(path)] = hashlib.sha256(data).hexdigest()
    _json(output / "source_manifest.json", manifest)
    return manifest


def simulate(output, condition, seed, budget, steps, threshold):
    objective = RotatedEnergyWells(8, seed)
    points, roots = parameter_cloud(objective, condition, seed)
    velocities = controlled_velocities(seed, .001)
    probes = PopulationOracle(objective)
    values = [probes.value(point) for point in points]
    references = [objective.value(objective.from_coordinates([root, 0.])) for root in roots]
    minimum = references[0]
    target = minimum + (.1 * (min(values) - minimum) if condition == "bad_only" else 1e-4)
    name = f"{condition}__k8__s{seed}"
    gap = loss_gap_score(values)
    selected_mode = source_policy(gap, threshold)
    context = {
        "context_id": name, "condition": condition, "seed": seed, "stiffness": 8,
        "longitudinal_scale": 1., "incumbent_energy": .001,
        "points": np.asarray(points).tolist(), "values": values, "rotation": objective.rotation.tolist(),
        "reference_roots": roots.tolist(), "reference_values": references, "minimum": minimum,
        "boundary": float(roots[1]), "target": target, "reference_point0":
            objective.from_coordinates([roots[0] if condition == "good_present" else roots[-1], 0.]).tolist(),
        "initial_horizontal_velocities": velocities.tolist(), "probe_counts": probes.counts(),
        "score": gap, "threshold": threshold, "conditional_source_policy": selected_mode,
    }
    _json(output / "contexts" / (name + ".json"), context)
    runs = []
    for mode in MODES:
        allocation = allocate_initial(values, (0.,) * 4, f"fixed_ref_{mode}", velocities)
        allocation.update(angular_allocation(points, values, allocation, velocities))
        state = initial_state(points, values, allocation, allocation["horizontal_velocities"])
        cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
        method = IndependentEnergyPopulation(cfg)
        oracle = PopulationOracle(objective, total_budget=budget)
        prefix = allocation["charged_value_prefix"]
        oracle.values = prefix
        result = run_population(method, oracle, state, steps)
        key = f"{name}__{mode}"
        save_trace(output, key, result.rows, result.events, method.map_attempts, {
            "stop_reason": result.stop_reason, "last_committed_state": pack(result.final_state),
            "counts": result.counts, "partial_intermediate_states": "Not returned by the E24 solver",
        })
        _json(output / "initial_states" / (key + ".json"), {
            **pack(state), "allocation": allocation, "context_id": name, "population_config": asdict(cfg),
        })
        total_calls = result.counts["total_oracle_calls"]
        runs.append({
            "run_id": key, "context_id": name, "policy": mode, "condition": condition, "seed": seed,
            "stiffness": 8, "longitudinal_scale": 1., "incumbent_energy": .001,
            "failed": result.stop_reason.startswith(("contact_error:", "numerical_error:")),
            "stop_reason": result.stop_reason, "total_oracle_calls": total_calls,
            "actual_branch_calls": total_calls - prefix,
            "physical_time": result.rows[-1]["time"], "wall_seconds": result.wall_seconds,
            "counts": result.counts, "initial_energy": result.rows[0]["energy"],
            "selected_index": allocation["selected_index"],
            **generalization_score(result.rows, context, objective),
        })
    return context, runs


def save_progress(output, contexts, runs):
    _json(output / "contexts.json", contexts)
    _json(output / "runs.json", runs)
    if runs:
        _csv(output / "runs.csv", runs)
    shared = sum(sum(c["probe_counts"].values()) for c in contexts)
    branches = sum(r["actual_branch_calls"] for r in runs)
    counts = {
        "contexts": len(contexts), "trajectory_executions": len(runs),
        "shared_probe_calls": shared, "actual_branch_calls": branches,
        "actual_solver_queries": shared + branches, "setup_reference_values": 3 * len(contexts),
        "actual_queries_including_setup": shared + branches + 3 * len(contexts),
        "charged_control_queries": sum(r["total_oracle_calls"] for r in runs),
    }
    _json(output / "execution_counts.json", counts)
    return counts


def run(args):
    output = args.output
    output.mkdir(parents=True)
    for name in ("traces", "events", "maps", "uncommitted", "initial_states", "contexts"):
        (output / name).mkdir()
    manifest = snapshot(output)
    preflight = args.stage == "preflight"
    if preflight:
        development = develop()
    else:
        assert read(args.lock / "audit.json")["passed"]
        lock_path = args.lock / "development.json"
        assert sha(lock_path) == read(args.lock / "config.json")["development_sha256"]
        development = read(lock_path)
        for path, digest in development["source_sha256"].items():
            assert sha(source_path(path)) == digest
    _json(output / "development.json", development)
    threshold = development["threshold"]
    budget, steps = (1200, 80) if preflight else (6000, 500)
    seeds = SEEDS[args.stage]
    config = {
        "phase": "E25", "stage": args.stage, "seeds": seeds, "conditions": list(CONDITIONS),
        "stiffness": 8, "threshold": threshold, "query_cap": budget, "max_steps": steps,
        "h": .05, "gravity": .5, "initial_kinetic": .6, "template_particle0_kinetic": .001,
        "height_of_selected_particle": .025, "total_clearance_height": 3.2,
        "initial_displacement": .001, "binary_interactions": False, "online_control": False,
        "policies": [*MODES, "conditional"], "conditional_trajectory_is_derived": True,
        "source_sha256": manifest, "source_archive_sha256": sha(output / "source.zip"),
        "development_sha256": sha(output / "development.json"),
        "development_lock_directory": str(args.lock) if args.lock else None,
        "command": ["python3", *sys.argv], "cwd": str(Path.cwd()), "python": sys.version,
        "numpy": np.__version__, "platform": platform.platform(), "workers": args.workers,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    _json(output / "config.json", config)
    contexts, runs = [], []
    save_progress(output, contexts, runs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(simulate, output, condition, seed, budget, steps, threshold)
                for seed in seeds for condition in CONDITIONS]
        try:
            for i, job in enumerate(as_completed(jobs), 1):
                context, rows = job.result()
                contexts.append(context)
                runs.extend(rows)
                counts = save_progress(output, contexts, runs)
                if i % 5 == 0 or i == len(jobs):
                    print(f"{args.stage}: {i}/{len(jobs)} contexts, {len(runs)} actual trajectories", flush=True)
        except BaseException:
            _json(output / "task_failure.json", {"traceback": traceback.format_exc(),
                "incomplete_cost": "Costs of workers that did not return are unknown."})
            for job in jobs:
                job.cancel()
            raise
    _json(output / "completion_observation.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "expected_contexts": len(jobs),
        "threshold": threshold, **counts,
    })
    print(json.dumps(counts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.stage == "formal") != bool(args.lock):
        parser.error("Positive workers required. Supply --lock for formal stage only.")
    run(args)


if __name__ == "__main__":
    main()
