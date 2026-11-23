"""E26: compare fixed initialization choices with matched relative loss."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile

import numpy as np

from collision_optim.curvature_energy import (
    CONDITIONS, IndependentEnergyPopulation, RotatedEnergyWells, initial_state, parameter_cloud,
)
from collision_optim.energy_generalization import controlled_velocities, generalization_score
from collision_optim.event_experiment import save_trace
from collision_optim.event_replay import pack
from collision_optim.gravity import GravityConfig
from collision_optim.initial_budget import allocate_initial, angular_allocation
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, loss_gap_score, read, sha, source_policy

SPEC = Path("docs/superpowers/specs/2026-11-10-overlap-transfer-design.md")
PARENT = Path("output/collision_conditional_transfer/config.json")
SEEDS = {"preflight": [1890], "development": list(range(1900, 1912)), "formal": list(range(1920, 1940))}
SCORES = (30, 300)
KINETICS = (.0002, .001, .004, .016)
PREFIX_STEPS = 10


def build_context(condition, seed, score_target, threshold):
    objective = RotatedEnergyWells(8, seed)
    template, roots = parameter_cloud(objective, condition, seed)
    references = [objective.value(objective.from_coordinates([r, 0.])) for r in roots]
    pivot_loss = objective.value(template[0])
    spread = .11 / score_target
    spectator_losses = pivot_loss + np.array([.11 - spread / 2, .11, .11 + spread / 2])
    positions = [template[0]]
    for point, target in zip(template[1:], spectator_losses):
        sign = -1 if objective.coordinates(point)[1] < 0 else 1
        y = sign * np.sqrt((target - references[-1]) / 4)
        positions.append(objective.from_coordinates([roots[-1], y]))
    probe = PopulationOracle(objective)
    values = [probe.value(point) for point in positions]
    score = loss_gap_score(values)
    assert abs(score - score_target) < 1e-8 and int(np.argmin(values)) == 0
    k0 = KINETICS[seed % len(KINETICS)]
    velocities = controlled_velocities(seed, k0)
    center = roots[0] if condition == "good_present" else roots[-1]
    margin = 1e-4 if condition == "good_present" else .1 * (min(values) - references[0])
    context = {
        "context_id": f"{condition}__g{score_target}__s{seed}",
        "condition": condition, "seed": seed, "stiffness": 8, "longitudinal_scale": 1.,
        "incumbent_energy": k0, "score_target": score_target, "score": score, "threshold": threshold,
        "conditional_source_policy": source_policy(score, threshold),
        "rotation": objective.rotation.tolist(), "points": np.asarray(positions).tolist(),
        "reference_roots": roots.tolist(), "reference_values": references, "minimum": references[0],
        "target": references[0] + margin, "boundary": float(roots[1]),
        "reference_point0": objective.from_coordinates([center, 0.]).tolist(), "values": values,
        "initial_horizontal_velocities": velocities.tolist(), "probe_counts": probe.counts(),
        "setup_reference_values": 4,
    }
    return objective, positions, values, velocities, context


def simulate(output, condition, seed, score_target, threshold, budget, steps):
    objective, points, values, velocities, context = build_context(condition, seed, score_target, threshold)
    name = context["context_id"]
    _json(output / "contexts" / (name + ".json"), context)
    runs = []
    for mode in MODES:
        allocation = allocate_initial(values, (0.,) * 4, f"fixed_ref_{mode}", velocities)
        allocation.update(angular_allocation(points, values, allocation, velocities))
        state = initial_state(points, values, allocation, allocation["horizontal_velocities"])
        config = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
        method = IndependentEnergyPopulation(config)
        oracle = PopulationOracle(objective, total_budget=budget)
        prefix = allocation["charged_value_prefix"]
        oracle.values = prefix
        result = run_population(method, oracle, state, steps)
        key = f"{name}__{mode}"
        save_trace(output, key, result.rows, result.events, method.map_attempts, {
            "stop_reason": result.stop_reason, "last_committed_state": pack(result.final_state),
            "counts": result.counts, "partial_intermediate_states": "Not returned by the E25 solver",
        })
        _json(output / "initial_states" / (key + ".json"), {
            **pack(state), "allocation": allocation, "context_id": name, "population_config": asdict(config),
        })
        total = result.counts["total_oracle_calls"]
        runs.append({
            "run_id": key, "context_id": name, "policy": mode, "seed": seed, "condition": condition,
            "score_target": score_target, "selected_index": allocation["selected_index"],
            "failed": result.stop_reason.startswith(("contact_error:", "numerical_error:")),
            "stop_reason": result.stop_reason, "total_oracle_calls": total, "actual_branch_calls": total - prefix,
            "initial_energy": result.rows[0]["energy"], "physical_time": result.rows[-1]["time"],
            "wall_seconds": result.wall_seconds, "counts": result.counts,
            **generalization_score(result.rows, context, objective),
        })
    return context, runs


def save(output, contexts, runs):
    _json(output / "contexts.json", contexts)
    _json(output / "runs.json", runs)
    if runs:
        _csv(output / "runs.csv", runs)
    shared = sum(sum(c["probe_counts"].values()) for c in contexts)
    branches = sum(r["actual_branch_calls"] for r in runs)
    setup = sum(c["setup_reference_values"] for c in contexts)
    counts = {
        "contexts": len(contexts), "trajectory_executions": len(runs), "shared_probe_calls": shared,
        "actual_branch_calls": branches, "setup_reference_values": setup,
        "actual_solver_queries": shared + branches, "actual_queries_including_setup": shared + branches + setup,
        "charged_control_queries": sum(r["total_oracle_calls"] for r in runs),
    }
    _json(output / "execution_counts.json", counts)
    return counts


def snapshot(output):
    files = {*Path("src").rglob("*.py"), *Path("tests").rglob("*.py"), *Path(".").glob("*.py"), *Path("scripts").glob("*.py"),
             SPEC, Path("AGENTS.md"), Path("pyproject.toml")}
    manifest = {}
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            data = path.read_bytes()
            archive.writestr(str(path), data)
            manifest[str(path)] = hashlib.sha256(data).hexdigest()
    _json(output / "source_manifest.json", manifest)
    return manifest


def run(args):
    output = args.output
    threshold = read(PARENT)["threshold"]
    assert threshold == 100
    locks = {str(PARENT): sha(PARENT)}
    if args.lock:
        lock_config = read(args.lock / "config.json")
        assert lock_config["phase"] == "E26"
        assert lock_config["stage"] == {"development": "preflight", "formal": "development"}[args.stage]
        prior = read(args.lock / "audit.json")
        assert prior["passed"] and not prior["solver_failures"]
        locks[str(args.lock / "audit.json")] = sha(args.lock / "audit.json")
        if args.stage == "formal":
            directions = args.lock / "feature_directions.json"
            locks[str(directions)] = sha(directions)
    output.mkdir(parents=True)
    for name in ("traces", "events", "maps", "uncommitted", "initial_states", "contexts"):
        (output / name).mkdir()
    source = snapshot(output)
    if args.lock:
        assert source == lock_config["source_sha256"]
    budget, steps = (1200, 80) if args.stage == "preflight" else (6000, 500)
    seeds = SEEDS[args.stage]
    _json(output / "config.json", {
        "phase": "E26", "stage": args.stage, "seeds": seeds, "conditions": list(CONDITIONS),
        "score_targets": list(SCORES), "threshold": threshold, "kinetics_by_seed_modulo_four": list(KINETICS),
        "query_cap": budget, "max_steps": steps, "prefix_steps": PREFIX_STEPS,
        "stiffness": 8, "h": .05, "gravity": .5, "initial_kinetic": .6, "selected_height": .025,
        "total_clearance_height": 3.2, "binary_interactions": False, "online_control": False,
        "source_sha256": source, "source_archive_sha256": sha(output / "source.zip"),
        "data_sha256": locks, "previous_stage": str(args.lock) if args.lock else None,
        "launch_command": ["python3", *sys.argv], "cwd": str(Path.cwd()), "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "workers": args.workers,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    contexts, runs = [], []
    save(output, contexts, runs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(simulate, output, condition, seed, score, threshold, budget, steps)
                for seed in seeds for score in SCORES for condition in CONDITIONS]
        try:
            for index, job in enumerate(as_completed(jobs), 1):
                context, rows = job.result()
                contexts.append(context)
                runs.extend(rows)
                counts = save(output, contexts, runs)
                if index % 8 == 0 or index == len(jobs):
                    print(f"{args.stage}: {index}/{len(jobs)} contexts, {len(runs)} trajectories", flush=True)
        except BaseException:
            _json(output / "task_failure.json", {"traceback": traceback.format_exc(),
                  "incomplete_cost": "Costs of workers that did not return are unknown."})
            for job in jobs:
                job.cancel()
            raise
    _json(output / "completion_observation.json", {
        **counts, "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.stage == "preflight") == bool(args.lock):
        parser.error("Positive workers required. Development and formal require the preceding stage with --lock.")
    run(args)
