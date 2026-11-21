"""E28: prospectively split short, no-intervention warning trajectories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile

import numpy as np

from collision_optim.curvature_energy import IndependentEnergyPopulation, initial_state
from collision_optim.energy_generalization import controlled_velocities
from collision_optim.gravity import GravityConfig
from collision_optim.initial_budget import allocate_initial
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import read, sha
from run_overlap_transfer import build_context
from run_shared_prefix_transfer import save_segment, stop_record

SPEC = Path("docs/superpowers/specs/2026-11-11-prebounce-warning-design.md")
PARENT = Path("output/collision_shared_prefix_transfer/config.json")
SEEDS = {"preflight": [2190], "development": list(range(2200, 2220)), "formal": list(range(2230, 2270))}
ENERGIES = (1e-5, 8e-5, 1.5e-4, 2.5e-4, 4e-4, 1e-3)
CAP = 300
STEPS = 8


def simulate(root, seed, energy):
    objective, points, values, _, context = build_context("good_present", seed, 300, 100)
    velocities = controlled_velocities(seed, energy)
    allocation = allocate_initial(values, (0.,) * 4, "fixed_ref_keep", velocities)
    state = initial_state(points, values, allocation, velocities)
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    oracle = PopulationOracle(objective, total_budget=CAP)
    oracle.values = 4
    run = run_population(IndependentEnergyPopulation(cfg), oracle, state, STEPS)
    name = f"good_present__s{seed}__k{energy:g}"
    save_segment(root, name, run)
    context.update(context_id=name, incumbent_energy=energy, initial_horizontal_velocities=velocities.tolist(),
                   heights=allocation["heights"])
    context.pop("conditional_source_policy")
    context.pop("threshold")
    return context, {"run_id": name, "seed": seed, "incumbent_energy": energy, **stop_record(run)}


def snapshot(root):
    paths = {*Path("src").rglob("*.py"), *Path("tests").rglob("*.py"), *Path(".").glob("*.py"),
             SPEC, Path("AGENTS.md"), Path("pyproject.toml")}
    manifest = {}
    with zipfile.ZipFile(root / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            archive.writestr(str(path), data)
            manifest[str(path)] = hashlib.sha256(data).hexdigest()
    _json(root / "source_manifest.json", manifest)
    return manifest


def save(root, contexts, runs):
    _json(root / "contexts.json", contexts)
    _json(root / "runs.json", runs)
    if runs:
        _csv(root / "runs.csv", runs)
    solver = sum(r["counts"]["total_oracle_calls"] for r in runs)
    setup = sum(c["setup_reference_values"] for c in contexts)
    counts = {"trajectory_executions": len(runs), "actual_solver_queries": solver,
              "setup_reference_values": setup, "actual_queries_including_setup": solver + setup}
    _json(root / "execution_counts.json", counts)
    return counts


def run(args):
    root, stage = args.output, args.stage
    dependencies = {str(PARENT): sha(PARENT)}
    if args.lock:
        prior = read(args.lock / "config.json")
        assert prior["stage"] == {"development": "preflight", "formal": "development"}[stage]
        audit = args.lock / "audit.json"
        assert read(audit)["passed"]
        dependencies[str(audit)] = sha(audit)
        if stage == "formal":
            model = args.lock / "model_lock.json"
            dependencies[str(model)] = sha(model)
    root.mkdir(parents=True)
    for name in ("traces", "events", "uncommitted"):
        (root / name).mkdir()
    source = snapshot(root)
    if args.lock:
        assert source == prior["source_sha256"]
    seeds = SEEDS[stage]
    _json(root / "config.json", {
        "phase": "E28", "stage": stage, "seeds": seeds, "initial_pivot_kinetics": list(ENERGIES),
        "condition": "good_present", "query_cap": CAP, "max_steps": STEPS, "h": .05, "gravity": .5,
        "stiffness": 8, "height": .025, "threshold": 1e-4, "forecast_end_step": 6, "initial_total_kinetic": .6,
        "policy": "keep_without_intervention", "binary_interactions": False,
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "previous_stage": str(args.lock) if args.lock else None,
        "launch_command": ["python3", *sys.argv], "cwd": str(Path.cwd()), "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "workers": args.workers,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    contexts, runs = [], []
    save(root, contexts, runs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate, root, seed, e) for seed in seeds for e in ENERGIES]
        try:
            for index, future in enumerate(futures, 1):
                context, result = future.result()
                contexts.append(context)
                runs.append(result)
                if index % 12 == 0 or index == len(futures):
                    counts = save(root, contexts, runs)
        except BaseException:
            save(root, contexts, runs)
            _json(root / "task_failure.json", {"traceback": traceback.format_exc(),
                  "unreturned_cost": "Unknown if the worker did not return its ledger."})
            for future in futures:
                future.cancel()
            raise
    _json(root / "completion_observation.json", {**counts, "observed_utc": datetime.now(timezone.utc).isoformat()})
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.stage == "preflight") == bool(args.lock):
        parser.error("Positive worker count required. Development and formal require --lock.")
    run(args)
