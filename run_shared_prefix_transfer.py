"""E27: branch once after a common, paid observation trajectory."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
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

from collision_optim.curvature_energy import CONDITIONS, IndependentEnergyPopulation, initial_state
from collision_optim.energy_generalization import controlled_velocities, generalization_score
from collision_optim.event_experiment import save_trace
from collision_optim.event_replay import pack
from collision_optim.gravity import GravityConfig
from collision_optim.initial_budget import allocate_initial
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, loss_gap_score, read, sha
from run_overlap_transfer import build_context

SPEC = Path("docs/superpowers/specs/2026-11-10-shared-prefix-transfer-design.md")
PARENT = Path("output/collision_overlap_transfer/config.json")
SEEDS = {"preflight": [2090], "development": list(range(2100, 2112)), "formal": list(range(2120, 2140))}
ENERGIES = (.001, .04, .12)
PREFIX_STEPS = 10
MODEL_FEATURES = ("loss_score", "horizontal_impulse")
ALL_FEATURES = (*MODEL_FEATURES, "relative_kinetic")


def snapshot(root):
    files = {*Path("src").rglob("*.py"), *Path("tests").rglob("*.py"), *Path(".").glob("*.py"),
             SPEC, Path("AGENTS.md"), Path("pyproject.toml")}
    manifest = {}
    with zipfile.ZipFile(root / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            data = path.read_bytes()
            archive.writestr(str(path), data)
            manifest[str(path)] = hashlib.sha256(data).hexdigest()
    _json(root / "source_manifest.json", manifest)
    return manifest


def stop_record(result):
    return {
        "stop_reason": result.stop_reason, "counts": result.counts,
        "failed": result.stop_reason.startswith(("contact_error:", "numerical_error:")),
        "wall_seconds": result.wall_seconds, "committed_steps": len(result.rows) - 1,
    }


def save_segment(root, key, result):
    save_trace(root, key, result.rows, result.events, uncommitted={
        "stop_reason": result.stop_reason, "last_committed_state": pack(result.final_state),
        "counts": result.counts, "partial_intermediate_states": "Not returned by the solver",
    })


def simulate(root, condition, seed, energy, budget, steps, preflight):
    objective, points, values, _, context = build_context(condition, seed, 300, 100)
    velocities = controlled_velocities(seed, energy)
    context.update(
        context_id=f"{condition}__s{seed}__k{energy:g}", incumbent_energy=energy,
        initial_horizontal_velocities=velocities.tolist(),
    )
    context.pop("conditional_source_policy")
    context.pop("threshold")
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    allocation = allocate_initial(values, (0.,) * 4, "fixed_ref_keep", velocities)
    initial = initial_state(points, values, allocation, velocities)
    oracle = PopulationOracle(objective, total_budget=budget)
    oracle.values = 4
    method = IndependentEnergyPopulation(cfg)
    prefix = run_population(method, oracle, initial, PREFIX_STEPS)
    key = context["context_id"]
    prefix_id = key + "__prefix"
    save_segment(root, prefix_id, prefix)
    prefix_record = {"run_id": prefix_id, "context_id": key, **stop_record(prefix)}
    context.update(prefix_run_id=prefix_id, population_config=asdict(cfg), start=pack(initial))
    if len(prefix.rows) != PREFIX_STEPS + 1:
        context["unavailable_reason"] = prefix.stop_reason
        _json(root / "contexts" / (key + ".json"), context)
        return context, prefix_record, [], []

    checkpoint = prefix.final_state
    packed = pack(checkpoint)
    latest_values = prefix.rows[-1]["values"]
    selected = int(np.argmin(latest_values))
    selected_events = [e for e in prefix.events if e["kind"] == "wall" and e["i"] == selected]
    net_impulse = sum((np.array(e["after_i"][:2]) - e["before_i"][:2] for e in selected_events), np.zeros(2))
    relative_velocity = checkpoint.particles[selected].v - checkpoint.momentum[:2] / 4
    context.update(checkpoint=packed, checkpoint_values=latest_values, selected=selected,
                   prefix_calls=prefix.counts["total_oracle_calls"], features={
                       "loss_score": loss_gap_score(latest_values),
                       "horizontal_impulse": float(np.linalg.norm(net_impulse)),
                       "relative_kinetic": float(.5 * (relative_velocity @ relative_velocity)),
                   })
    _json(root / "contexts" / (key + ".json"), context)
    branches = []
    for mode in MODES:
        start = transfer_at_checkpoint(checkpoint, selected) if mode == "transfer" else checkpoint
        branch_method = IndependentEnergyPopulation(cfg)
        branch_method.committed_steps = PREFIX_STEPS
        branch = run_population(branch_method, deepcopy(oracle), start, steps - PREFIX_STEPS)
        run_id = key + "__" + mode
        save_segment(root, run_id, branch)
        _json(root / "interventions" / (run_id + ".json"), {
            "prefix_run_id": prefix_id, "start": pack(start), "selected": selected, "time": PREFIX_STEPS * .05,
            "method": mode, "damping_clock_at_start": PREFIX_STEPS,
            "velocity_delta": (np.array(pack(start)["velocities"]) - packed["velocities"]).tolist(),
        })
        branch_rows = [{**r, "time": (r["step"] + PREFIX_STEPS) * .05} for r in branch.rows[1:]]
        score = generalization_score(prefix.rows + branch_rows, context, objective)
        branches.append({
            "run_id": run_id, "context_id": key, "policy": mode, "condition": condition, "seed": seed,
            "incumbent_energy": energy, "prefix_calls": context["prefix_calls"],
            "actual_suffix_calls": branch.counts["total_oracle_calls"] - context["prefix_calls"],
            "prefix_max_rise": max(0., max(row["values"][0] - values[0] for row in prefix.rows)),
            **stop_record(branch), **score,
        })
    direct = []
    if preflight:
        direct_oracle = PopulationOracle(objective, total_budget=budget)
        direct_oracle.values = 4
        control = run_population(IndependentEnergyPopulation(cfg), direct_oracle, initial, steps)
        direct_id = key + "__uninterrupted"
        save_segment(root, direct_id, control)
        direct.append({"run_id": direct_id, "context_id": key, **stop_record(control)})
    return context, prefix_record, branches, direct


def save_tables(root, contexts, prefixes, runs, uninterrupted):
    for name, table in (("contexts", contexts), ("prefixes", prefixes), ("runs", runs), ("uninterrupted", uninterrupted)):
        _json(root / (name + ".json"), table)
    if runs:
        _csv(root / "runs.csv", runs)
    prefix_calls = sum(r["counts"]["total_oracle_calls"] for r in prefixes)
    suffix_calls = sum(r["actual_suffix_calls"] for r in runs)
    direct_calls = sum(r["counts"]["total_oracle_calls"] - 4 for r in uninterrupted)
    setup = sum(c["setup_reference_values"] for c in contexts)
    counts = {
        "contexts": len(contexts), "shared_prefix_executions": len(prefixes), "suffix_executions": len(runs),
        "uninterrupted_replication_executions": len(uninterrupted),
        "shared_prefix_queries_including_initial_values": prefix_calls,
        "actual_suffix_queries": suffix_calls, "actual_uninterrupted_queries_excluding_shared_initial_values": direct_calls,
        "setup_reference_values": setup, "actual_solver_queries": prefix_calls + suffix_calls + direct_calls,
        "actual_queries_including_setup": prefix_calls + suffix_calls + direct_calls + setup,
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
    }
    _json(root / "execution_counts.json", counts)
    return counts


def run(args):
    stage = args.stage
    root = args.output
    preflight = stage == "preflight"
    energies = (.04,) if preflight else ENERGIES
    conditions = ("bad_only",) if stage == "development" else CONDITIONS
    dependencies = {str(PARENT): sha(PARENT)}
    if args.lock:
        prior = read(args.lock / "config.json")
        assert prior["stage"] == {"development": "preflight", "formal": "development"}[stage]
        audit = args.lock / "audit.json"
        assert read(audit)["passed"]
        dependencies[str(audit)] = sha(audit)
        if stage == "formal":
            selection = args.lock / "selection.json"
            energies = (read(selection)["energy"],)
            dependencies[str(selection)] = sha(selection)
    root.mkdir(parents=True)
    for name in ("traces", "events", "contexts", "uncommitted", "interventions"):
        (root / name).mkdir()
    source = snapshot(root)
    if args.lock:
        assert source == prior["source_sha256"]
    budget, steps = (1200, 80) if preflight else (6000, 500)
    seeds = SEEDS[stage]
    _json(root / "config.json", {
        "phase": "E27", "stage": stage, "seeds": seeds, "initial_pivot_kinetics": list(energies),
        "conditions": list(conditions), "h": .05, "gravity": .5, "stiffness": 8, "initial_kinetic": .6,
        "fixed_height": .025, "initial_relative_loss_score": 300, "prefix_steps": PREFIX_STEPS,
        "query_cap": budget, "max_steps": steps, "binary_interactions": False,
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "previous_stage": str(args.lock) if args.lock else None,
        "launch_command": ["python3", *sys.argv], "cwd": str(Path.cwd()), "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "workers": args.workers,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    contexts, prefixes, runs, uninterrupted = [], [], [], []
    save_tables(root, contexts, prefixes, runs, uninterrupted)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(simulate, root, condition, seed, energy, budget, steps, preflight)
                for seed in seeds for condition in conditions for energy in energies]
        try:
            for index, job in enumerate(as_completed(jobs), 1):
                context, prefix, branches, direct = job.result()
                contexts.append(context)
                prefixes.append(prefix)
                runs.extend(branches)
                uninterrupted.extend(direct)
                counts = save_tables(root, contexts, prefixes, runs, uninterrupted)
                if index % 8 == 0 or index == len(jobs):
                    print(f"{stage}: {index}/{len(jobs)} shared prefixes, {len(runs)} continuations", flush=True)
        except BaseException:
            _json(root / "task_failure.json", {"traceback": traceback.format_exc(),
                  "incomplete_cost": "Unknown for workers without returned results"})
            for job in jobs:
                job.cancel()
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
