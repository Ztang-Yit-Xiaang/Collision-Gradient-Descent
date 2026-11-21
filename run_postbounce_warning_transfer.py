"""E30: forecast from the first saved rebound state and run matched continuations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile

import numpy as np

from collision_optim.curvature_energy import IndependentEnergyPopulation, RotatedEnergyWells
from collision_optim.event_replay import pack, unpack
from collision_optim.gravity import GravityConfig
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.postbounce_warning import forecast_after_bounce, normal_quadratic_score
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, read, sha
from run_shared_prefix_transfer import save_segment, stop_record
from run_warning_triggered_transfer import CAP, CHECKPOINT_STEP as PARENT_STEP, END_STEP, H, LIMIT

SPEC = Path("docs/superpowers/specs/2026-11-11-postbounce-warning-transfer-design.md")
PARENT = {
    "preflight": Path("output/collision_warning_triggered_transfer_preflight"),
    "formal": Path("output/collision_warning_triggered_transfer"),
}
CURVATURE_SOURCE = Path("output/collision_prebounce_warning_development/model_lock.json")
APPROX_END_STEP = 18
MODELS = ("normal_quadratic", "one_path", "through_next_reflection")


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


def checkpoint_data(parent, context, expected_hashes):
    key = context["context_id"] + "__keep"
    trace = parent / "traces" / (key + ".csv.gz")
    events = parent / "events" / (key + ".json")
    for path in (trace, events):
        assert sha(path) == expected_hashes[str(path)]
    contact = next(e for e in read(events) if e["kind"] == "wall" and e["i"] == 0)
    time = PARENT_STEP * H + contact["time"]
    with gzip.open(trace, "rt") as stream:
        prefix = []
        for row in csv.DictReader(stream):
            prefix.append(row)
            if (PARENT_STEP + int(row["step"])) * H > time:
                break
    last = prefix[-1]
    step = PARENT_STEP + int(last["step"])
    assert step * H > time and step < APPROX_END_STEP
    return trace, events, contact, time, step, prefix


def load_counters(oracle, row):
    oracle.values, oracle.gradients, oracle.paths = (
        int(row[name]) for name in ("value_calls", "gradient_calls", "path_calls")
    )


def simulate(root, parent, context, expected_hashes, curvature):
    name = context["context_id"]
    trace, event_path, contact, contact_time, step, prefix = checkpoint_data(parent, context, expected_hashes)
    row = prefix[-1]
    checkpoint = {"positions": json.loads(row["positions"]), "velocities": json.loads(row["velocities"])}
    state = unpack(checkpoint)
    initial_loss = context["initial_loss"]
    rises = context["prefix_loss_rises"] + [json.loads(r["values"])[0] - initial_loss for r in prefix[1:]]
    assert len(rises) == step + 1
    late = max(rises) > LIMIT
    objective = RotatedEnergyWells(8, context["seed"])
    forecast_oracle = PopulationOracle(objective, total_budget=CAP)
    load_counters(forecast_oracle, row)
    past_calls = forecast_oracle.counts()
    forecast = None
    warnings = None
    if not late:
        approximate_score = normal_quadratic_score(state.particles[0], json.loads(row["values"])[0],
                                                  initial_loss, contact, curvature, step, APPROX_END_STEP)
        forecast = forecast_after_bounce(forecast_oracle, state.particles[0], initial_loss, LIMIT, step, END_STEP)
        warnings = {
            "normal_quadratic": approximate_score > LIMIT,
            "one_path": any(x["loss_rise"] > LIMIT and not x["after_next_contact"] for x in forecast["observations"]),
            "through_next_reflection": any(x["loss_rise"] > LIMIT for x in forecast["observations"]),
        }
        forecast["normal_quadratic_score"] = approximate_score
    charge = forecast_oracle.counts()
    forecast_calls = {key: charge[key] - past_calls[key] for key in past_calls}
    shared = {
        "context_id": name, "seed": context["seed"], "incumbent_energy": context["incumbent_energy"],
        "source_keep_run": name + "__keep", "source_trace": str(trace),
        "source_trace_sha256": expected_hashes[str(trace)], "source_event_sha256": expected_hashes[str(event_path)],
        "initial_loss": initial_loss, "prefix_rises": rises, "checkpoint": checkpoint,
        "checkpoint_step": step, "checkpoint_time": step * H, "first_contact_time": contact_time,
        "first_contact_loss_rise": contact["point_i"][2] - initial_loss, "already_deviated": late,
        "warnings": warnings, "forecast": forecast, "forecast_calls": forecast_calls,
        "historical_prefix_calls": sum(past_calls.values()), "charged_at_branch": sum(charge.values()),
    }
    _json(root / "checkpoints" / (name + ".json"), shared)
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    branches = []
    for mode in MODES:
        start = transfer_at_checkpoint(state, 0) if mode == "transfer" else state
        oracle = PopulationOracle(objective, total_budget=CAP)
        load_counters(oracle, charge)
        method = IndependentEnergyPopulation(cfg)
        method.committed_steps = step
        result = run_population(method, oracle, start, END_STEP - step)
        key = name + "__" + mode
        save_segment(root, key, result)
        _json(root / "interventions" / (key + ".json"), {
            "start": pack(start), "selected": 0, "mode": mode, "damping_clock_at_start": step,
            "velocity_delta": (np.array(pack(start)["velocities"]) - checkpoint["velocities"]).tolist(),
        })
        branches.append({
            "run_id": key, "context_id": name, "seed": context["seed"], "policy": mode,
            "actual_suffix_calls": result.counts["total_oracle_calls"] - shared["charged_at_branch"],
            **stop_record(result),
        })
    return shared, branches


def save(root, checkpoints, runs):
    _json(root / "checkpoints.json", checkpoints)
    _json(root / "runs.json", runs)
    if runs:
        _csv(root / "runs.csv", runs)
    forecast = sum(sum(c["forecast_calls"].values()) for c in checkpoints)
    suffix = sum(r["actual_suffix_calls"] for r in runs)
    counts = {
        "reused_prefixes": len(checkpoints), "new_prefix_executions": 0, "suffix_executions": len(runs),
        "historical_prefix_queries": sum(c["historical_prefix_calls"] for c in checkpoints),
        "forecast_path_queries": sum(c["forecast_calls"]["path_calls"] for c in checkpoints),
        "forecast_gradient_queries": sum(c["forecast_calls"]["gradient_calls"] for c in checkpoints),
        "actual_forecast_queries": forecast, "actual_suffix_queries": suffix,
        "actual_new_solver_queries_including_forecast": forecast + suffix,
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
    }
    _json(root / "execution_counts.json", counts)
    return counts


def run(args):
    root, stage = args.output, args.stage
    parent = PARENT[stage]
    assert read(parent / "audit.json")["passed"]
    paths = [parent / name for name in ("config.json", "checkpoints.json", "paired_outcomes.json",
                                        "audit.json", "raw_artifact_manifest.json")]
    paths.append(CURVATURE_SOURCE)
    if args.lock:
        previous = args.lock / "audit.json"
        assert read(previous)["passed"]
        paths.append(previous)
    dependencies = {str(p): sha(p) for p in paths}
    curvature = read(CURVATURE_SOURCE)["prior_curvature"]
    contexts = [c for c in read(parent / "checkpoints.json") if c["status"] == "no_alert"]
    assert len(contexts) == (4 if stage == "preflight" else 136)
    parent_manifest = {a["path"]: a["sha256"] for a in read(parent / "raw_artifact_manifest.json")["files"]}
    root.mkdir(parents=True)
    for folder in ("traces", "events", "uncommitted", "checkpoints", "interventions"):
        (root / folder).mkdir()
    source = snapshot(root)
    if args.lock:
        assert source == read(args.lock / "config.json")["source_sha256"]
    _json(root / "config.json", {
        "phase": "E30", "stage": stage, "parent": str(parent), "reused_parent_contexts": len(contexts),
        "new_independent_seed_groups": 0, "seed_groups_reused": sorted({c["seed"] for c in contexts}),
        "query_cap": CAP, "end_step": END_STEP, "h": H, "threshold": LIMIT, "binary_interactions": False,
        "normal_quadratic_end_step": APPROX_END_STEP, "normal_quadratic_prior_curvature": curvature,
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "launch_command": ["python3", *sys.argv], "cwd": str(Path.cwd()),
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "workers": args.workers,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    checkpoints, runs = [], []
    save(root, checkpoints, runs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for c in contexts:
            key = c["context_id"] + "__keep"
            paths = (parent / "traces" / (key + ".csv.gz"), parent / "events" / (key + ".json"))
            hashes = {str(path): parent_manifest[str(path)] for path in paths}
            jobs.append(pool.submit(simulate, root, parent, c, hashes, curvature))
        try:
            for index, job in enumerate(jobs, 1):
                shared, pair = job.result()
                checkpoints.append(shared)
                runs.extend(pair)
                if index % 24 == 0 or index == len(jobs):
                    counts = save(root, checkpoints, runs)
                    print(f"{stage}: {index}/{len(jobs)} paired contexts", flush=True)
        except BaseException:
            save(root, checkpoints, runs)
            _json(root / "task_failure.json", {"traceback": traceback.format_exc(),
                  "unreturned_cost": "Unknown if the worker did not return its ledger."})
            for job in jobs:
                job.cancel()
            raise
    _json(root / "completion_observation.json", {**counts, "observed_utc": datetime.now(timezone.utc).isoformat()})
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=PARENT, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.stage == "preflight") == bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args)
