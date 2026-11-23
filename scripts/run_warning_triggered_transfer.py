"""E29: replay E28 checkpoints and intervene only after the frozen alert."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import islice
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
from collision_optim.prebounce_warning import flight_factors, forecast_rise
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, read, sha
from run_shared_prefix_transfer import save_segment, stop_record

SPEC = Path("docs/superpowers/specs/2026-11-11-warning-triggered-transfer-design.md")
PARENT = {
    "preflight": Path("output/collision_prebounce_warning_preflight"),
    "formal": Path("output/collision_prebounce_warning"),
}
CHECKPOINT_STEP = 2
HORIZON_STEP = 6
END_STEP = 20
H = .05
LIMIT = 1e-4
CAP = 480


def snapshot(root):
    paths = {*Path("src").rglob("*.py"), *Path("tests").rglob("*.py"), *Path(".").glob("*.py"), *Path("scripts").glob("*.py"),
             SPEC, Path("AGENTS.md"), Path("pyproject.toml")}
    manifest = {}
    with zipfile.ZipFile(root / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            archive.writestr(str(path), data)
            manifest[str(path)] = hashlib.sha256(data).hexdigest()
    _json(root / "source_manifest.json", manifest)
    return manifest


def parent_prefix(parent, context_id):
    path = parent / "traces" / (context_id + ".csv.gz")
    with gzip.open(path, "rt") as stream:
        rows = list(islice(csv.DictReader(stream), CHECKPOINT_STEP + 1))
    assert [int(row["step"]) for row in rows] == list(range(CHECKPOINT_STEP + 1))
    event_path = parent / "events" / (context_id + ".json")
    events = [event for event in read(event_path) if event["step"] <= CHECKPOINT_STEP]
    assert not events
    return rows, path, event_path


def simulate(root, parent, context, preflight, parent_hashes):
    name = context["context_id"]
    prefix, trace_path, event_path = parent_prefix(parent, name)
    for path in (trace_path, event_path):
        assert sha(path) == parent_hashes[str(path)]
    initial_speed = float(np.linalg.norm(json.loads(prefix[0]["velocities"])[0][:2]))
    losses = [json.loads(row["values"])[0] for row in prefix]
    rises = [value - losses[0] for value in losses]
    risk = forecast_rise(rises[1:], initial_speed, flight_factors(steps=HORIZON_STEP))
    status = "already_deviated" if max(rises) > LIMIT else ("alert" if risk > LIMIT else "no_alert")
    last = prefix[-1]
    checkpoint = {"positions": json.loads(last["positions"]), "velocities": json.loads(last["velocities"])}
    prefix_calls = int(last["total_oracle_calls"])
    assert prefix_calls == 32
    state = unpack(checkpoint)
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    objective = RotatedEnergyWells(8, context["seed"])
    assert np.array_equal(objective.rotation, context["rotation"])
    shared = {
        "context_id": name, "status": status, "risk": risk, "prefix_calls": prefix_calls,
        "initial_loss": losses[0], "prefix_loss_rises": rises, "checkpoint": checkpoint, "selected": 0,
        "checkpoint_time": CHECKPOINT_STEP * H, "seed": context["seed"],
        "incumbent_energy": context["incumbent_energy"], "source_parent_trace": str(trace_path),
        "source_parent_trace_sha256": parent_hashes[str(trace_path)],
        "source_parent_event_sha256": parent_hashes[str(event_path)],
    }
    _json(root / "checkpoints" / (name + ".json"), shared)
    modes = MODES if preflight or status != "no_alert" else ("keep",)
    runs = []
    for mode in modes:
        start = transfer_at_checkpoint(state, 0) if mode == "transfer" else state
        oracle = PopulationOracle(objective, total_budget=CAP)
        oracle.values, oracle.gradients, oracle.paths = (
            int(last[k]) for k in ("value_calls", "gradient_calls", "path_calls")
        )
        method = IndependentEnergyPopulation(cfg)
        method.committed_steps = CHECKPOINT_STEP
        result = run_population(method, oracle, start, END_STEP - CHECKPOINT_STEP)
        key = name + "__" + mode
        save_segment(root, key, result)
        _json(root / "interventions" / (key + ".json"), {
            "start": pack(start), "selected": 0, "mode": mode, "damping_clock_at_start": CHECKPOINT_STEP,
            "velocity_delta": (np.array(pack(start)["velocities"]) - checkpoint["velocities"]).tolist(),
        })
        runs.append({
            "run_id": key, "context_id": name, "policy": mode, "status": status,
            "seed": shared["seed"], "incumbent_energy": shared["incumbent_energy"],
            "actual_suffix_calls": result.counts["total_oracle_calls"] - prefix_calls, **stop_record(result),
        })
    return shared, runs


def save(root, checkpoints, runs):
    _json(root / "checkpoints.json", checkpoints)
    _json(root / "runs.json", runs)
    if runs:
        _csv(root / "runs.csv", runs)
    counts = {
        "reused_prefixes": len(checkpoints), "new_prefix_executions": 0, "suffix_executions": len(runs),
        "historical_prefix_queries": sum(c["prefix_calls"] for c in checkpoints),
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
        "actual_new_solver_queries": sum(r["actual_suffix_calls"] for r in runs),
        "actual_new_setup_queries": 0,
    }
    _json(root / "execution_counts.json", counts)
    return counts


def run(args):
    root, stage = args.output, args.stage
    parent = PARENT[stage]
    preflight = stage == "preflight"
    assert read(parent / "audit.json")["passed"]
    source_files = ["config.json", "contexts.json", "labeled_observations.json", "audit.json", "raw_artifact_manifest.json"]
    if not preflight:
        assert read(args.lock / "audit.json")["passed"]
        source_files.append("results.json")
    dependencies = {str(parent / f): sha(parent / f) for f in source_files}
    if args.lock:
        dependencies[str(args.lock / "audit.json")] = sha(args.lock / "audit.json")
    raw = {a["path"]: a["sha256"] for a in read(parent / "raw_artifact_manifest.json")["files"]}
    root.mkdir(parents=True)
    for folder in ("traces", "events", "uncommitted", "checkpoints", "interventions"):
        (root / folder).mkdir()
    source = snapshot(root)
    if args.lock:
        assert source == read(args.lock / "config.json")["source_sha256"]
    source_contexts = read(parent / "contexts.json")
    _json(root / "config.json", {
        "phase": "E29", "stage": stage, "parent": str(parent), "checkpoint_step": CHECKPOINT_STEP,
        "end_step": END_STEP, "forecast_end_step": HORIZON_STEP, "h": H, "threshold": LIMIT, "query_cap": CAP,
        "same_historical_seeds": read(parent / "config.json")["seeds"], "new_independent_seed_groups": 0,
        "reused_parent_contexts": len(source_contexts), "binary_interactions": False,
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "launch_command": ["python3", *sys.argv],
        "cwd": str(Path.cwd()), "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__, "workers": args.workers,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    checkpoints, runs = [], []
    save(root, checkpoints, runs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for context in source_contexts:
            key = context["context_id"]
            files = (parent / "traces" / (key + ".csv.gz"), parent / "events" / (key + ".json"))
            hashes = {str(f): raw[str(f)] for f in files}
            futures.append(pool.submit(simulate, root, parent, context, preflight, hashes))
        try:
            for index, future in enumerate(futures, 1):
                checkpoint, pair = future.result()
                checkpoints.append(checkpoint)
                runs.extend(pair)
                if index % 40 == 0 or index == len(futures):
                    counts = save(root, checkpoints, runs)
                    print(f"{stage}: {index}/{len(futures)} frozen checkpoints, {len(runs)} continuations", flush=True)
        except BaseException:
            save(root, checkpoints, runs)
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
    parser.add_argument("--stage", choices=PARENT, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.stage == "preflight") == bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args)
