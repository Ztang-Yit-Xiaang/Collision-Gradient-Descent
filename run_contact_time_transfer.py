"""E31: branch at the event within a step, then finish the step on the same clock."""

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

from collision_optim.contact_time_transfer import first_reflection, finish_fraction, forecast_to_boundary
from collision_optim.curvature_energy import IndependentEnergyPopulation, RotatedEnergyWells
from collision_optim.event_replay import pack, unpack
from collision_optim.gravity import GravityConfig
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, read, sha
from run_postbounce_warning_transfer import load_counters
from run_shared_prefix_transfer import save_segment, stop_record
from run_warning_triggered_transfer import CAP, CHECKPOINT_STEP as PARENT_STEP, END_STEP, H, LIMIT

SPEC = Path("docs/superpowers/specs/2026-11-11-contact-time-transfer-design.md")
PARENT = {
    "preflight": Path("output/collision_postbounce_warning_transfer_preflight"),
    "formal": Path("output/collision_postbounce_warning_transfer"),
}


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


def simulate(root, source, inherited, context, expected_hashes):
    name = context["context_id"]
    trace = source / "traces" / (name + "__keep.csv.gz")
    events = source / "events" / (name + "__keep.json")
    for path in (trace, events):
        assert sha(path) == expected_hashes[str(path)]
    boundary_step = context["checkpoint_step"]
    step_before = boundary_step - 1
    with gzip.open(trace, "rt") as stream:
        saved = list(islice(csv.DictReader(stream), step_before - PARENT_STEP + 1))
    previous = saved[-1]
    assert int(previous["step"]) + PARENT_STEP == step_before
    start = unpack({k: json.loads(previous[k]) for k in ("positions", "velocities")})
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    objective = RotatedEnergyWells(8, context["seed"])
    oracle = PopulationOracle(objective, total_budget=CAP)
    load_counters(oracle, previous)
    past_calls = sum(oracle.counts().values())
    outgoing, contact = first_reflection(oracle, start, cfg, step_before)
    after_contact = sum(oracle.counts().values())
    assert contact["index"] == 0 and after_contact - past_calls == 10
    initial_loss = inherited["initial_loss"]
    contact_rise = contact["contact_value"] - initial_loss
    remaining = H - contact["offset"]
    prediction = None if contact_rise > LIMIT else forecast_to_boundary(oracle, outgoing.particles[0], remaining, cfg)
    charges = oracle.counts()
    charged_at_branch = sum(charges.values())
    shared = {
        "context_id": name, "seed": context["seed"], "incumbent_energy": context["incumbent_energy"],
        "previous_step": step_before, "boundary_step": boundary_step, "remaining_duration": remaining,
        "contact": contact, "contact_time": step_before * H + contact["offset"],
        "initial_loss": initial_loss, "contact_rise": contact_rise, "already_deviated": contact_rise > LIMIT,
        "predicted_boundary_rise": prediction - initial_loss if prediction is not None else None,
        "warning": prediction is not None and prediction - initial_loss > LIMIT,
        "history_rises": inherited["prefix_loss_rises"] + [
            json.loads(row["values"])[0] - initial_loss for row in saved[1:]],
        "before_step": pack(start), "post_reflection": pack(outgoing),
        "source_trace": str(trace), "source_trace_sha256": expected_hashes[str(trace)],
        "source_event_sha256": expected_hashes[str(events)],
        "historical_prefix_calls": past_calls, "actual_event_reconstruction_calls": after_contact - past_calls,
        "actual_forecast_calls": charged_at_branch - after_contact, "charged_at_branch": charged_at_branch,
    }
    assert len(shared["history_rises"]) == step_before + 1 and max(shared["history_rises"]) <= LIMIT
    _json(root / "checkpoints" / (name + ".json"), shared)
    runs = []
    for mode in MODES:
        state = transfer_at_checkpoint(outgoing, 0) if mode == "transfer" else outgoing
        branch_oracle = PopulationOracle(objective, total_budget=CAP)
        load_counters(branch_oracle, charges)
        key = name + "__" + mode
        try:
            partial = finish_fraction(branch_oracle, state, remaining, cfg)
        except Exception:
            _json(root / "fractions" / (key + ".json"), {
                "failed": True, "counts": branch_oracle.counts(), "error": traceback.format_exc(),
            })
            raise
        partial_calls = sum(branch_oracle.counts().values())
        _json(root / "fractions" / (key + ".json"), {
            "start": pack(state), "end": pack(partial.state), "values": partial.values,
            "duration": remaining, "damping_beta": 1., "diagnostics": partial.diagnostics,
            "events": partial.events, "total_oracle_calls_at_end": partial_calls,
        })
        method = IndependentEnergyPopulation(cfg)
        method.committed_steps = boundary_step
        tail = run_population(method, branch_oracle, partial.state, END_STEP - boundary_step)
        save_segment(root, key, tail)
        runs.append({
            "run_id": key, "context_id": name, "seed": context["seed"], "policy": mode,
            "actual_fraction_calls": partial_calls - charged_at_branch,
            "actual_tail_calls": tail.counts["total_oracle_calls"] - partial_calls, **stop_record(tail),
        })
    return shared, runs


def save(root, checkpoints, runs):
    _json(root / "checkpoints.json", checkpoints)
    _json(root / "runs.json", runs)
    if runs:
        _csv(root / "runs.csv", runs)
    counts = {
        "reused_prefixes": len(checkpoints), "new_event_reconstructions": len(checkpoints),
        "suffix_executions": len(runs), "historical_prefix_queries": sum(c["historical_prefix_calls"] for c in checkpoints),
        "actual_event_reconstruction_queries": sum(c["actual_event_reconstruction_calls"] for c in checkpoints),
        "actual_forecast_queries": sum(c["actual_forecast_calls"] for c in checkpoints),
        "actual_fraction_queries": sum(r["actual_fraction_calls"] for r in runs),
        "actual_tail_queries": sum(r["actual_tail_calls"] for r in runs),
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
    }
    counts["actual_new_solver_queries"] = sum(counts[k] for k in (
        "actual_event_reconstruction_queries", "actual_forecast_queries", "actual_fraction_queries", "actual_tail_queries"))
    _json(root / "execution_counts.json", counts)
    return counts


def run(args):
    root, stage = args.output, args.stage
    parent = PARENT[stage]
    source = Path(read(parent / "config.json")["parent"])
    assert read(parent / "audit.json")["passed"] and read(source / "audit.json")["passed"]
    paths = [parent / n for n in ("config.json", "checkpoints.json", "already_deviated_cases.json", "paired_outcomes.json",
                                 "audit.json", "raw_artifact_manifest.json")]
    paths.extend(source / n for n in ("config.json", "checkpoints.json", "raw_artifact_manifest.json", "audit.json"))
    if args.lock:
        audit = args.lock / "audit.json"
        assert read(audit)["passed"]
        paths.append(audit)
    dependencies = {str(p): sha(p) for p in paths}
    identifiers = {r["context_id"] for r in read(parent / "already_deviated_cases.json")}
    contexts = [c for c in read(parent / "checkpoints.json") if stage == "preflight" or c["context_id"] in identifiers]
    assert len(contexts) == (4 if stage == "preflight" else 10)
    inherited = {c["context_id"]: c for c in read(source / "checkpoints.json")}
    parents = {r["path"]: r["sha256"] for r in read(source / "raw_artifact_manifest.json")["files"]}
    root.mkdir(parents=True)
    for folder in ("traces", "events", "uncommitted", "checkpoints", "fractions"):
        (root / folder).mkdir()
    source_hashes = snapshot(root)
    if args.lock:
        assert source_hashes == read(args.lock / "config.json")["source_sha256"]
    _json(root / "config.json", {
        "phase": "E31", "stage": stage, "parent": str(parent), "raw_parent_source": str(source),
        "new_independent_seed_groups": 0, "seed_groups_reused": sorted({c["seed"] for c in contexts}),
        "query_cap": CAP, "end_step": END_STEP, "h": H, "threshold": LIMIT, "binary_interactions": False,
        "source_sha256": source_hashes, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "launch_command": ["python3", *sys.argv], "cwd": str(Path.cwd()),
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "workers": args.workers, "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    checkpoints, runs = [], []
    save(root, checkpoints, runs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for c in contexts:
            name = c["context_id"]
            files = (source / "traces" / (name + "__keep.csv.gz"), source / "events" / (name + "__keep.json"))
            hashes = {str(file): parents[str(file)] for file in files}
            jobs.append(pool.submit(simulate, root, source, inherited[name], c, hashes))
        try:
            for job in jobs:
                shared, pair = job.result()
                checkpoints.append(shared)
                runs.extend(pair)
                counts = save(root, checkpoints, runs)
        except BaseException:
            _json(root / "task_failure.json", {"traceback": traceback.format_exc(),
                  "unreturned_cost": "Unknown if the worker did not return its ledger."})
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
