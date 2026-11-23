"""E50: resume the E48 window through one more natural reflection."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.event_replay import unpack
from collision_optim.population import PopulationOracle
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_capped_reflection_transfer import checked_hashes
from run_contact_warning_replication import continue_branch, snapshot
from run_mixed_post_contact_curvature import BASE as PARENT
from run_post_contact_curvature_screen import predict
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, CFG, H, LIMIT, START, STOP_STEP, read, sha

BASE = Path("output/collision_second_contact_reflection")
SPEC = Path("docs/superpowers/specs/2026-11-15-second-contact-reflection-design.md")
PREFLIGHT = {f"mixed__s3200__{suffix}" for suffix in ("energy_safe", "early_low", "early_high", "late_high")}
FORECAST_CAP = 5


def run(args, *, phase="E50", spec=SPEC, preflight=PREFLIGHT, forecast_cap=FORECAST_CAP,
        comparison=None, third_saved_point=False, earlier_archives=(), reflection_loss_bounds=None,
        parent=PARENT, fourth_saved_point=False):
    names = ("config", "audit", "contexts", "runs", "paired_outcomes", "physical_ledgers")
    dependencies = checked_hashes(parent, [parent/(n+".json") for n in names])
    manifest = parent/"raw_artifact_manifest.json"
    dependencies[str(manifest)] = sha(manifest)
    assert read(parent/"audit.json")["passed"]
    added_archives = (*earlier_archives, comparison) if comparison is not None else earlier_archives
    for folder in added_archives:
        paths = [folder/(n+".json") for n in ("config","audit","decisions","runs","paired_outcomes","physical_ledgers")]
        dependencies.update(checked_hashes(folder,paths))
        manifest = folder/"raw_artifact_manifest.json"
        dependencies[str(manifest)] = sha(manifest)
        assert read(folder/"audit.json")["passed"]
    contexts = read(parent/"contexts.json")
    if args.stage == "preflight":
        contexts = [c for c in contexts if c["context_id"] in preflight]
    assert len(contexts) == (len(preflight) if args.stage=="preflight" else 168)
    archived = {r["run_id"]:r for r in read(parent/"runs.json")}
    for folder in added_archives:
        extra = {r["run_id"]:r for r in read(folder/"runs.json")}
        assert not archived.keys() & extra.keys()
        archived.update(extra)
    if args.lock:
        lock = args.lock/"audit.json"
        assert read(lock)["passed"]
        dependencies[str(lock)] = sha(lock)
    root = args.output
    root.mkdir(parents=True)
    for name in ("traces", "events", "fractions", "uncommitted"):
        (root/name).mkdir()
    source = snapshot(root, spec)
    if args.lock:
        assert source == read(args.lock/"config.json")["source_sha256"]
    _json(root/"config.json", {
        "phase":phase, "stage":args.stage, "parent":str(parent), "new_independent_seed_groups":0,
        "comparison":str(comparison) if comparison is not None else None, "third_saved_point":third_saved_point,
        "earlier_archives":[str(p) for p in earlier_archives],
        "fourth_saved_point":fourth_saved_point,
        "reflection_loss_bounds":reflection_loss_bounds.__name__ if reflection_loss_bounds is not None else None,
        "context_ids":[c["context_id"] for c in contexts],
        "start":START, "boundary_step":BOUNDARY_STEP, "end_step":STOP_STEP, "h":H, "threshold":LIMIT,
        "configuration":asdict(CFG), "follow_second_contact":True, "transfer_external_energy_budget":0.,
        "forecast_query_cap_each":forecast_cap, "continuation_query_cap_each":ADDED_BUDGET,
        "max_new_queries":len(contexts)*(forecast_cap+ADDED_BUDGET),
        "source_sha256":source, "source_archive_sha256":sha(root/"source.zip"), "dependency_data_sha256":dependencies,
        "launch_command":["python3",*sys.argv], "cwd":str(Path.cwd()), "python":sys.version, "numpy":np.__version__,
        "platform":platform.platform(), "launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
    })
    decisions, runs, charges = [], [], Counter(forecast=0, continuation=0)
    try:
        for c in contexts:
            key = c["context_id"]
            objective = RotatedEnergyWells(8, c["seed"])
            state = unpack(c["start"])
            oracle = PopulationOracle(objective, total_budget=forecast_cap)
            try:
                forecast = predict(oracle, objective, state, c["initial_loss"],
                                   second_saved_point=True, follow_second_contact=True,
                                   third_saved_point=third_saved_point, reflection_loss_bounds=reflection_loss_bounds,
                                   fourth_saved_point=fourth_saved_point)
            finally:
                charges["forecast"] += sum(oracle.counts().values())
            mode = "capped" if forecast["warning"] else "keep"
            selected = key+"__"+mode
            reused = selected in archived
            decisions.append({"context_id":key, "forecast":forecast, "selected_run":selected, "reused":reused})
            if reused:
                continue
            moved = capped_transfer_at_checkpoint(state,0)
            oracle = PopulationOracle(objective, total_budget=ADDED_BUDGET)
            try:
                branch = continue_branch(root, key, mode, moved, oracle, CFG, BOUNDARY_STEP*H-START,
                                         end_step=STOP_STEP, boundary_step=BOUNDARY_STEP)
            finally:
                charges["continuation"] += sum(oracle.counts().values())
            runs.append(branch)
            if branch["stop_reason"] != "completed":
                raise RuntimeError(branch)
    except Exception:
        _json(root/"failure.json", {"context_id":key, "traceback":traceback.format_exc()})
        raise
    finally:
        _json(root/"decisions.json", decisions)
        _json(root/"runs.json", runs)
        if runs:
            _csv(root/"runs.csv", runs)
        cost = {"contexts":len(decisions), "continuations":len(runs),
                "reused_selected_continuations":sum(d["reused"] for d in decisions),
                "actual_new_forecast_queries":charges["forecast"], "actual_new_continuation_queries":charges["continuation"],
                "actual_new_queries":sum(charges.values())}
        _json(root/"execution_counts.json", cost)
    _json(root/"completion_observation.json", {**cost, "observed_utc":datetime.now(timezone.utc).isoformat()})
    print(cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight","formal"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if (args.stage=="formal") != bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args)
