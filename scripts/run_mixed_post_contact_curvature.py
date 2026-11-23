"""E49 paired E47/E48 comparison on fresh mixed launch states."""

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

from collision_optim.population import PopulationOracle
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_contact_warning_replication import continue_branch, snapshot
from run_post_contact_curvature_screen import BASE as E47, predict
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, CFG, H, LIMIT, START, STOP_STEP, build, read, sha
from run_second_post_contact_curvature import BASE as E48

BASE = Path("output/collision_mixed_post_contact_curvature")
SPEC = Path("docs/superpowers/specs/2026-11-15-mixed-post-contact-curvature-design.md")
SEEDS = {"preflight": (3180,), "formal": tuple(range(3200, 3224))}
STRATA = (
    ("energy_safe", (1.562, 1.567), (.00002, .00020)),
    ("early_low", (1.602, 1.612), (.005, .30)),
    ("early_mid", (1.602, 1.612), (.30, .80)),
    ("early_high", (1.602, 1.612), (.80, 1.05)),
    ("late_low", (1.633, 1.647), (.005, .38)),
    ("late_mid", (1.633, 1.647), (.38, .65)),
    ("late_high", (1.633, 1.647), (.65, 1.05)),
)
POLICIES = ("e47", "e48")
FORECAST_OPTIONS = {"e47": {}, "e48": {"second_saved_point": True}}


def build_context(seed, case):
    label, time, ratio = STRATA[case]
    rng = np.random.default_rng([seed, case])
    objective, state, row = build(seed, case, contact_case=(rng.uniform(*time), rng.uniform(*ratio)))
    row["stratum"] = label
    row["context_id"] = f"mixed__s{seed}__{label}"
    return objective, state, row


def simulate(root, seed, case, costs, contexts, runs, *, forecast_options, forecast_cap):
    objective, state, row = build_context(seed, case)
    costs["setup_queries"] += sum(row["setup_counts"].values())
    for policy, options in forecast_options.items():
        oracle = PopulationOracle(objective, total_budget=forecast_cap)
        try:
            row[policy] = predict(oracle, objective, state, row["initial_loss"], **options)
        finally:
            costs[policy + "_forecast_queries"] += sum(oracle.counts().values())
    contexts.append(row)
    branches = {"keep": state}
    if any(row[p]["warning"] for p in forecast_options):
        branches["capped"] = capped_transfer_at_checkpoint(state, 0)
    for mode, start in branches.items():
        oracle = PopulationOracle(objective, total_budget=ADDED_BUDGET)
        try:
            branch = continue_branch(root, row["context_id"], mode, start, oracle, CFG, BOUNDARY_STEP*H-START,
                                     end_step=STOP_STEP, boundary_step=BOUNDARY_STEP)
        finally:
            costs["continuation_queries"] += sum(oracle.counts().values())
        runs.append(branch)
        if branch["stop_reason"] != "completed":
            raise RuntimeError(branch)


def save(root, contexts, runs, costs):
    _json(root / "contexts.json", contexts)
    _json(root / "runs.json", runs)
    if runs:
        _csv(root / "runs.csv", runs)
    total = {"contexts": len(contexts), "continuations": len(runs), **costs, "actual_new_queries": sum(costs.values())}
    _json(root / "execution_counts.json", total)
    return total


def run(args, *, phase="E49", spec=SPEC, seeds=SEEDS, parents=(E47,E48),
        forecast_options=FORECAST_OPTIONS, forecast_cap=3):
    jobs = [(seed, case) for seed in seeds[args.stage] for case in range(len(STRATA))]
    paths = [p / n for p in parents for n in ("raw_artifact_manifest.json", "config.json")]
    if args.lock:
        lock = args.lock / "audit.json"
        assert read(lock)["passed"]
        paths.append(lock)
    root = args.output
    root.mkdir(parents=True)
    for name in ("traces", "events", "fractions", "uncommitted"):
        (root / name).mkdir()
    source = snapshot(root, spec)
    if args.lock:
        assert source == read(args.lock / "config.json")["source_sha256"]
    _json(root / "config.json", {
        "phase": phase, "stage": args.stage, "jobs": jobs, "strata": STRATA,
        "forecast_options": {p: {k: v.__name__ if callable(v) else v for k,v in opts.items()}
                             for p,opts in forecast_options.items()},
        "start": START, "boundary_step": BOUNDARY_STEP, "end_step": STOP_STEP, "h": H,
        "configuration": asdict(CFG), "threshold": LIMIT, "transfer_external_energy_budget": 0.,
        "max_continuation_queries_each": ADDED_BUDGET, "max_policy_forecast_queries_each": forecast_cap,
        "max_new_queries": len(jobs)*(4 + len(forecast_options)*forecast_cap + 2*ADDED_BUDGET),
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": {str(p): sha(p) for p in paths},
        "launch_command": ["python3", *sys.argv], "cwd": str(Path.cwd()), "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in
                               ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    contexts, runs, costs = [], [], Counter()
    try:
        for seed, case in jobs:
            simulate(root, seed, case, costs, contexts, runs, forecast_options=forecast_options,
                     forecast_cap=forecast_cap)
            total = save(root, contexts, runs, costs)
    except Exception:
        total = save(root, contexts, runs, costs)
        _json(root / "failure.json", {"seed": seed, "case": case, "traceback": traceback.format_exc(),
                                      "execution_counts": total})
        raise
    _json(root / "completion_observation.json", {**total, "observed_utc": datetime.now(timezone.utc).isoformat()})
    print(total, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if (args.stage == "formal") != bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args)
