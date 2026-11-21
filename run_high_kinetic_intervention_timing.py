"""E34: compare first-step and first-reflection transfer with one common start."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from collision_optim.contact_time_transfer import first_reflection
from collision_optim.curvature_energy import IndependentEnergyPopulation, initial_state
from collision_optim.energy_generalization import controlled_velocities
from collision_optim.event_replay import pack, unpack
from collision_optim.gravity import GravityConfig
from collision_optim.initial_budget import allocate_initial
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.prebounce_warning import flight_factors, forecast_rise
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, read, sha
from run_contact_warning_replication import PRECONTACT_STEP, continue_branch, snapshot
from run_overlap_transfer import build_context
from run_postbounce_warning_transfer import load_counters
from run_shared_prefix_transfer import save_segment, stop_record
from run_warning_triggered_transfer import H, LIMIT

SPEC = Path("docs/superpowers/specs/2026-11-11-high-kinetic-intervention-timing-design.md")
PARENT = Path("output/collision_contact_long_forecast")
WARNING_SOURCE = Path("output/collision_prebounce_warning_development/model_lock.json")
SEEDS = {"preflight": [2430], "formal": list(range(2440, 2480))}
PRIMARY_STEP = 20
END_STEP = 40
CAP = 960
ENERGY = .001


def simulate(root, seed, curvature, preflight):
    objective, points, values, _, c = build_context("good_present", seed, 300, 100)
    v = controlled_velocities(seed, ENERGY)
    initial = initial_state(points, values, allocate_initial(values, (0.,)*4, "fixed_ref_keep", v), v)
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    oracle = PopulationOracle(objective, total_budget=CAP)
    oracle.values = 4
    prefix = run_population(IndependentEnergyPopulation(cfg), oracle, initial, PRECONTACT_STEP)
    name = f"good_present__s{seed}__k{ENERGY:g}"
    prefix_id = name + "__prefix"
    save_segment(root, prefix_id, prefix)
    p_record = {"run_id": prefix_id, "context_id": name, **stop_record(prefix)}
    assert p_record["committed_steps"] == PRECONTACT_STEP and not prefix.events
    prefix_calls = sum(oracle.counts().values())
    outgoing, contact = first_reflection(oracle, prefix.final_state, cfg, PRECONTACT_STEP)
    assert contact["index"] == 0
    contact_cost = sum(oracle.counts().values()) - prefix_calls
    assert contact_cost == 10
    remaining = H - contact["offset"]
    history = [row["values"][0] - values[0] for row in prefix.rows]
    contact_rise = contact["contact_value"] - values[0]
    step1 = prefix.rows[1]
    early_state = unpack(step1)
    early_score = forecast_rise(history[1:2], float(np.linalg.norm(v[0])), flight_factors(), curvature)
    alert = max(history[:2]) <= LIMIT and early_score > LIMIT
    c.update(
        context_id=name, incumbent_energy=ENERGY, initial_horizontal_velocities=v.tolist(),
        prefix_run_id=prefix_id, start=pack(initial), before_step=pack(prefix.final_state),
        post_reflection=pack(outgoing), previous_step=PRECONTACT_STEP,
        boundary_step=PRECONTACT_STEP+1, contact=contact, contact_time=PRECONTACT_STEP*H + contact["offset"],
        remaining_duration=remaining, initial_loss=values[0], history_rises=history, contact_rise=contact_rise,
        already_deviated_at_reflection=max(*history, contact_rise) > LIMIT,
        early_state=pack(early_state), early_score=early_score, early_warning=alert,
        already_deviated_at_step1=max(history[:2]) > LIMIT, early_prefix_calls=step1["total_oracle_calls"],
        shared_prefix_calls=prefix_calls, shared_contact_calls=contact_cost, charged_at_reflection=sum(oracle.counts().values()),
    )
    _json(root / "checkpoints" / (name + ".json"), c)
    branches = []
    for mode in MODES:
        branch = transfer_at_checkpoint(outgoing, 0) if mode == "transfer" else outgoing
        branch_oracle = PopulationOracle(objective, total_budget=CAP)
        load_counters(branch_oracle, oracle.counts())
        row = continue_branch(root, name, mode, branch, branch_oracle, cfg, remaining, end_step=END_STEP)
        branches.append({**row, "actual_suffix_calls": row["actual_fraction_calls"] + row["actual_tail_calls"]})
    if alert:
        branch = transfer_at_checkpoint(early_state, 0)
        early_oracle = PopulationOracle(objective, total_budget=CAP)
        load_counters(early_oracle, step1)
        method = IndependentEnergyPopulation(cfg)
        method.committed_steps = 1
        early = run_population(method, early_oracle, branch, END_STEP-1)
        early_id = name + "__early"
        save_segment(root, early_id, early)
        branches.append({
            "run_id": early_id, "context_id": name, "policy": "early",
            "actual_suffix_calls": early.counts["total_oracle_calls"] - c["early_prefix_calls"], **stop_record(early),
        })
    direct = []
    if preflight:
        direct_oracle = PopulationOracle(objective, total_budget=CAP)
        direct_oracle.values = 4
        full = run_population(IndependentEnergyPopulation(cfg), direct_oracle, initial, END_STEP)
        key = name + "__uninterrupted"
        save_segment(root, key, full)
        direct.append({"run_id": key, "context_id": name, **stop_record(full)})
    return c, p_record, branches, direct


def save(root, contexts, prefixes, runs, direct):
    for name, rows in (("contexts", contexts), ("prefixes", prefixes), ("runs", runs), ("uninterrupted", direct)):
        _json(root / (name + ".json"), rows)
    if runs:
        _csv(root / "runs.csv", runs)
    counts = {
        "contexts": len(contexts), "prefix_executions": len(prefixes), "exact_contacts": len(contexts),
        "continuations": len(runs), "uninterrupted_preflight_executions": len(direct),
        "actual_prefix_queries": sum(r["counts"]["total_oracle_calls"] for r in prefixes),
        "actual_contact_queries": sum(c["shared_contact_calls"] for c in contexts),
        "actual_forecast_queries": 0, "actual_continuation_queries": sum(r["actual_suffix_calls"] for r in runs),
        "actual_uninterrupted_queries_excluding_shared_initial_values": sum(r["counts"]["total_oracle_calls"]-4 for r in direct),
        "setup_reference_queries": sum(c["setup_reference_values"] for c in contexts),
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
    }
    counts["actual_solver_queries"] = sum(v for k, v in counts.items() if k.startswith("actual_"))
    counts["actual_queries_including_setup"] = counts["actual_solver_queries"] + counts["setup_reference_queries"]
    _json(root / "execution_counts.json", counts)
    return counts


def run(args):
    root, stage = args.output, args.stage
    dependencies = {str(p): sha(p) for p in (
        WARNING_SOURCE, PARENT/"config.json", PARENT/"paired_outcomes.json", PARENT/"raw_artifact_manifest.json")}
    if args.lock:
        previous = args.lock / "audit.json"
        assert read(previous)["passed"]
        dependencies[str(previous)] = sha(previous)
    root.mkdir(parents=True)
    for name in ("traces", "events", "uncommitted", "checkpoints", "fractions"):
        (root / name).mkdir()
    source = snapshot(root, SPEC)
    if args.lock:
        assert source == read(args.lock / "config.json")["source_sha256"]
    seeds = SEEDS[stage]
    curvature = read(WARNING_SOURCE)["prior_curvature"]
    _json(root / "config.json", {
        "phase": "E34", "stage": stage, "seeds": seeds, "initial_pivot_kinetic": ENERGY,
        "h": H, "gravity": .5, "stiffness": 8, "query_cap": CAP, "primary_step": PRIMARY_STEP,
        "end_step": END_STEP, "threshold": LIMIT, "early_prior_curvature": curvature, "binary_interactions": False,
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "launch_command": ["python3", *sys.argv],
        "cwd": str(Path.cwd()), "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "workers": args.workers, "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    contexts, prefixes, runs, direct = [], [], [], []
    save(root, contexts, prefixes, runs, direct)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate, root, s, curvature, stage == "preflight") for s in seeds]
        try:
            for i, job in enumerate(futures, 1):
                c, prefix, branches, ctrl = job.result()
                contexts.append(c)
                prefixes.append(prefix)
                runs.extend(branches)
                direct.extend(ctrl)
                if i % 8 == 0 or i == len(futures):
                    counts = save(root, contexts, prefixes, runs, direct)
                    print(f"{stage}: {i}/{len(futures)} matched initial states", flush=True)
        except BaseException:
            save(root, contexts, prefixes, runs, direct)
            _json(root / "task_failure.json", {"traceback": traceback.format_exc(),
                  "unreturned_cost": "Unknown if the worker did not return its ledger."})
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
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args)
