"""E32: fresh starts, all eligible first reflections, and matched continuations."""

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

from collision_optim.contact_time_transfer import first_reflection, finish_fraction, forecast_to_boundary
from collision_optim.curvature_energy import IndependentEnergyPopulation, initial_state
from collision_optim.energy_generalization import controlled_velocities
from collision_optim.event_replay import pack
from collision_optim.gravity import GravityConfig
from collision_optim.hybrid import ContactResolutionError
from collision_optim.initial_budget import allocate_initial
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.prebounce_warning import flight_factors, forecast_rise
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import MODES, read, sha
from run_overlap_transfer import build_context
from run_postbounce_warning_transfer import load_counters
from run_prebounce_warning import ENERGIES
from run_shared_prefix_transfer import save_segment, stop_record
from run_warning_triggered_transfer import CAP, END_STEP, H, LIMIT

SPEC = Path("docs/superpowers/specs/2026-11-11-contact-warning-replication-design.md")
PARENT = Path("output/collision_contact_time_transfer/config.json")
SEEDS = {"preflight": [2310], "formal": list(range(2320, 2360))}
PRECONTACT_STEP = 6
BOUNDARY_STEP = PRECONTACT_STEP + 1


def snapshot(root, spec=SPEC, *, extra_paths=()):
    paths = {*Path("src").rglob("*.py"), *Path("tests").rglob("*.py"), *Path(".").glob("*.py"), *Path("scripts").glob("*.py"),
             spec, Path("AGENTS.md"), Path("pyproject.toml"), *extra_paths}
    manifest = {}
    with zipfile.ZipFile(root / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            archive.writestr(str(path), data)
            manifest[str(path)] = hashlib.sha256(data).hexdigest()
    _json(root / "source_manifest.json", manifest)
    return manifest


def continue_branch(root, name, mode, state, oracle, cfg, remaining, *, end_step=END_STEP,
                    boundary_step=BOUNDARY_STEP):
    key = name + "__" + mode
    fork_calls = sum(oracle.counts().values())
    try:
        part = finish_fraction(oracle, state, remaining, cfg)
    except Exception:
        _json(root / "fractions" / (key + ".json"), {
            "failed": True, "counts": oracle.counts(), "traceback": traceback.format_exc(),
        })
        raise
    fraction_end = sum(oracle.counts().values())
    _json(root / "fractions" / (key + ".json"), {
        "start": pack(state), "end": pack(part.state), "values": part.values,
        "duration": remaining, "damping_beta": 1., "diagnostics": part.diagnostics, "events": part.events,
        "total_oracle_calls_at_end": fraction_end,
    })
    method = IndependentEnergyPopulation(cfg)
    method.committed_steps = boundary_step
    tail = run_population(method, oracle, part.state, end_step - boundary_step)
    save_segment(root, key, tail)
    return {
        "run_id": key, "context_id": name, "policy": mode,
        "actual_fraction_calls": fraction_end - fork_calls,
        "actual_tail_calls": tail.counts["total_oracle_calls"] - fraction_end, **stop_record(tail),
    }


def simulate(root, seed, energy, preflight, forecast_extension=None):
    objective, points, values, _, c = build_context("good_present", seed, 300, 100)
    v = controlled_velocities(seed, energy)
    heights = allocate_initial(values, (0.,)*4, "fixed_ref_keep", v)
    start = initial_state(points, values, heights, v)
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    oracle = PopulationOracle(objective, total_budget=CAP)
    oracle.values = 4
    prefix = run_population(IndependentEnergyPopulation(cfg), oracle, start, PRECONTACT_STEP)
    name = f"good_present__s{seed}__k{energy:g}"
    prefix_id = name + "__prefix"
    save_segment(root, prefix_id, prefix)
    prefix_row = {"run_id": prefix_id, "context_id": name, **stop_record(prefix)}
    c.update(context_id=name, incumbent_energy=energy, initial_horizontal_velocities=v.tolist(),
             prefix_run_id=prefix_id, start=pack(start), selected=0, eligible=False,
             actual_event_reconstruction_calls=0, actual_forecast_calls=0)
    branches, directs = [], []
    if prefix_row["committed_steps"] != PRECONTACT_STEP:
        c["unavailable_reason"] = prefix.stop_reason
    elif any(e["i"] == 0 for e in prefix.events):
        c["unavailable_reason"] = "first_reflection_before_step_7"
    else:
        prefix_calls = sum(oracle.counts().values())
        try:
            outgoing, contact = first_reflection(oracle, prefix.final_state, cfg, PRECONTACT_STEP)
        except ContactResolutionError as exc:
            c["unavailable_reason"] = str(exc)
            c["actual_event_reconstruction_calls"] = sum(oracle.counts().values()) - prefix_calls
        else:
            event_calls = sum(oracle.counts().values()) - prefix_calls
            c["actual_event_reconstruction_calls"] = event_calls
            if contact["index"] != 0:
                c["unavailable_reason"] = "first_event_in_step_7_is_another_particle"
            else:
                assert event_calls == 10
                initial_loss = values[0]
                contact_rise = contact["contact_value"] - initial_loss
                history = [row["values"][0] - initial_loss for row in prefix.rows]
                already = max(*history, contact_rise) > LIMIT
                remaining = H - contact["offset"]
                prediction = None
                if not already:
                    if forecast_extension is None:
                        prediction = forecast_to_boundary(oracle, outgoing.particles[0], remaining, cfg)
                    else:
                        prediction, c["extended_forecast"] = forecast_extension(
                            oracle, outgoing.particles[0], remaining, cfg, initial_loss)
                charges = oracle.counts()
                total = sum(charges.values())
                risk = forecast_rise(history[1:3], float(np.linalg.norm(v[0])), flight_factors())
                early_status = ("already_deviated" if max(history[:3]) > LIMIT else
                                ("alert" if risk > LIMIT else "no_alert"))
                c.update(
                    eligible=True, before_step=pack(prefix.final_state), post_reflection=pack(outgoing),
                    contact=contact, previous_step=PRECONTACT_STEP, boundary_step=BOUNDARY_STEP,
                    remaining_duration=remaining, contact_time=PRECONTACT_STEP*H + contact["offset"],
                    initial_loss=initial_loss, contact_rise=contact_rise, history_rises=history,
                    already_deviated=already, earlier_saved_deviation=max(history) > LIMIT,
                    above_limit_at_contact=contact_rise > LIMIT, earlier_E28_status=early_status,
                    predicted_boundary_rise=prediction - initial_loss if prediction is not None else None,
                    warning=prediction is not None and prediction - initial_loss > LIMIT,
                    actual_forecast_calls=total - prefix_calls - event_calls,
                    historical_prefix_calls=prefix_calls, charged_at_branch=total,
                )
                _json(root / "checkpoints" / (name + ".json"), c)
                for mode in MODES:
                    state = transfer_at_checkpoint(outgoing, 0) if mode == "transfer" else outgoing
                    branch_oracle = PopulationOracle(objective, total_budget=CAP)
                    load_counters(branch_oracle, charges)
                    branches.append(continue_branch(root, name, mode, state, branch_oracle, cfg, remaining))
    if preflight:
        direct_oracle = PopulationOracle(objective, total_budget=CAP)
        direct_oracle.values = 4
        direct = run_population(IndependentEnergyPopulation(cfg), direct_oracle, start, END_STEP)
        direct_id = name + "__uninterrupted"
        save_segment(root, direct_id, direct)
        directs.append({"run_id": direct_id, "context_id": name, **stop_record(direct)})
    return c, prefix_row, branches, directs


def save(root, contexts, prefixes, runs, directs):
    for name, rows in (("contexts", contexts), ("prefixes", prefixes), ("runs", runs), ("uninterrupted", directs)):
        _json(root / (name + ".json"), rows)
    if runs:
        _csv(root / "runs.csv", runs)
    counts = {
        "contexts": len(contexts), "prefix_executions": len(prefixes),
        "exact_contacts": sum(c["eligible"] for c in contexts), "continuations": len(runs),
        "uninterrupted_preflight_executions": len(directs),
        "actual_prefix_queries": sum(r["counts"]["total_oracle_calls"] for r in prefixes),
        "actual_contact_queries": sum(c["actual_event_reconstruction_calls"] for c in contexts),
        "actual_forecast_queries": sum(c["actual_forecast_calls"] for c in contexts),
        "actual_fraction_queries": sum(r["actual_fraction_calls"] for r in runs),
        "actual_tail_queries": sum(r["actual_tail_calls"] for r in runs),
        "actual_uninterrupted_queries_excluding_shared_initial_values":
            sum(r["counts"]["total_oracle_calls"] - 4 for r in directs),
        "setup_reference_queries": sum(c["setup_reference_values"] for c in contexts),
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
    }
    counts["actual_solver_queries"] = sum(v for k, v in counts.items() if k.startswith("actual_"))
    counts["actual_queries_including_setup"] = counts["actual_solver_queries"] + counts["setup_reference_queries"]
    _json(root / "execution_counts.json", counts)
    return counts


def run(args, *, phase="E32", spec=SPEC, parent=PARENT, seed_sets=SEEDS, forecast_extension=None):
    root, stage = args.output, args.stage
    dependencies = {str(parent): sha(parent)}
    if args.lock:
        audit = args.lock / "audit.json"
        assert read(audit)["passed"]
        dependencies[str(audit)] = sha(audit)
    root.mkdir(parents=True)
    for folder in ("traces", "events", "uncommitted", "checkpoints", "fractions"):
        (root / folder).mkdir()
    source = snapshot(root, spec)
    if args.lock:
        assert source == read(args.lock / "config.json")["source_sha256"]
    seeds = seed_sets[stage]
    _json(root / "config.json", {
        "phase": phase, "stage": stage, "seeds": seeds, "energies": list(ENERGIES),
        "query_cap": CAP, "precontact_step": PRECONTACT_STEP, "boundary_step": BOUNDARY_STEP,
        "end_step": END_STEP, "h": H, "threshold": LIMIT, "binary_interactions": False,
        "source_sha256": source, "source_archive_sha256": sha(root / "source.zip"),
        "dependency_data_sha256": dependencies, "launch_command": ["python3", *sys.argv],
        "cwd": str(Path.cwd()), "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__, "workers": args.workers, "launch_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    })
    contexts, prefixes, runs, directs = [], [], [], []
    save(root, contexts, prefixes, runs, directs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate, root, seed, e, stage == "preflight", forecast_extension)
                   for seed in seeds for e in ENERGIES]
        try:
            for i, future in enumerate(futures, 1):
                c, prefix, branches, direct = future.result()
                contexts.append(c)
                prefixes.append(prefix)
                runs.extend(branches)
                directs.extend(direct)
                if i % 40 == 0 or i == len(futures):
                    counts = save(root, contexts, prefixes, runs, directs)
                    print(f"{stage}: {i}/{len(futures)} fresh states, {len(runs)} continuations", flush=True)
        except BaseException:
            save(root, contexts, prefixes, runs, directs)
            _json(root / "task_failure.json", {"traceback": traceback.format_exc(),
                  "unreturned_cost": "Unknown if a worker did not return its ledger."})
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
