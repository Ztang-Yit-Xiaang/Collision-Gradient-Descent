"""E38: examine all E37 selected paths and branch on later terrain reflections."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from audit_contact_window_guard import guarded
from audit_second_step_protection import select
from audit_shared_prefix_transfer import load_segment
from audit_two_stage_protection import lookup
from collision_optim.core import State
from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.event_replay import pack, unpack
from collision_optim.gravity import GravityConfig, GravityContact
from collision_optim.population import PopulationConfig, PopulationOracle, PopulationState
from collision_optim.reflection_window import forecast_flight
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import read, sha
from run_contact_warning_replication import BOUNDARY_STEP, continue_branch, snapshot
from run_two_stage_protection import CAP, END_STEP, H, LIMIT, PRIMARY_STEP

SPEC = Path("docs/superpowers/specs/2026-11-12-later-reflection-warning-design.md")
PARENT = {stage: Path("output/collision_contact_window_guard" + suffix) for stage, suffix in (
    ("preflight", "_preflight"), ("replay", ""))}
CFG = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)


def reconstruct(previous, boundary, step_events, target_index):
    """Rebuild the population at a saved event and audit the end of its step."""
    beta = float(boundary["drag_beta"])
    state = unpack(previous)
    particles = [State(p.x, beta*p.v, p.z, beta*p.vz) for p in state.particles]
    gravity = GravityContact(CFG.physics)
    start = float(previous["time"])
    clock = start
    worst = 0.
    outgoing = None
    for idx, event in step_events:
        t = event["time"]
        assert t >= clock - 1e-12
        particles = [gravity._flight(p, max(0., t-clock)) for p in particles]
        i = event["i"]
        p = particles[i]
        q, before, after = (np.array(event[k]) for k in ("point_i", "before_i", "after_i"))
        worst = max(worst, float(np.max(abs(np.r_[p.x, p.z]-q))),
                    float(np.max(abs(p.velocity-before))))
        particles[i] = State(q[:-1], after[:-1], q[-1], after[-1])
        if idx == target_index:
            outgoing = PopulationState(tuple(particles))
        clock = t
    end = float(boundary["time"])
    rebuilt = pack(PopulationState(tuple(gravity._flight(p, max(0., end-clock)) for p in particles)))
    worst = max(worst, *(float(np.max(abs(np.array(rebuilt[k])-np.array(boundary[k])))) for k in rebuilt))
    assert outgoing is not None and worst < 1e-10 and end-start > 0.
    return outgoing, worst


def decode_rows(root, key):
    rows, q, v, f = load_segment(root, key)
    return [{**r, "positions": x.tolist(), "velocities": u.tolist(), "values": y.tolist()}
            for r, x, u, y in zip(rows, q, v, f)]


def selected_history(parent, row, c, hashes):
    policy_row = guarded(row)
    chosen = select(policy_row, True)
    source = lookup(policy_row, chosen)
    key = source["run_id"]
    paths = [parent / "traces" / (key+".csv.gz"), parent / "events" / (key+".json")]
    for path in paths:
        assert sha(path) == hashes[str(path)], path
    trace = decode_rows(parent, key)
    family = c["families"][chosen["family"]]
    events = read(paths[1])
    assert len(trace) == END_STEP-BOUNDARY_STEP+1
    assert all(e["kind"] == "wall" for e in events)
    dependencies = {str(p): hashes[str(p)] for p in paths}
    return source, family, trace, events, dependencies


def late_checkpoint(trace, events, index, ordinal, context, family):
    event = events[index]
    local_step = event["step"]
    step = BOUNDARY_STEP+local_step
    outgoing, error = reconstruct(trace[local_step-1], trace[local_step],
                                  [(i,e) for i,e in enumerate(events) if e["step"]==local_step], index)
    time = BOUNDARY_STEP*H+event["time"]
    remaining = step*H-time
    assert remaining > CFG.physics.time_tolerance
    initial = context["initial_loss"]
    rise = event["point_i"][-1]/CFG.physics.height_scale-initial
    saved_history = [*family["history_rises"], family["contact_rise"],
                     *[r["values"][0]-initial for r in trace[:local_step]]]
    contact_history = [e["point_i"][-1]/CFG.physics.height_scale-initial
                       for e in events[:index] if e["i"]==0]
    return outgoing, {
        "checkpoint_id": f"{context['context_id']}__bounce{ordinal}", "ordinal": ordinal,
        "context_id": context["context_id"], "boundary_step": step, "time": time,
        "remaining_duration": remaining, "initial_loss": initial, "contact_rise": rise,
        "source_event_index": index, "event": event, "post_reflection": pack(outgoing),
        "reconstruction_error": error, "prior_saved_peak_rise": max(saved_history),
        "prior_contact_peak_rise": max(contact_history, default=None),
        "already_deviated": max([rise, *saved_history, *contact_history])>LIMIT,
        "above_at_contact": rise>LIMIT,
    }


def simulate(root, parent, row, c, hashes, force_pair):
    source, family, trace, events, dependencies = selected_history(parent,row,c,hashes)
    target = [i for i,e in enumerate(events) if e["i"]==0]
    assert len(target)==2
    objective = RotatedEnergyWells(8,c["seed"])
    forecasts, branches = [], []
    for ordinal,index in enumerate(target,2):
        state, record = late_checkpoint(trace,events,index,ordinal,c,family)
        oracle = PopulationOracle(objective,total_budget=1)
        pred = None
        if not record["already_deviated"]:
            pred = forecast_flight(oracle,state.particles[0],CFG,record["time"],record["boundary_step"],END_STEP)
        points = [*pred["observations"], *([pred["next_contact"]] if pred["next_contact"] else [])] if pred else []
        first = next((p for p in points if p["loss"]-c["initial_loss"]>LIMIT),None)
        record.update(prediction=pred, first_predicted_crossing=first, warning=first is not None,
                      forecast_counts=oracle.counts(), engineering_pair=force_pair)
        forecasts.append(record)
        key = record["checkpoint_id"]
        _json(root/"checkpoints"/(key+".json"),record)
        if record["warning"] or record["already_deviated"] or force_pair:
            for mode in ("keep","transfer"):
                start = state if mode=="keep" else transfer_at_checkpoint(state,0)
                branch = PopulationOracle(objective,total_budget=CAP)
                r = continue_branch(root,key,mode,start,branch,CFG,record["remaining_duration"],
                                    end_step=END_STEP,boundary_step=record["boundary_step"])
                branches.append({**r,"checkpoint_id":key,"context_id":c["context_id"],"ordinal":ordinal})
    return {"context_id":c["context_id"],"seed":c["seed"],"incumbent_energy":c["incumbent_energy"],
            "source_branch":source,"source_first_contact_rise":family["contact_rise"],
            "source_dependencies":dependencies,"checkpoints":forecasts}, branches


def save(root, cases, runs):
    _json(root/"contexts.json",cases)
    _json(root/"runs.json",runs)
    if runs:
        _csv(root/"runs.csv",runs)
    records = [x for c in cases for x in c["checkpoints"]]
    counts = {"contexts":len(cases),"reused_parent_trajectories":len(cases),"new_initial_states":0,
              "saved_event_reconstructions":len(records),"continuations":len(runs),
              "actual_forecast_queries":sum(sum(r["forecast_counts"].values()) for r in records),
              "actual_fraction_queries":sum(r["actual_fraction_calls"] for r in runs),
              "actual_tail_queries":sum(r["actual_tail_calls"] for r in runs)}
    counts["actual_new_solver_queries"] = sum(v for k,v in counts.items() if k.startswith("actual_"))
    _json(root/"execution_counts.json",counts)
    return counts


def run(args):
    root, parent = args.output, PARENT[args.stage]
    paths = [parent/f for f in ("config.json","contexts.json","audit.json","paired_outcomes.json","raw_artifact_manifest.json")]
    hashes = {r["path"]:r["sha256"] for r in read(parent/"raw_artifact_manifest.json")["files"]}
    assert read(parent/"audit.json")["passed"]
    assert all(sha(p)==hashes[str(p)] for p in paths[:-1])
    if args.lock:
        assert read(args.lock/"audit.json")["passed"]
        paths.append(args.lock/"audit.json")
    root.mkdir(parents=True)
    for d in ("traces","events","uncommitted","checkpoints","fractions"):
        (root/d).mkdir()
    source = snapshot(root,SPEC)
    if args.lock:
        assert source==read(args.lock/"config.json")["source_sha256"]
    contexts = {c["context_id"]:c for c in read(parent/"contexts.json")}
    rows = read(parent/"paired_outcomes.json")
    assert len(rows)==(6 if args.stage=="preflight" else 480)
    _json(root/"config.json",{
        "phase":"E38","stage":args.stage,"parent":str(parent),"new_independent_seed_groups":0,
        "reused_seeds":sorted({r["seed"] for r in rows}),"forecast_ordinals":[2,3],"end_step":END_STEP,
        "primary_step":PRIMARY_STEP,"threshold":LIMIT,"h":H,"gravity":CFG.physics.gravity,
        "continuation_query_cap":CAP,"new_query_cap":len(rows)*(2+4*CAP),"binary_interactions":False,
        "source_sha256":source,"source_archive_sha256":sha(root/"source.zip"),
        "dependency_data_sha256":{str(p):sha(p) for p in paths},"launch_command":["python3",*sys.argv],
        "cwd":str(Path.cwd()),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
        "workers":args.workers,"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in (
            "PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
    })
    cases, runs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for i,row in enumerate(rows):
            c = contexts[row["context_id"]]
            upgraded = guarded(row)
            key = lookup(upgraded,select(upgraded,True))["run_id"]
            needed = {str(p):hashes[str(p)] for p in (
                parent/"traces"/(key+".csv.gz"),parent/"events"/(key+".json"))}
            jobs.append(pool.submit(simulate,root,parent,row,c,needed,args.stage=="preflight" and i==0))
        try:
            for i,job in enumerate(jobs,1):
                case, pair = job.result()
                cases.append(case)
                runs.extend(pair)
                if i%80==0 or i==len(jobs):
                    counts = save(root,cases,runs)
                    print(f"{args.stage}: {i}/{len(rows)} parent trajectories, {len(runs)} continuations",flush=True)
        except BaseException:
            save(root,cases,runs)
            _json(root/"task_failure.json",{"traceback":traceback.format_exc(),
                  "unreturned_cost":"Unknown for workers that did not return their ledger."})
            raise
    _json(root/"completion_observation.json",{**counts,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(counts,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=PARENT,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=="replay")!=bool(args.lock):
        parser.error("Positive worker count required. Replay requires the audited preflight lock.")
    run(args)
