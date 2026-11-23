"""Run the additional E40 branch from the E37 control's second reflection."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from _source_paths import manifest_path
from audit_contact_window_guard import guarded, report
from audit_second_step_protection import select
from audit_two_stage_protection import audit, lookup
from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.reflection_window import forecast_flight
from collision_optim.population import PopulationOracle
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_contact_warning_replication import continue_branch
from run_later_reflection_warning import BOUNDARY_STEP, CAP, CFG, END_STEP, H, LIMIT, decode_rows, late_checkpoint, read, sha
from run_postbounce_warning_transfer import load_counters
from run_second_reflection_capped_policy import SEEDS


def selected_source(root, row, context):
    choice = select(guarded(row),True)
    branch = lookup(row,choice)
    key = branch["run_id"]
    files = [root/d/(key+ext) for d,ext in (("traces",".csv.gz"),("events",".json"))]
    trace = decode_rows(root,key)
    events = read(files[1])
    assert len(trace)==END_STEP-BOUNDARY_STEP+1 and all(e["kind"]=="wall" for e in events)
    return choice,branch,context["families"][choice["family"]],trace,events,files


def simulate(root, target, row, context):
    choice,source,family,trace,events,files = selected_source(root,row,context)
    indices = [i for i,e in enumerate(events) if e["i"]==0]
    entry = {"context_id":row["context_id"],"seed":row["seed"],"incumbent_energy":row["incumbent_energy"],
             "source":source["run_id"],"source_hashes":{str(f):sha(f) for f in files},
             "checkpoint":None,"forecast_queries":0,"warning":False,"transferred":False}
    if not indices:
        return entry,None
    state,checkpoint = late_checkpoint(trace,events,indices[0],2,context,family)
    entry["checkpoint"] = checkpoint
    if checkpoint["already_deviated"]:
        return entry,None
    objective = RotatedEnergyWells(8,row["seed"])
    probe = PopulationOracle(objective,total_budget=1)
    prediction = forecast_flight(probe,state.particles[0],CFG,checkpoint["time"],checkpoint["boundary_step"],END_STEP)
    points = prediction["observations"]+([prediction["next_contact"]] if prediction["next_contact"] else [])
    first = next((p for p in points if p["loss"]-context["initial_loss"]>LIMIT),None)
    entry.update(prediction=prediction,forecast_queries=sum(probe.counts().values()),warning=first is not None,
                 first_predicted_crossing=first)
    if first is None:
        return entry,None
    moved = capped_transfer_at_checkpoint(state,0)
    if moved is state:
        return entry,None
    # Charge the true prefix, including the observed second wall event. Each
    # prior event iteration and the current one paid 4 values, 4 paths and
    # two contact probes (one value and one gradient). Simultaneous groups
    # share the iteration's 4+4 calls and pay one probe pair per wall.
    event = checkpoint["event"]
    step = event["step"]
    oracle = PopulationOracle(objective,total_budget=CAP)
    load_counters(oracle,trace[step-1])
    before = [e for e in events[:checkpoint["source_event_index"]+1] if e["step"]==step]
    assert before[-1]==event and all(abs(e["time"]-event["time"])>CFG.physics.time_tolerance for e in events[checkpoint["source_event_index"]+1:] if e["step"]==step)
    groups = 1+sum(b["time"]-a["time"]>CFG.physics.time_tolerance for a,b in zip(before,before[1:]))
    oracle.values += 4*groups+len(before)
    oracle.gradients += len(before)
    oracle.paths += 4*groups
    # E37 physically shared prefixes omit this charge on an unchanged path.
    omitted_guard = int(context["contact_guard_queries"] and not context["contact_guard_alert"])
    oracle.paths += omitted_guard
    prefix = oracle.counts()
    oracle.paths += entry["forecast_queries"]
    assert sum(prefix.values()) < int(trace[step]["total_oracle_calls"])+omitted_guard
    r = continue_branch(target,checkpoint["checkpoint_id"],"capped",moved,oracle,CFG,
                        checkpoint["remaining_duration"],end_step=END_STEP,boundary_step=checkpoint["boundary_step"])
    entry.update(transferred=True,prefix_queries=prefix,additional_e37_guard_queries=omitted_guard)
    return entry,{**r,"checkpoint_id":checkpoint["checkpoint_id"],"context_id":row["context_id"],
                  "charged_prefix_queries":sum(prefix.values())+entry["forecast_queries"]}


def save(target, cases, runs):
    _json(target/"contexts.json",cases)
    _json(target/"runs.json",runs)
    if runs:
        _csv(target/"runs.csv",runs)
    costs = {"contexts":len(cases),"actual_forecast_queries":sum(r["forecast_queries"] for r in cases),
             "continuations":len(runs),"actual_fraction_queries":sum(r["actual_fraction_calls"] for r in runs),
             "actual_tail_queries":sum(r["actual_tail_calls"] for r in runs)}
    costs["actual_new_solver_queries"] = sum(v for k,v in costs.items() if k.startswith("actual_"))
    _json(target/"execution_counts.json",costs)
    return costs


def run(root,workers,*,phase="E40",seed_sets=SEEDS):
    cfg = read(root/"config.json")
    assert cfg["phase"]==phase
    if not (root/"audit.json").exists():
        audit(root,seed_sets=seed_sets,evaluator=report,second_stage=True)
    check = read(root/"audit.json")
    assert check["passed"] and not any(check[k] for k in (
        "solver_failures","incomplete_families","incomplete_continuations","incomplete_forecasts"))
    target = root/"second_reflection"
    target.mkdir()
    for folder in ("traces","events","fractions","uncommitted"):
        (target/folder).mkdir()
    source = Path(__file__).resolve().relative_to(Path.cwd())
    manifest = cfg["source_sha256"]
    assert sha(source)==manifest[manifest_path(source, manifest)]
    dependencies = [root/(n+".json") for n in ("config","contexts","paired_outcomes","audit","runs","execution_counts")]
    if cfg["stage"]=="formal":
        lock = Path(cfg["launch_command"][cfg["launch_command"].index("--lock")+1])/"second_reflection"/"audit.json"
        assert read(lock)["passed"]
        dependencies.append(lock)
    _json(target/"config.json",{
        "phase":phase,"stage":cfg["stage"],"workers":workers,"second_reflection_only":True,
        "dependency_data_sha256":{str(p):sha(p) for p in dependencies},"launch_command":["python3",*sys.argv],
        "cwd":str(Path.cwd()),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
        "launch_utc":datetime.now(timezone.utc).isoformat(),"thread_environment":{k:os.environ.get(k) for k in (
            "PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
    })
    contexts = {c["context_id"]:c for c in read(root/"contexts.json")}
    cases,runs = [],[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(simulate,root,target,r,contexts[r["context_id"]]) for r in read(root/"paired_outcomes.json")]
        try:
            for i,job in enumerate(jobs,1):
                case,branch = job.result()
                cases.append(case)
                if branch:
                    runs.append(branch)
                if i%80==0 or i==len(jobs):
                    costs = save(target,cases,runs)
                    print(f"second reflection: {i}/{len(jobs)} matched starts, {len(runs)} capped continuations",flush=True)
        except BaseException:
            save(target,cases,runs)
            _json(target/"task_failure.json",{"traceback":traceback.format_exc(),"unreturned_cost":"Unknown for non-returning workers"})
            raise
    _json(target/"completion_observation.json",{**costs,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(costs,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    run(args.output,args.workers)
