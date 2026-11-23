"""Evaluate a third-reflection warning on archived E41 policy paths."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.population import PopulationOracle
from collision_optim.reflection_window import forecast_flight
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_capped_reflection_transfer import checked_hashes
from run_contact_warning_replication import continue_branch, snapshot
from run_later_reflection_warning import CFG, END_STEP, H, LIMIT, decode_rows, reconstruct, read, sha
from run_long_horizon_exploration import BASE as E41, PARENT as E40, STOP_STEP, ADDED_BUDGET, sources

BASE = Path("output/collision_third_reflection_guard")
SPEC = Path("docs/superpowers/specs/2026-11-12-third-reflection-guard-design.md")
EPS = 1e-10


def points(rows, offset):
    return [(float((offset+int(r["step"]))*H),r["values"][0]) for r in rows]


def contacts(events, offset):
    return [(offset+e["time"],e["point_i"][-1]) for e in events if e["i"]==0]


class Archive:
    def __init__(self):
        self.hashes = {str(root):{r["path"]:r["sha256"] for r in read(root/"raw_artifact_manifest.json")["files"]}
                       for root in (E40,E41)}
        self.dependencies = {}

    def verify(self, path):
        key = str(path)
        if key not in self.dependencies:
            root = E41 if path.is_relative_to(E41) else E40
            self.dependencies[key] = sha(path)
            assert self.hashes[str(root)][key]==self.dependencies[key],key
        return path

    def read(self, path):
        return read(self.verify(path))

    def trace(self, folder,key):
        self.verify(folder/"traces"/(key+".csv.gz"))
        return decode_rows(folder,key),self.read(folder/"events"/(key+".json"))


def history(data, case, ctx, *, parent=E40, extension=E41):
    mode,folder,key = list(sources(case,parent))[-1]
    continuation,extra = data.trace(extension,case["context_id"]+"__"+mode)
    rows,events = data.trace(folder,key)
    family = ctx["families"][case["source"].rsplit("__",2)[-2]]
    initial = ctx["initial_loss"]
    past = [(i*H,initial+rise) for i,rise in enumerate(family["history_rises"])]
    hits = [(family["contact_time"],family["contact"]["contact_value"])]
    if case["transferred"]:
        cp = case["checkpoint"]
        step = cp["boundary_step"]
        control,prior_events = data.trace(parent,case["source"])
        past += points(control[:step-family["boundary_step"]],family["boundary_step"])
        hits += [p for p in contacts(prior_events,family["boundary_step"]*H) if p[0]<=cp["time"]+EPS]
        fraction = data.read(folder/"fractions"/(key+".json"))
        hits += contacts(fraction["events"],cp["time"])
    else:
        step = family["boundary_step"]
        fraction = data.read(folder/"fractions"/(key+".json"))
        hits += contacts(fraction["events"],family["contact_time"])
    assert rows[-1]["positions"]==continuation[0]["positions"]
    assert rows[-1]["velocities"]==continuation[0]["velocities"]
    assert rows[-1]["values"]==continuation[0]["values"]
    saved = past+points(rows,step)+points(continuation[1:],END_STEP)
    assert len(saved)==STOP_STEP+1 and all(abs(t-i*H)<EPS for i,(t,_) in enumerate(saved))
    hits += contacts(events,step*H)+contacts(extra,END_STEP*H)
    hits.sort()
    return saved,hits,rows,events,step


def checkpoint(case, ctx, saved, hits, rows, events, start):
    third = hits[2][0]
    index,e = next((i,e) for i,e in enumerate(events) if e["i"]==0 and abs(start*H+e["time"]-third)<EPS)
    local = e["step"]
    state,error = reconstruct(rows[local-1],rows[local],[(i,v) for i,v in enumerate(events) if v["step"]==local],index)
    step = start+local
    initial = ctx["initial_loss"]
    loss = hits[2][1]
    before = [f for t,f in saved+hits if t<=third+EPS]
    remaining = step*H-third
    assert remaining>CFG.physics.time_tolerance
    return state,{"checkpoint_id":case["context_id"]+"__bounce3","time":third,"boundary_step":step,
        "remaining_duration":remaining,"initial_loss":initial,"contact_rise":loss-initial,
        "already_deviated":max(before)-initial>LIMIT,"above_at_contact":loss-initial>LIMIT,
        "event":e,"reconstruction_error":error,"post_reflection":{"positions":[np.r_[p.x,p.z].tolist() for p in state.particles],
        "velocities":[p.velocity.tolist() for p in state.particles]}}


def simulate(root,case,ctx,record=None,control=False,*,data=None,parent=E40,extension=E41,saved_window=False,
             contact_fraction=None):
    data = Archive() if data is None else data
    saved,hits,rows,events,start = history(data,case,ctx,parent=parent,extension=extension)
    state,cp = checkpoint(case,ctx,saved,hits,rows,events,start)
    result = {"context_id":case["context_id"],"seed":case["seed"],"incumbent_energy":case["incumbent_energy"],
              "checkpoint":cp,"dependencies":data.dependencies,"prediction":None,"forecast_queries":0,
              "warning":False,"transferred":False}
    if record is not None:
        result["source_late_crossing"] = record["good_with_contacts"]["first_after_t2_crossing"]
    objective = RotatedEnergyWells(8,case["seed"])
    probe = PopulationOracle(objective,total_budget=3 if saved_window else 1)
    if saved_window:
        result.update(saved_warning=False,saved_forecast_queries=0)
    if contact_fraction is not None:
        assert saved_window
        result.update(gated_warning=False,gated_forecast_queries=0)
    if not cp["already_deviated"]:
        prediction = forecast_flight(probe,state.particles[0],CFG,cp["time"],cp["boundary_step"],STOP_STEP,
                                     saved_limit=cp["initial_loss"]+LIMIT if saved_window else None)
        obs = prediction["observations"]+([prediction["next_contact"]] if prediction["next_contact"] else [])
        warning = any(p["loss"]-cp["initial_loss"]>LIMIT for p in obs)
        result.update(prediction=prediction,forecast_queries=1,warning=warning)
        if saved_window:
            extra = prediction.get("after_contact_saved")
            saved_points = prediction["observations"]+([extra] if extra is not None else [])
            result.update(saved_warning=any(p["loss"]-cp["initial_loss"]>LIMIT for p in saved_points),
                          saved_forecast_queries=sum(probe.counts().values()))
            if contact_fraction is not None:
                hit = prediction["next_contact"]
                allowed = hit is not None and hit["loss"]-cp["initial_loss"]>=contact_fraction*LIMIT
                gated_points = prediction["observations"]+([extra] if allowed and extra is not None else [])
                result.update(gated_warning=any(p["loss"]-cp["initial_loss"]>LIMIT for p in gated_points),
                              gated_forecast_queries=1+2*int(allowed and "after_contact_saved" in prediction))
    runs = []
    trigger = result.get("saved_warning",False) or (contact_fraction is None and result["warning"])
    modes = (["capped"] if trigger else [])+(["keep"] if control else [])
    for mode in modes:
        moved = capped_transfer_at_checkpoint(state,0) if mode=="capped" else state
        if mode=="capped" and moved is state:
            continue
        oracle = PopulationOracle(objective,total_budget=ADDED_BUDGET)
        run = continue_branch(root,cp["checkpoint_id"],mode,moved,oracle,CFG,cp["remaining_duration"],
                              end_step=STOP_STEP,boundary_step=cp["boundary_step"])
        runs.append({**run,"context_id":case["context_id"],"checkpoint_id":cp["checkpoint_id"]})
        result["transferred"] |= mode=="capped"
    return result,runs


def save(root,records,runs):
    _json(root/"decisions.json",records)
    _json(root/"runs.json",runs)
    if runs:
        _csv(root/"runs.csv",runs)
    costs = Counter(contexts=len(records),continuations=len(runs),new_initial_states=0,
                    forecast_queries=sum(r.get("saved_forecast_queries",r["forecast_queries"]) for r in records),
                    fraction_queries=sum(r["actual_fraction_calls"] for r in runs),tail_queries=sum(r["actual_tail_calls"] for r in runs))
    costs["actual_new_queries"] = costs["forecast_queries"]+costs["fraction_queries"]+costs["tail_queries"]
    _json(root/"execution_counts.json",dict(costs))
    return dict(costs)


def run(args):
    root = args.output
    files = [E41/f for f in ("contexts.json","paired_outcomes.json","audit.json")]
    deps = checked_hashes(E41,files)
    deps.update(checked_hashes(E40,[E40/"contexts.json"]))
    for folder in (E40,E41):
        path = folder/"raw_artifact_manifest.json"
        deps[str(path)] = sha(path)
    assert read(files[-1])["passed"]
    if args.lock:
        path = args.lock/"audit.json"
        assert read(path)["passed"]
        deps[str(path)] = sha(path)
    cases = read(files[0])
    assert len(cases)==480
    if args.stage=="preflight":
        cases = [c for c in cases if c["seed"] in (2722,2742,2757) and c["incumbent_energy"]==8e-5]
        assert len(cases)==3
    parent = {r["context_id"]:r for r in read(E40/"contexts.json")}
    observed = {r["context_id"]:r["capped_policy"] for r in read(files[1])}
    root.mkdir(parents=True)
    for d in ("traces","events","fractions","uncommitted"):
        (root/d).mkdir()
    source = snapshot(root,SPEC)
    if args.lock:
        assert source==read(args.lock/"config.json")["source_sha256"]
    _json(root/"config.json",{"phase":"E42","stage":args.stage,"contexts":len(cases),"parent":str(E41),
        "threshold":LIMIT,"h":H,"gravity":CFG.physics.gravity,"end_step":STOP_STEP,"new_independent_seed_groups":0,
        "binary_interactions":False,"per_continuation_query_cap":ADDED_BUDGET,
        "new_query_cap":16387 if args.stage=="preflight" else 1966560,
        "source_sha256":source,"source_archive_sha256":sha(root/"source.zip"),"dependency_data_sha256":deps,
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"platform":platform.platform(),"python":sys.version,
        "numpy":np.__version__,"workers":args.workers,"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    _json(root/"contexts.json",cases)
    records,runs = [],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(simulate,root,c,parent[c["context_id"]],observed[c["context_id"]],
                            args.stage=="preflight" and c["seed"]==2722) for c in cases]
        try:
            for i,job in enumerate(jobs,1):
                entry,branches = job.result()
                records.append(entry)
                runs.extend(branches)
                if i%80==0 or i==len(jobs):
                    costs = save(root,records,runs)
                    print(f"{i}/{len(jobs)} checkpoints, {len(runs)} continuations",flush=True)
        except BaseException:
            save(root,records,runs)
            _json(root/"task_failure.json",{"traceback":traceback.format_exc(),"unreturned_cost":"Unknown for non-returning workers"})
            raise
    _json(root/"completion_observation.json",{**costs,"observed_utc":datetime.now(timezone.utc).isoformat()})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","replay"),required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=="replay")!=bool(args.lock):
        parser.error("Replay requires the audited preflight lock and positive worker count.")
    run(args)
