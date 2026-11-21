"""E46 controlled positive and negative cases for the post-contact forecast."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys

import numpy as np

from collision_optim.cooling import drag_rate
from collision_optim.core import State
from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.event_replay import pack
from collision_optim.population import PopulationOracle, PopulationState
from collision_optim.reflection_window import forecast_flight
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_contact_warning_replication import continue_branch, snapshot
from run_later_reflection_warning import CFG, H, LIMIT, read, sha
from run_long_horizon_exploration import ADDED_BUDGET, STOP_STEP

BASE = Path("output/collision_post_contact_positive_stress")
PARENT = Path("output/collision_third_reflection_contact_gate")
SPEC = Path("docs/superpowers/specs/2026-11-13-post-contact-positive-stress-design.md")
START, BOUNDARY_STEP = 1.54, 31
FRACTION = .8
CASES = ((1.605,.60),(1.605,.75),(1.605,.7999),(1.605,.8001),(1.605,.90),(1.605,1.02),
         (1.645,.75),(1.645,.85))


def launch_velocity(contact_time, rise):
    scale = 1.
    distance = gravity_distance = gravity_speed = 0.
    for step in range(BOUNDARY_STEP,int(np.ceil(contact_time/H))+1):
        start = START if step==BOUNDARY_STEP else (step-1)*H
        dt = min(step*H,contact_time)-start
        if step!=BOUNDARY_STEP:
            beta = np.exp(-H*drag_rate("ramp",start,H))
            scale *= beta
            gravity_speed *= beta
        distance += dt*scale
        gravity_distance += dt*gravity_speed+.5*CFG.physics.gravity*dt**2
        gravity_speed += CFG.physics.gravity*dt
    return np.sqrt(rise/4)/distance,(rise+gravity_distance)/distance


def build(seed,case,*,contact_case=None):
    objective = RotatedEnergyWells(8,seed)
    roots = np.sort(np.roots([1.,0.,-1.,.05]).real)
    t,fraction = CASES[case] if contact_case is None else contact_case
    vx,vz = launch_velocity(t,fraction*LIMIT)
    points = [objective.from_coordinates(p) for p in ((roots[0],0.),(roots[-1],-.025),(roots[-1],0.),(roots[-1],.025))]
    relative = np.array([[-.025,-.035],[.050,0.],[-.025,.035]])
    angle = np.random.default_rng(seed).uniform(0,2*np.pi)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    spectators = relative@rotation.T-[0.,vx/3]
    velocities = [objective.from_coordinates(u) for u in ([0.,vx],*spectators)]
    setup = PopulationOracle(objective,total_budget=4)
    loss = [setup.value(p) for p in points]
    state = PopulationState(tuple(State(p,v,y+height,w) for p,v,y,height,w in
        zip(points,velocities,loss,(0.,.04,.045,.05),(vz,0.,0.,0.))))
    assert np.linalg.norm(state.momentum[:2])<1e-12
    name = f"stress__s{seed}__case{case+1}"
    return objective,state,{"context_id":name,"seed":seed,"case":case+1,"contact_time_target":t,
        "contact_ratio_target":fraction,"initial_loss":loss[0],"rotation":objective.rotation.tolist(),
        "start":pack(state),"initial_losses":loss,"setup_counts":setup.counts()}


def predict(objective,state,initial,gated):
    probe = PopulationOracle(objective,total_budget=3)
    pred = forecast_flight(probe,state.particles[0],CFG,START,BOUNDARY_STEP,STOP_STEP,saved_limit=initial+LIMIT,
                           minimum_contact_loss=initial+FRACTION*LIMIT if gated else None)
    after = pred.get("after_contact_saved")
    saved = pred["observations"]+([after] if after else [])
    return {**pred,"counts":probe.counts(),"warning":any(p["loss"]-initial>LIMIT for p in saved)}


def simulate(root,seed,case):
    objective,state,context = build(seed,case)
    initial = context["initial_loss"]
    full,gated = (predict(objective,state,initial,is_gated) for is_gated in (False,True))
    context.update(full=full,gated=gated)
    branches = [("keep",state)]
    if full["warning"] or gated["warning"]:
        moved = capped_transfer_at_checkpoint(state,0)
        branches.append(("capped",moved))
    runs = []
    for mode,start in branches:
        oracle = PopulationOracle(objective,total_budget=ADDED_BUDGET)
        result = continue_branch(root,context["context_id"],mode,start,oracle,CFG,BOUNDARY_STEP*H-START,
                                 end_step=STOP_STEP,boundary_step=BOUNDARY_STEP)
        runs.append(result)
    return context,runs


def save(root,contexts,runs):
    _json(root/"contexts.json",contexts)
    _json(root/"runs.json",runs)
    if runs:
        _csv(root/"runs.csv",runs)
    counts = {"contexts":len(contexts),"continuations":len(runs),
              "setup_queries":sum(sum(c["setup_counts"].values()) for c in contexts),
              "full_forecast_queries":sum(sum(c["full"]["counts"].values()) for c in contexts),
              "gated_forecast_queries":sum(sum(c["gated"]["counts"].values()) for c in contexts),
              "continuation_queries":sum(r["counts"]["total_oracle_calls"] for r in runs)}
    counts["actual_new_queries"] = sum(v for k,v in counts.items() if k.endswith("_queries"))
    _json(root/"execution_counts.json",counts)
    return counts


def run(args):
    root = args.output
    paths = [PARENT/"raw_artifact_manifest.json",PARENT/"third_reflection"/"contact_gate_results.json"]
    if args.lock:
        path = args.lock/"audit.json"
        assert read(path)["passed"]
        paths.append(path)
    jobs = [(3110,i) for i in (2,3,6)] if args.stage=="preflight" else [(s,i) for s in range(3120,3126) for i in range(8)]
    root.mkdir(parents=True)
    for d in ("traces","events","fractions","uncommitted"):
        (root/d).mkdir()
    source = snapshot(root,SPEC)
    if args.lock:
        assert source==read(args.lock/"config.json")["source_sha256"]
    _json(root/"config.json",{"phase":"E46","stage":args.stage,"jobs":[(s,i+1) for s,i in jobs],
        "contact_cases":CASES,"start":START,"start_boundary_step":BOUNDARY_STEP,"end_step":STOP_STEP,"h":H,
        "gravity":CFG.physics.gravity,"threshold":LIMIT,"contact_fraction":FRACTION,"binary_interactions":False,
        "max_new_queries":len(jobs)*(4+6+2*ADDED_BUDGET),"source_sha256":source,
        "source_archive_sha256":sha(root/"source.zip"),"dependency_data_sha256":{str(p):sha(p) for p in paths},
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"platform":platform.platform(),
        "python":sys.version,"numpy":np.__version__,"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    contexts,runs = [],[]
    for seed,case in jobs:
        context,branches = simulate(root,seed,case)
        contexts.append(context)
        runs.extend(branches)
        counts = save(root,contexts,runs)
    _json(root/"completion_observation.json",{**counts,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(counts)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","formal"),required=True)
    parser.add_argument("--lock",type=Path)
    args = parser.parse_args()
    if (args.stage=="formal")!=bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args)
