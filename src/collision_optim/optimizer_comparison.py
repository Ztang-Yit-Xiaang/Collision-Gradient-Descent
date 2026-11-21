"""Held-out first-order baselines and optimizer/physical intervention controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from .core import State
from .highdim_objectives import StructuredRotation,CoupledDoubleWell,CoupledRosenbrock
from .objectives import Quadratic
from .optimizer_controls import OptimizerConfig,ControlConfig,run_optimizer
from .population import PopulationOracle,PopulationState
from .predictive import PredictiveConfig
from .predictive_experiment import initial_case
from .predictive_runner import run_predictive
from .strategic_experiment import _json,_csv


CASES=("bad_well","good_well","saddle","quadratic10","rosenbrock10","wells20")
METHODS=("momentum","adam","momentum_random","momentum_predictive","adam_random","adam_predictive","gravity","gravity_predictive")
GRIDS={"momentum":(.001,.01,.05,.1),"adam":(.0005,.005,.03,.1),"gravity":(.1,.3,1.,3.)}


def make_case(case,seed,height_scale=1.):
    cfg=PredictiveConfig();physics=replace(cfg.strategy.population.physics,height_scale=height_scale)
    cfg=replace(cfg,strategy=replace(cfg.strategy,population=replace(cfg.strategy.population,physics=physics)))
    if case in {"bad_well","good_well","saddle"}:
        objective,minimum,boundary,unscaled=initial_case(case,seed,PredictiveConfig())
        p=unscaled.particles[0];speed=np.sqrt(height_scale)
        initial=PopulationState((State(p.x,speed*p.v,height_scale*p.z,speed*p.vz),))
        return objective,minimum,boundary,initial
    rng=np.random.default_rng(605239+seed)
    if case=="quadratic10":
        rotation=StructuredRotation(10)
        q=np.column_stack([rotation.forward(np.eye(10)[:,j]) for j in range(10)])
        objective=Quadratic(q.T @ np.diag(np.geomspace(1.,100.,10)) @ q);minimum=0.
        x=rng.normal(size=10);x/=np.linalg.norm(x)
    elif case=="rosenbrock10":
        objective=CoupledRosenbrock(10);minimum=0.
        r=np.tile([-1.2,1.],5)+rng.normal(scale=.02,size=10);x=objective.from_coordinates(r)
    elif case=="wells20":
        objective=CoupledDoubleWell(20);minimum=objective.minimum
        x=objective.from_coordinates(rng.normal(loc=.2,scale=.9,size=20))
    else:raise ValueError("Unknown benchmark case.")
    velocity=rng.normal(scale=.006/np.sqrt(x.size),size=x.size)
    initial=PopulationState((State(x,np.sqrt(height_scale)*velocity,height_scale*(objective.value(x)+.14),np.sqrt(height_scale)*-.03),))
    return objective,minimum,None,initial


def run_case(case,seed,method,setting,budget,*,protocol="budget"):
    physical=method.startswith("gravity")
    objective,minimum,boundary,initial=make_case(case,seed,setting if physical else 1.)
    x0=initial.particles[0].x;f0=objective.value(x0);gap=max(f0-minimum,1e-12);threshold=minimum+.1*gap
    oracle=PopulationOracle(objective,total_budget=budget)
    if physical:
        config=PredictiveConfig();physics=replace(config.strategy.population.physics,height_scale=setting)
        config=replace(config,strategy=replace(config.strategy,population=replace(config.strategy.population,physics=physics)))
        result=run_predictive(oracle,initial,500,mode="continue" if method=="gravity" else "predictive_ball",
                              config=config,seed=seed,basin_boundary=boundary,target_value=threshold,cost_checkpoints=(1500,))
        rows=result.rows;interventions=result.interventions;decisions=result.decisions
        best=result.best_value;stop=result.stop_reason;seconds=result.wall_seconds;counts=result.counts
        for row in rows:row["actual_iteration"]=round(row["time"]/physics.h)
        shots=counts["shots"];supply=rows[-1]["positive_supply"]
        initial_energy=initial.kinetic+physics.gravity*initial.particles[0].z
    else:
        kind,*suffix=method.split("_");control=suffix[0] if suffix else "none"
        result=run_optimizer(oracle,x0,OptimizerConfig(kind=kind,lr=setting),mode=control,seed=seed,
                             max_steps=500 if protocol=="fixed_updates" else 10000,checkpoints=(1500,))
        rows=result.rows;interventions=result.interventions;decisions=result.decisions
        best=result.best_value;stop=result.stop_reason;seconds=result.wall_seconds;counts=result.counts
        for row in rows:row["actual_iteration"]=row["step"]
        shots=rows[-1]["shots"];supply=rows[-1]["supplied_model_kinetic"];initial_energy=None
    hits=[]
    for row in rows:
        row["relative_best_gap"]=max(0.,(row["best_value"]-minimum)/gap)
        if row["record"]!="observation":continue
        x=np.array(row["target_x"] if physical else row["x"])
        success=row["value"]<=threshold and (boundary is None or x[0]<boundary)
        if case=="wells20":success=success and float(np.mean(objective.coordinates(x)<0))>=.9
        if success:hits.append(row["total_oracle_calls"])
    failure=stop.startswith(("numerical_error:","contact_error:"))
    summary={"case":case,"seed":seed,"method":method,"protocol":protocol,"setting":setting,
             "stop_reason":stop,"failure":failure,"initial_f":f0,"minimum":minimum,
             "initial_x":x0.tolist(),"initial_physical_energy":initial_energy,
             "best_f":best,"relative_best_gap":max(0.,(best-minimum)/gap),"target_hit":bool(hits),
             "first_target_calls":min(hits) if hits else None,"actual_iterations":rows[-1]["actual_iteration"],
             "shots":shots,"supply_diagnostic":supply,"supply_type":"mechanical energy" if physical else "model update kinetic increment",
             "wall_seconds":seconds,**counts}
    return summary,rows,interventions,decisions


def run_benchmark(output,*,seeds=10,seed_start=200,cases=CASES,tuning_seeds=(10,11)):
    output=Path(output)
    if (output/"config.json").exists():raise ValueError("Use a new benchmark output directory.")
    for folder in [output,output/"trajectories",output/"interventions",output/"decisions"]:folder.mkdir(parents=True,exist_ok=True)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    _json(output/"config.json",{"cases":list(cases),"methods":list(METHODS),"grids":GRIDS,"tuning_seeds":list(tuning_seeds),
          "test_seeds":list(range(seed_start,seed_start+seeds)),"primary_budget":1500,"diagnostic_budget":24000,
          "primary_protocol":"Shared 1500 oracle cap; gradient methods run to cap, physical methods at most 500 macrosteps",
          "diagnostic_protocol":"At most 500 actual updates and 24000 oracle cap; explicitly not equal cost",
          "optimizer_control":asdict(ControlConfig()),"python":sys.version,"numpy":np.__version__,"source_sha256":source,
          "zero_decay_alias":"Adam and AdamW are the same update when weight decay is zero",
          "initial_state":"Same x0. Gradient optimizer states zero. Physical height and vertical kinetic reserve disclosed separately."})
    tuning=[];selected={}
    for case in cases:
        selected[case]={}
        for family,grid in GRIDS.items():
            scores=[]
            for setting in grid:
                group=[]
                for seed in tuning_seeds:
                    row,*_=run_case(case,seed,family,setting,1500)
                    tuning.append(row);group.append(row)
                scores.append((sum(r["failure"] for r in group),float(np.mean([np.log10(max(r["relative_best_gap"],1e-12)) for r in group])),setting))
                _json(output/"tuning.json",tuning)
            chosen=min(scores,key=lambda item:(item[0],item[1]))
            selected[case][family]={"setting":chosen[2],"tuning_failures":chosen[0],"mean_log10_gap":chosen[1]}
            print(f"Synthetic tuning {case} {family}: {chosen}",flush=True)
    _json(output/"selected.json",selected);_csv(output/"tuning.csv",tuning)
    runs=[]
    for protocol,budget in [("budget",1500),("fixed_updates",24000)]:
        for case in cases:
            for seed in range(seed_start,seed_start+seeds):
                for method in METHODS:
                    family="gravity" if method.startswith("gravity") else method.split("_")[0]
                    row,trace,actions,decisions=run_case(case,seed,method,selected[case][family]["setting"],budget,protocol=protocol)
                    runs.append(row);name=f"{protocol}__{case}__s{seed}__{method}"
                    _csv(output/"trajectories"/(name+".csv.gz"),trace,True)
                    _json(output/"interventions"/(name+".json"),actions);_json(output/"decisions"/(name+".json"),decisions)
                    _json(output/"runs.json",runs)
                print(f"Synthetic {protocol} {case} seed {seed}: {len(runs)} runs saved",flush=True)
    _csv(output/"runs.csv",runs)
    aggregate=[];pairs=[];lookup={(r["protocol"],r["case"],r["seed"],r["method"]):r for r in runs}
    for r in runs:
        family="gravity" if r["method"].startswith("gravity") else r["method"].split("_")[0]
        if r["method"]==family:continue
        base=lookup[r["protocol"],r["case"],r["seed"],family]
        delta=r["relative_best_gap"]-base["relative_best_gap"]
        pairs.append({"protocol":r["protocol"],"case":r["case"],"seed":r["seed"],"method":r["method"],
                      "base":family,"difference":delta,"valid":not r["failure"] and not base["failure"],
                      "outcome":"unavailable" if r["failure"] or base["failure"] else "better" if delta < -1e-8 else "worse" if delta > 1e-8 else "tie"})
    for protocol in ["budget","fixed_updates"]:
        for case in cases:
            for method in METHODS:
                group=[r for r in runs if r["protocol"]==protocol and r["case"]==case and r["method"]==method]
                valid=[r for r in group if not r["failure"]]
                p=[r for r in pairs if r["protocol"]==protocol and r["case"]==case and r["method"]==method]
                aggregate.append({"protocol":protocol,"case":case,"method":method,"runs":len(group),"failures":len(group)-len(valid),
                    "target_hits":sum(r["target_hit"] for r in valid),"median_relative_best_gap":float(np.median([r["relative_best_gap"] for r in valid])) if valid else None,
                    "median_calls":float(np.median([r["total_oracle_calls"] for r in group])),
                    "median_iterations":float(np.median([r["actual_iterations"] for r in group])),
                    "median_seconds":float(np.median([r["wall_seconds"] for r in group])),
                    "shots":sum(r["shots"] for r in group),"better":sum(r["outcome"]=="better" for r in p),
                    "worse":sum(r["outcome"]=="worse" for r in p),"ties":sum(r["outcome"]=="tie" for r in p)})
    for name,data in [("aggregate",aggregate),("paired",pairs)]:_json(output/(name+".json"),data);_csv(output/(name+".csv"),data)
    return aggregate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=Path("output/optimizer_comparison"))
    parser.add_argument("--seeds",type=int,default=10);parser.add_argument("--seed-start",type=int,default=200)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    args=parser.parse_args(argv);run_benchmark(args.output,seeds=args.seeds,seed_start=args.seed_start,cases=args.cases)
    return 0
