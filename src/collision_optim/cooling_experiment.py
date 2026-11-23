"""E17 fixed physical-drag screening with shared query budgets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

from .cooling import CoolingPopulation, SCHEDULES, GAMMA0, GAMMA1, HORIZON, trajectory_metrics
from .gravity import GravityConfig
from .population import PopulationConfig, PopulationOracle, run_population
from .population_experiment import initial_population, make_objective, _summarize
from .strategic_experiment import _csv, _json

ROOT = Path(__file__).resolve().parents[2]
CASES = ("quadratic", "double_well", "rosenbrock")
METHODS = ("independent", "elastic", "random")
SPEC = "docs/superpowers/specs/2026-11-08-collision-cooling-design.md"


def save_snapshot(output):
    paths = set(ROOT.glob("*.py"))
    for name in ("src", "tests", "scripts"):
        paths.update(p for p in (ROOT/name).rglob("*.py") if "__pycache__" not in p.parts)
    paths.update(ROOT/p for p in ("AGENTS.md", "pyproject.toml", SPEC))
    manifest = {}
    with zipfile.ZipFile(output/"source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            name = str(p.relative_to(ROOT)); data = p.read_bytes()
            archive.writestr(name,data); manifest[name]=hashlib.sha256(data).hexdigest()
    _json(output/"source_manifest.json",manifest)
    return manifest


def run_one(output, case, seed, mode, schedule, steps, budget):
    output = Path(output)
    objective, minimum, boundary = make_objective(case)
    initial = initial_population(case, objective, seed, rho_max=.24)
    f0 = min(objective.value(p.x) for p in initial.particles)
    target = minimum+.1*(f0-minimum)
    cfg = PopulationConfig(physics=GravityConfig(),rho=.24,collisions=mode!="independent",max_pair_tests=100000)
    method = CoolingPopulation(cfg, schedule=schedule, rule="random" if mode=="random" else "elastic",
                               seed=np.random.SeedSequence([20261108,seed,240]))
    result = run_population(method,PopulationOracle(objective,total_budget=budget),initial,steps,
                            basin_boundary=boundary,target_value=target)
    row = _summarize(result,initial,minimum,case=case,seed=seed,method=mode,rho=None if mode=="independent" else .24)
    final=result.rows[-1]; gap=max(f0-minimum,1e-12)
    for point in result.rows:
        point["relative_current_gap"]=max(0.,(min(point["values"])-minimum)/gap)
    impacts=[e for e in result.events if e["kind"]=="pair"]
    row.update({"schedule":schedule,"minimum":minimum,"target":target,
                "final_value":min(final["values"]),"relative_final_gap":final["relative_current_gap"],
                "final_observation_calls":final["total_oracle_calls"],
                "positive_supply":0.,"no_new_particles":len(result.final_state.particles)==4,
                "map_attempts":len(method.map_attempts),"uncommitted_maps":len(method.map_attempts)-len(impacts),
                **trajectory_metrics(result.rows,target,boundary)})
    row["final_target_hit"]=row["observed_final_hit"]
    name=f"{case}__s{seed}__{mode}__{schedule}"
    _csv(output/"traces"/(name+".csv.gz"),result.rows,True)
    _json(output/"events"/(name+".json"),result.events)
    _json(output/"map_attempts"/(name+".json"),[{**a,"committed":i<len(impacts)} for i,a in enumerate(method.map_attempts)])
    return row


def aggregate(runs):
    summaries=[]
    for case in CASES:
        for mode in METHODS:
            for schedule in SCHEDULES:
                group=[r for r in runs if (r["case"],r["method"],r["schedule"])==(case,mode,schedule)]
                valid=[r for r in group if not r["failed"]]
                row={"case":case,"method":mode,"schedule":schedule,"runs":len(group),"valid":len(valid),
                     "failures":len(group)-len(valid),"historical_hits":sum(r["observed_ever_hit"] for r in valid),
                     "final_hits":sum(r["final_target_hit"] for r in valid),
                     "lost_after_hit":sum(r["lost_after_any_hit"] for r in valid),
                     "pair_collisions":sum(r["pair_collisions"] for r in group)}
                for key in ("relative_final_gap","relative_best_gap","final_kinetic","removed_energy",
                            "final_clearance_potential","committed_drag_exposure","tail_target_occupancy",
                            "total_oracle_calls","completed_time","wall_seconds"):
                    vals=[r[key] for r in valid if r[key] is not None]
                    row["median_"+key]=float(np.median(vals)) if vals else None
                summaries.append(row)
    return summaries


def paired(runs):
    lookup={(r["case"],r["seed"],r["method"],r["schedule"]):r for r in runs}; pairs=[]
    for r in runs:
        refs=[]
        if r["schedule"]!="weak": refs.append((r["method"],"weak","schedule_vs_weak"))
        if r["schedule"]=="ramp":refs.append((r["method"],"constant","ramp_vs_constant"))
        if r["method"]=="elastic":refs.extend((m,r["schedule"],"elastic_vs_"+m) for m in ("independent","random"))
        for mode,schedule,contrast in refs:
            ref=lookup[r["case"],r["seed"],mode,schedule]; valid=not(r["failed"] or ref["failed"])
            diff=r["relative_final_gap"]-ref["relative_final_gap"]
            pairs.append({"case":r["case"],"seed":r["seed"],"method":r["method"],"schedule":r["schedule"],
                          "contrast":contrast,"reference_method":mode,"reference_schedule":schedule,"valid":valid,
                          "final_gap_difference":diff if valid else None,
                          "outcome":"unavailable" if not valid else "better" if diff < -1e-6 else "worse" if diff > 1e-6 else "tie"})
    return pairs


def run(output, *, seeds=range(600,610), steps=500, budget=6000, workers=4):
    output=Path(output); seeds=list(seeds)
    if (output/"config.json").exists():raise ValueError("Use a fresh output directory.")
    for path in (output,*(output/name for name in ("traces","events","initial_states","map_attempts"))):path.mkdir(parents=True,exist_ok=True)
    manifest=save_snapshot(output)
    _json(output/"config.json",{"seeds":seeds,"steps":steps,"cases":CASES,"methods":METHODS,"schedules":SCHEDULES,
          "rho":.24,"physics":asdict(GravityConfig()),"population_size":4,"initial_total_kinetic":.6,
          "oracle_budget":budget,"pair_prediction_budget":100000,"gamma0":float(GAMMA0),"gamma1":GAMMA1,
          "nominal_horizon":HORIZON,"workers":workers,"launch_argv":sys.argv,"python":sys.version,"numpy":np.__version__,
          "environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTHONPATH")},
          "source_sha256":manifest,"source_archive_sha256":hashlib.sha256((output/"source.zip").read_bytes()).hexdigest(),
          "spec":SPEC,"scoring":"Last complete observed state; failures stay in denominator; no target-triggered stopping",
          "execution":"One parent, disjoint unique run tasks; four processes, numeric threads1; no duplicated merged executions"})
    tasks=[]
    for case in CASES:
        obj,minimum,boundary=make_objective(case)
        for seed in seeds:
            initial=initial_population(case,obj,seed,rho_max=.24); f0=min(obj.value(p.x) for p in initial.particles)
            _json(output/"initial_states"/f"{case}__s{seed}.json",{"positions":[np.r_[p.x,p.z].tolist() for p in initial.particles],
                  "velocities":[p.velocity.tolist() for p in initial.particles],"minimum":minimum,"boundary":boundary,
                  "target":minimum+.1*(f0-minimum),"initial_energy":initial.kinetic+.5*sum(p.z for p in initial.particles)})
            tasks.extend((output,case,seed,mode,schedule,steps,budget) for mode in METHODS for schedule in SCHEDULES)
    runs=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_one,*task) for task in tasks]
        for future in as_completed(futures):
            runs.append(future.result()); _json(output/"runs.json",runs)
            if len(runs)%9==0: print(f"Cooling {len(runs)}/{len(tasks)}, failures={sum(r['failed'] for r in runs)}",flush=True)
    runs.sort(key=lambda r:(r["case"],r["seed"],r["method"],r["schedule"]))
    for name,data in (("runs",runs),("aggregate",aggregate(runs)),("paired",paired(runs))):
        _json(output/(name+".json"),data);_csv(output/(name+".csv"),data)
    return runs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed-start",type=int,default=600);parser.add_argument("--seeds",type=int,default=10)
    parser.add_argument("--steps",type=int,default=500);parser.add_argument("--workers",type=int,default=4)
    a=parser.parse_args();run(a.output,seeds=range(a.seed_start,a.seed_start+a.seeds),steps=a.steps,workers=a.workers)
