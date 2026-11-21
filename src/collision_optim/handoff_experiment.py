"""E18 shared-prefix handoff comparison and exact-protocol cooling replication."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace, asdict
import argparse
import hashlib
import os
from pathlib import Path
import sys
import zipfile
import numpy as np

from .cooling import trajectory_metrics
from .cooling_experiment import save_snapshot, run_one as replicate_one
from .gravity import GravityConfig
from .handoff import HandoffPopulation, particle_record, partition_state, gd_refine
from .population import PopulationConfig, PopulationOracle, run_population
from .population_experiment import initial_population, make_objective
from .strategic_experiment import _json, _csv

CASES=("quadratic","double_well","rosenbrock")
METHODS=("independent","elastic","random")
SUFFIXES=("continue","independent","gd")
SPEC="docs/superpowers/specs/2026-11-08-collision-handoff-design.md"
ROOT=Path(__file__).resolve().parents[2]


def failed(reason):return reason.startswith(("contact_error","numerical_error","task_error"))


def observations(rows, gd=False):
    if gd:return [([r["value"]],[r["x"]]) for r in rows]
    return [(r["values"],r["positions"]) for r in rows]


def score(points, minimum, initial_value, target, boundary):
    hits=[any(f<=target and (boundary is None or p[0]<boundary) for f,p in zip(values,positions)) for values,positions in points]
    gap=max(initial_value-minimum,1e-12)
    final=min(points[-1][0]);best=min(min(v) for v,p in points)
    return {"final_value":final,"best_value":best,
            "relative_final_gap":max(0.,(final-minimum)/gap),"relative_best_gap":max(0.,(best-minimum)/gap),
            "historical_hit":any(hits),"final_target_hit":hits[-1]}


def store_phase(output, kind, name, result, attempts):
    for row in result.rows:
        row["total_oracle_calls"]=sum(row[k] for k in ("value_calls","gradient_calls","path_calls"))
    _csv(output/(kind+"_traces")/(name+".csv.gz"),result.rows,True)
    _json(output/(kind+"_events")/(name+".json"),result.events)
    pairs=sum(e["kind"]=="pair" for e in result.events)
    _json(output/(kind+"_maps")/(name+".json"),[{**a,"committed":i<pairs} for i,a in enumerate(attempts)])


def prefix_task(output, case, seed, mode, steps, budget, threshold):
    output=Path(output);objective,minimum,boundary=make_objective(case)
    initial=initial_population(case,objective,seed,rho_max=.24)
    f0=min(objective.value(p.x) for p in initial.particles);target=minimum+.1*(f0-minimum)
    cfg=PopulationConfig(physics=GravityConfig(),rho=.24,collisions=mode!="independent",max_pair_tests=100000)
    method=HandoffPopulation(cfg,schedule="ramp",rule="random" if mode=="random" else "elastic",
                             seed=np.random.SeedSequence([20261108,seed,240]),handoff_budget=threshold)
    oracle=PopulationOracle(objective,total_budget=budget)
    prefix=run_population(method,oracle,initial,steps,basin_boundary=boundary,target_value=target)
    name=f"{case}__s{seed}__{mode}";calls=prefix.counts["total_oracle_calls"]
    reached=calls>=threshold and not failed(prefix.stop_reason)
    checkpoint={"case":case,"seed":seed,"prefix_method":mode,"calls":calls,
        "handoff_time":method.committed_steps*.05,"committed_steps":method.committed_steps,
        "pair_tests":method.pair_tests,"rng_state":deepcopy(method.rng.bit_generator.state),
        "particles":[particle_record(p) for p in prefix.final_state.particles],"values":prefix.rows[-1]["values"],
        "initial_value":f0,"minimum":minimum,"target":target,"boundary":boundary,
        "energy":method.energy(prefix.final_state),"momentum":prefix.final_state.momentum.tolist(),
        "reached":reached,"stop_reason":prefix.stop_reason,"counts":prefix.counts}
    _json(output/"checkpoints"/(name+".json"),checkpoint)
    store_phase(output,"prefix",name,prefix,method.map_attempts)
    prefix_row={"case":case,"seed":seed,"prefix_method":mode,"prefix_id":name,"failed":failed(prefix.stop_reason),
                "reached":reached,"total_oracle_calls":calls,"stop_reason":prefix.stop_reason,
                "committed_steps":method.committed_steps,**score(observations(prefix.rows),minimum,f0,target,boundary)}
    rows=[]
    for suffix in SUFFIXES:
        key=name+"__"+suffix
        row={"case":case,"seed":seed,"prefix_method":mode,"suffix":suffix,"prefix_id":name,
             "prefix_calls":calls,"handoff_time":checkpoint["handoff_time"],"initial_value":f0,
             "minimum":minimum,"target":target,"positive_supply":0.,"suffix_executed":reached}
        if not reached:
            rows.append({**row,"failed":True,"stop_reason":"unavailable: prefix did not reach committed handoff",
                         "total_oracle_calls":calls,"suffix_calls":0,"final_value":None,"relative_final_gap":None,
                         "best_value":prefix_row["best_value"],"relative_best_gap":prefix_row["relative_best_gap"],
                         "historical_hit":prefix_row["historical_hit"],"final_target_hit":False})
            continue
        child_oracle=deepcopy(oracle)
        if suffix=="gd":
            interface=partition_state(prefix.final_state,prefix.rows[-1]["values"])
            _json(output/"interfaces"/(key+".json"),interface)
            gd=gd_refine(child_oracle,interface["selected_state"]["position"][:-1],interface["selected_value"])
            _csv(output/"suffix_traces"/(key+".csv.gz"),gd.rows,True)
            _json(output/"gd_attempts"/(key+".json"),gd.uncommitted)
            row.update({"failed":failed(gd.stop_reason),"stop_reason":gd.stop_reason,"total_oracle_calls":gd.counts["total_oracle_calls"],
                "suffix_calls":gd.counts["total_oracle_calls"]-calls,"gd_updates":len(gd.rows)-1,
                "physical_final_time":None,"retained_energy":interface["retained_energy"],"exported_energy":interface["exported_energy"],
                **score(observations(prefix.rows)+observations(gd.rows,True),minimum,f0,target,boundary)})
        else:
            child=deepcopy(method);child.handoff_budget=None;child.map_attempts=[];child.resuming=True
            if suffix=="independent":child.config=replace(child.config,collisions=False)
            suffix_run=run_population(child,child_oracle,prefix.final_state,max(0,steps-method.committed_steps),
                                      basin_boundary=boundary,target_value=target)
            store_phase(output,"suffix",key,suffix_run,child.map_attempts)
            total_time=checkpoint["handoff_time"]+suffix_run.rows[-1]["time"]
            row.update({"failed":failed(suffix_run.stop_reason),"stop_reason":suffix_run.stop_reason,
                "total_oracle_calls":suffix_run.counts["total_oracle_calls"],"suffix_calls":suffix_run.counts["total_oracle_calls"]-calls,
                "physical_final_time":total_time,"final_kinetic":suffix_run.rows[-1]["kinetic"],
                "prefix_pair_collisions":sum(e["kind"]=="pair" for e in prefix.events),
                "suffix_pair_collisions":sum(e["kind"]=="pair" for e in suffix_run.events),
                "removed_energy":-prefix.rows[-1]["cumulative_damping"]-suffix_run.rows[-1]["cumulative_damping"],
                **score(observations(prefix.rows)+observations(suffix_run.rows),minimum,f0,target,boundary)})
        rows.append(row)
    return {"prefix":prefix_row,"runs":rows}


def replication_task(output, case, seed, schedule, steps, budget):
    row=replicate_one(Path(output)/"replication",case,seed,"elastic",schedule,steps,budget)
    row["prefix_method"]="elastic";return {"replication":row}


def aggregate(rows, replica=False):
    result=[]
    groups=((c,m,s) for c in CASES for m in (("elastic",) if replica else METHODS)
            for s in (("weak","constant","ramp") if replica else SUFFIXES))
    for case,mode,arm in groups:
        group=[r for r in rows if r["case"]==case and r["prefix_method"]==mode and r["schedule" if replica else "suffix"]==arm]
        valid=[r for r in group if not r["failed"]]
        result.append({"case":case,"prefix_method":mode,"arm":arm,"runs":len(group),"valid":len(valid),"failures":len(group)-len(valid),
            "historical_hits":sum(r["population_target_hit" if replica else "historical_hit"] for r in valid),
            "final_hits":sum(r["final_target_hit"] for r in valid),
            "median_relative_final_gap":float(np.median([r["relative_final_gap"] for r in valid])) if valid else None,
            "median_relative_best_gap":float(np.median([r["relative_best_gap"] for r in valid])) if valid else None,
            "median_total_oracle_calls":float(np.median([r["total_oracle_calls"] for r in group])) if group else None})
    return result


def comparisons(rows, replica=False):
    key="schedule" if replica else "suffix";pairs=[]
    lookup={(r["case"],r["seed"],r["prefix_method"],r[key]):r for r in rows}
    for r in rows:
        refs=[]
        if replica and r[key]=="ramp":refs=[("elastic",s,"ramp_vs_"+s) for s in ("weak","constant")]
        if not replica:
            if r[key]!="continue":refs.append((r["prefix_method"],"continue",r[key]+"_vs_continue"))
            if r[key]=="gd":refs.append((r["prefix_method"],"independent","gd_vs_independent"))
            if r["prefix_method"]=="elastic":refs.extend((m,r[key],"elastic_prefix_vs_"+m) for m in ("independent","random"))
        for mode,arm,contrast in refs:
            ref=lookup[r["case"],r["seed"],mode,arm];valid=not(r["failed"] or ref["failed"])
            diff=r["relative_final_gap"]-ref["relative_final_gap"] if valid else None
            pairs.append({"case":r["case"],"seed":r["seed"],"prefix_method":r["prefix_method"],"arm":r[key],
                "contrast":contrast,"reference_prefix":mode,"reference_arm":arm,"valid":valid,
                "final_gap_difference":diff,"outcome":"unavailable" if not valid else "better" if diff < -1e-6 else "worse" if diff > 1e-6 else "tie"})
    return pairs


def run(output, *, seeds=range(720,740), steps=500, budget=6000, threshold=3000, workers=4):
    output=Path(output);seeds=list(seeds)
    if output.exists():raise ValueError("Use a fresh output directory.")
    dirs=["prefix_traces","prefix_events","prefix_maps","checkpoints","suffix_traces","suffix_events","suffix_maps",
          "interfaces","gd_attempts","initial_states","replication/traces","replication/events","replication/map_attempts"]
    for name in dirs:(output/name).mkdir(parents=True,exist_ok=True)
    manifest=save_snapshot(output);raw=(ROOT/SPEC).read_bytes()
    with zipfile.ZipFile(output/"source.zip","a",compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr(SPEC,raw)
    manifest[SPEC]=hashlib.sha256(raw).hexdigest();_json(output/"source_manifest.json",manifest)
    _json(output/"config.json",{"cases":CASES,"seeds":seeds,"prefix_methods":METHODS,"suffixes":SUFFIXES,
        "replication_schedules":["weak","constant","ramp"],"steps":steps,"oracle_budget":budget,"handoff_threshold":threshold,
        "physics":asdict(GravityConfig()),"rho":.24,"initial_total_kinetic":.6,"gd_eta":.02,"workers":workers,
        "launch_argv":sys.argv,"python":sys.version,"numpy":np.__version__,"spec":SPEC,
        "environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTHONPATH")},
        "source_sha256":manifest,"source_archive_sha256":hashlib.sha256((output/"source.zip").read_bytes()).hexdigest(),
        "counting":"One executed prefix per case/seed/method, three executed suffixes; full-pipeline views charge prefix each time. Replica trajectories executed separately."})
    tasks=[]
    for case in CASES:
        obj,minimum,boundary=make_objective(case)
        for seed in seeds:
            initial=initial_population(case,obj,seed,rho_max=.24);f0=min(obj.value(p.x) for p in initial.particles)
            _json(output/"initial_states"/f"{case}__s{seed}.json",{"particles":[particle_record(p) for p in initial.particles],
                  "minimum":minimum,"boundary":boundary,"initial_value":f0,"target":minimum+.1*(f0-minimum)})
            tasks.extend((prefix_task,(output,case,seed,m,steps,budget,threshold)) for m in METHODS)
            tasks.extend((replication_task,(output,case,seed,s,steps,budget)) for s in ("weak","constant","ramp"))
    prefixes=[];runs=[];replicas=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,*args) for fn,args in tasks]
        for i,future in enumerate(as_completed(futures),1):
            try:
                data=future.result()
            except Exception as exc:
                _json(output/"task_failure.json",{"task_index":futures.index(future),"exception":repr(exc),
                    "batch_incomplete":True,"note":"Saved prefixes/branches retained; no complete-batch inference. Active worker counters may not be recoverable."})
                raise
            if "prefix" in data:prefixes.append(data["prefix"]);runs.extend(data["runs"])
            else:replicas.append(data["replication"])
            for name,rows in (("prefixes",prefixes),("runs",runs),("replication_runs",replicas)):_json(output/(name+".json"),rows)
            if i%6==0:print(f"Handoff tasks {i}/{len(tasks)}, prefixes={len(prefixes)}, suffixes={len(runs)}, replicas={len(replicas)}, failures={sum(r['failed'] for r in runs+replicas)}",flush=True)
    for name,rows in (("prefixes",prefixes),("runs",runs),("replication_runs",replicas),
                      ("aggregate",aggregate(runs)),("replication_aggregate",aggregate(replicas,True)),
                      ("paired",comparisons(runs)),("replication_paired",comparisons(replicas,True))):
        _json(output/(name+".json"),rows);_csv(output/(name+".csv"),rows)
    actual=sum(r["total_oracle_calls"] for r in prefixes)+sum(r["suffix_calls"] for r in runs)+sum(r["total_oracle_calls"] for r in replicas)
    _json(output/"execution_counts.json",{"prefix_executions":len(prefixes),"suffix_executions":sum(r["suffix_executed"] for r in runs),
        "uninterrupted_executions":len(replicas),"derived_split_pipeline_rows":len(runs),
        "actual_query_calls":actual,"charged_pipeline_query_calls":sum(r["total_oracle_calls"] for r in runs+replicas),
        "independent_seed_count":len(seeds),"note":"Sharing a prefix saves actual execution; each compared pipeline still pays its full prefix cost."})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed-start",type=int,default=720);p.add_argument("--seeds",type=int,default=20)
    p.add_argument("--steps",type=int,default=500);p.add_argument("--budget",type=int,default=6000)
    p.add_argument("--threshold",type=int,default=3000);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args();run(a.output,seeds=range(a.seed_start,a.seed_start+a.seeds),steps=a.steps,budget=a.budget,threshold=a.threshold,workers=a.workers)
