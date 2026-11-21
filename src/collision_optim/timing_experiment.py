"""E19 timing sweep with shared all-on checkpoints and untouched confirmation."""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np

from .timing import CheckpointPopulation,resume_off,protected_population,multistart,FRACTIONS
from .gravity import GravityConfig
from .population import PopulationConfig,PopulationOracle,run_population
from .population_experiment import make_objective,initial_population
from .handoff import particle_record
from .handoff_experiment import score,observations,failed
from .cooling_experiment import save_snapshot
from .strategic_experiment import _json,_csv

ROOT=Path(__file__).resolve().parents[2]
SPEC='docs/superpowers/specs/2026-11-09-collision-timing-design.md'
CASES=('quadratic','double_well','rosenbrock')
RATES=(.005,.02,.05,.1)


def save_run(output,name,result,attempts):
    for row in result.rows:row['total_oracle_calls']=sum(row[k] for k in ('value_calls','gradient_calls','path_calls'))
    _csv(output/'traces'/(name+'.csv.gz'),result.rows,True)
    _json(output/'events'/(name+'.json'),result.events)
    committed=sum(e['kind']=='pair' for e in result.events)
    _json(output/'maps'/(name+'.json'),[{**a,'committed':i<committed} for i,a in enumerate(attempts)])


def summarize_path(points,initial,minimum,target,boundary,safety):
    if not safety:return score(points,minimum,min(initial),target,boundary)
    target_values=[values[0] for values,positions in points]
    rise=max(target_values)-initial[0]
    return {'final_value':min(points[-1][0]),'best_value':min(min(v) for v,p in points),
        'protected_initial_value':initial[0],'protected_final_rise':target_values[-1]-initial[0],
        'protected_max_rise':max(0.,rise),'protected_harmed':rise>1e-4}


def physical_task(output,case,seed,mode,budget,steps,safety=False):
    output=Path(output);obj,minimum,boundary=make_objective(case)
    initial=protected_population(case,obj,minimum,seed) if safety else initial_population(case,obj,seed,rho_max=.24)
    values=[obj.value(p.x) for p in initial.particles];f0=min(values);target=minimum+.1*(f0-minimum)
    thresholds=[int(budget*f) for f in (.25,.5,.75)] if mode!='independent' else []
    method=CheckpointPopulation(PopulationConfig(physics=GravityConfig(),rho=.24,collisions=mode!='independent',max_pair_tests=100000),
        rule='random' if mode=='random' else 'elastic',seed=np.random.SeedSequence([20261108,seed,240]),
        thresholds=thresholds,budget=budget,initial_value=f0)
    oracle=PopulationOracle(obj,total_budget=budget)
    full=run_population(method,oracle,initial,steps,basin_boundary=boundary,target_value=None if safety else target)
    name=f'{case}__s{seed}__{mode}';save_run(output,name+'__full',full,method.map_attempts)
    _json(output/'initial_states'/(name+'.json'),{'particles':[particle_record(p) for p in initial.particles],
        'values':values,'minimum':minimum,'boundary':boundary,'target':target,'safety':safety,
        'kinetic':initial.kinetic,'momentum':initial.momentum.tolist(),'intended_early_pair':safety})
    rows=[{'case':case,'seed':seed,'rule':mode,'fraction':0. if mode=='independent' else 1.,'full_id':name,
        'trace_id':name+'__full','checkpoint':None,'failed':failed(full.stop_reason),'stop_reason':full.stop_reason,
        'total_oracle_calls':full.counts['total_oracle_calls'],'actual_suffix_calls':0,'physical_time':full.rows[-1]['time'],
        'wall_seconds':full.wall_seconds,'safety':safety,
        **summarize_path(observations(full.rows),values,minimum,target,boundary,safety)}]
    checkpoints=[]
    for fraction,threshold in zip((.25,.5,.75),thresholds):
        key=name+f'__off{fraction}';snap=method.snapshots.get(threshold)
        if snap is None or snap['step']>=len(full.rows):
            rows.append({'case':case,'seed':seed,'rule':mode,'fraction':fraction,'full_id':name,'trace_id':None,
                'failed':True,'stop_reason':'unavailable checkpoint','total_oracle_calls':full.counts['total_oracle_calls'],
                'actual_suffix_calls':0,'relative_final_gap':None,'final_target_hit':False,'historical_hit':False,'safety':safety})
            continue
        cp={'case':case,'seed':seed,'rule':mode,'fraction':fraction,'threshold':threshold,'step':snap['step'],'calls':snap['calls'],
            'particles':[particle_record(p) for p in snap['state'].particles],'values':snap['values'],'rng':snap['rng'],
            'pair_tests':snap['pair_tests'],'features':snap['features'],'history_step_range':snap['history_step_range'],
            'feature_last_observation_calls':snap['feature_last_observation_calls'],'source_full_id':name,
            'minimum':minimum,'initial_value':f0,'target':target,'boundary':boundary,'safety':safety}
        _json(output/'checkpoints'/(key+'.json'),cp);checkpoints.append(cp)
        child,child_oracle=resume_off(snap)
        suffix=run_population(child,child_oracle,snap['state'],max(0,steps-snap['step']),basin_boundary=boundary,target_value=None if safety else target)
        save_run(output,key,suffix,child.map_attempts)
        points=observations(full.rows[:snap['step']+1])+observations(suffix.rows)
        rows.append({'case':case,'seed':seed,'rule':mode,'fraction':fraction,'full_id':name,'trace_id':key,'checkpoint':key,
            'failed':failed(suffix.stop_reason),'stop_reason':suffix.stop_reason,'total_oracle_calls':suffix.counts['total_oracle_calls'],
            'prefix_calls':snap['calls'],'actual_suffix_calls':suffix.counts['total_oracle_calls']-snap['calls'],
            'physical_time':(snap['step']+len(suffix.rows)-1)*.05,'suffix_wall_seconds':suffix.wall_seconds,'safety':safety,
            **summarize_path(points,values,minimum,target,boundary,safety)})
    return {'runs':rows,'checkpoints':checkpoints,'full_calls':full.counts['total_oracle_calls'],
            'suffix_calls':sum(r['actual_suffix_calls'] for r in rows),'full_id':name}


def baseline_task(output,case,seed,family,eta,budget):
    output=Path(output);obj,minimum,boundary=make_objective(case);initial=initial_population(case,obj,seed,rho_max=.24)
    values=[obj.value(p.x) for p in initial.particles];target=minimum+.1*(min(values)-minimum)
    rows,reason,counts,attempts=multistart(PopulationOracle(obj,total_budget=budget),initial,family=family,eta=eta)
    name=f'{case}__s{seed}__{family}__eta{eta}'
    _csv(output/'baseline_traces'/(name+'.csv.gz'),rows,True);_json(output/'baseline_attempts'/(name+'.json'),attempts)
    row={'case':case,'seed':seed,'family':family,'eta':eta,'trace_id':name,'failed':failed(reason),'stop_reason':reason,
        **counts,**score(observations(rows),minimum,min(values),target,boundary)}
    return {'baseline':row}


def aggregate(runs,safety=False):
    groups={}
    for r in runs:groups.setdefault((r['case'],r['rule'],r['fraction']),[]).append(r)
    result=[]
    for (case,rule,fraction),group in sorted(groups.items()):
        valid=[r for r in group if not r['failed']]
        row={'case':case,'rule':rule,'fraction':fraction,'runs':len(group),'valid':len(valid),'failures':len(group)-len(valid)}
        if safety:
            row.update(harmed=sum(r['protected_harmed'] for r in valid),median_max_rise=float(np.median([r['protected_max_rise'] for r in valid])) if valid else None,
                median_final_rise=float(np.median([r['protected_final_rise'] for r in valid])) if valid else None)
        else:
            row.update(final_hits=sum(r['final_target_hit'] for r in valid),historical_hits=sum(r['historical_hit'] for r in valid),
                median_relative_final_gap=float(np.median([r['relative_final_gap'] for r in valid])) if valid else None)
        result.append(row)
    return result


def run(output,*,stage,seeds,budget=6000,steps=500,workers=4,selection=None):
    output=Path(output);seeds=list(seeds);safety=stage=='safety';preflight=stage=='preflight'
    if output.exists():raise ValueError('Use a fresh output directory.')
    for name in ('traces','events','maps','checkpoints','initial_states','baseline_traces','baseline_attempts'):(output/name).mkdir(parents=True,exist_ok=True)
    manifest=save_snapshot(output);raw=(ROOT/SPEC).read_bytes()
    with zipfile.ZipFile(output/'source.zip','a',compression=zipfile.ZIP_DEFLATED) as z:z.writestr(SPEC,raw)
    manifest[SPEC]=hashlib.sha256(raw).hexdigest();_json(output/'source_manifest.json',manifest)
    selected=json.loads(Path(selection).read_text()) if selection else None
    if stage=='confirmation' and selected is None:raise ValueError('Confirmation requires frozen development selection.')
    if selected:_json(output/'selection_input.json',selected)
    _json(output/'config.json',{'stage':stage,'seeds':seeds,'cases':CASES,'fractions':FRACTIONS,'oracle_budget':budget,'steps':steps,
        'physics':asdict(GravityConfig()),'rho':.24,'initial_kinetic':.6,'workers':workers,'source_sha256':manifest,
        'source_archive_sha256':hashlib.sha256((output/'source.zip').read_bytes()).hexdigest(),'spec':SPEC,
        'launch_argv':sys.argv,'python':sys.version,'numpy':np.__version__,
        'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH')},
        'selection_path':str(selection) if selection else None,'selection_sha256':hashlib.sha256(Path(selection).read_bytes()).hexdigest() if selection else None,
        'counting':'Full all-on runs reused as prefixes for three off suffixes; each policy charged its prefix, actual execution counted once.'})
    tasks=[]
    for case in CASES:
        for seed in seeds:
            for mode in (('independent','elastic') if safety else ('independent','elastic','random')):
                tasks.append((physical_task,(output,case,seed,mode,budget,steps,safety)))
            if not safety:
                for family in ('gd','momentum'):
                    rates=(.02,) if preflight else RATES if stage=='development' else (selected['baseline_rates'][case][family],)
                    tasks.extend((baseline_task,(output,case,seed,family,eta,budget)) for eta in rates)
    runs=[];baselines=[];checkpoints=[];actual=0;completed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,*args):{'name':fn.__name__,'case':args[1],'seed':args[2]} for fn,args in tasks}
        try:
            for i,future in enumerate(as_completed(futures),1):
                result=future.result();completed.append(futures[future])
                if 'baseline' in result:baselines.append(result['baseline']);actual+=result['baseline']['total_oracle_calls']
                else:runs.extend(result['runs']);checkpoints.extend(result['checkpoints']);actual+=result['full_calls']+result['suffix_calls']
                for name,data in (('runs',runs),('baselines',baselines),('checkpoints',checkpoints)):_json(output/(name+'.json'),data)
                if i%10==0 or i==len(futures):print(f'Timing {stage} {i}/{len(futures)} tasks, policies={len(runs)}, baselines={len(baselines)}, failures={sum(r["failed"] for r in runs+baselines)}',flush=True)
        except BaseException as exc:
            _json(output/'task_failure.json',{'exception':repr(exc),'completed_tasks':completed,'incomplete':True})
            for f in futures:f.cancel()
            raise
    for name,data in (('runs',runs),('baselines',baselines),('aggregate',aggregate(runs,safety))):_json(output/(name+'.json'),data);_csv(output/(name+'.csv'),data)
    _json(output/'execution_counts.json',{'full_runs':len(seeds)*3*(2 if safety else 3),'suffix_runs':len(checkpoints),
        'baseline_runs':len(baselines),'policy_views':len(runs),'checkpoints':len(checkpoints),'actual_query_calls':actual,
        'charged_policy_and_baseline_queries':sum(r['total_oracle_calls'] for r in runs+baselines),'seed_count':len(seeds)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['preflight','development','confirmation','safety'],required=True)
    p.add_argument('--seed-start',type=int,required=True);p.add_argument('--seeds',type=int,required=True)
    p.add_argument('--budget',type=int,default=6000);p.add_argument('--steps',type=int,default=500);p.add_argument('--workers',type=int,default=4);p.add_argument('--selection',type=Path)
    a=p.parse_args();run(a.output,stage=a.stage,seeds=range(a.seed_start,a.seed_start+a.seeds),budget=a.budget,steps=a.steps,workers=a.workers,selection=a.selection)
