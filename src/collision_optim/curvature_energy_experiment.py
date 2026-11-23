"""Fixed E22 energy-allocation study with explicit shared-probe accounting."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,os,platform,sys,traceback,zipfile
from pathlib import Path
import numpy as np

from .curvature_energy import (POLICIES,STIFFNESSES,CONDITIONS,RotatedEnergyWells,parameter_cloud,
    measured_curvatures,allocate,initial_state,IndependentEnergyPopulation,score,explorer_velocities)
from .event_experiment import save_trace
from .event_replay import pack
from .gravity import GravityConfig
from .population import PopulationConfig,PopulationOracle,run_population
from .strategic_experiment import _json,_csv

ROOT=Path(__file__).resolve().parents[2]
SPEC='docs/superpowers/specs/2026-11-09-curvature-energy-design.md'
PLAN='docs/superpowers/plans/2026-11-09-curvature-energy-plan.md'


def snapshot(output):
    paths=set(ROOT.glob('*.py'))
    for name in ('src','tests','scripts'):paths.update(p for p in (ROOT/name).rglob('*.py') if '__pycache__' not in p.parts)
    paths.update(ROOT/f for f in ('AGENTS.md','pyproject.toml',SPEC,PLAN))
    amendment=ROOT/'docs/research/2026-11-10-energy-exploration-amendment.md'
    if amendment.exists():paths.add(amendment)
    manifest={}
    with zipfile.ZipFile(output/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            data=p.read_bytes();name=str(p.relative_to(ROOT));z.writestr(name,data);manifest[name]=hashlib.sha256(data).hexdigest()
    _json(output/'source_manifest.json',manifest);return manifest


def task(output,condition,stiffness,seed,budget,steps,kinetic=False):
    output=Path(output);objective=RotatedEnergyWells(stiffness,seed);points,roots=parameter_cloud(objective,condition,seed)
    velocities=explorer_velocities(seed) if kinetic else np.zeros((4,2))
    probe=PopulationOracle(objective);values,curvatures,raw_probes=measured_curvatures(probe,points)
    references=[objective.value(objective.from_coordinates([r,0.])) for r in roots]
    minimum=references[0];target=minimum+(.1*(min(values)-minimum) if condition=='bad_only' else 1e-4)
    name=f'{condition}__k{stiffness}__s{seed}'
    context={'context_id':name,'condition':condition,'stiffness':stiffness,'seed':seed,'rotation':objective.rotation.tolist(),
             'points':[x.tolist() for x in points],'values':values,'curvatures':curvatures,'curvature_probes':raw_probes,
             'reference_roots':roots.tolist(),'reference_values':references,'minimum':minimum,'boundary':float(roots[1]),
             'target':target,'reference_point0':objective.from_coordinates([roots[0] if condition=='good_present' else roots[-1],0.]).tolist(),
             'actual_shared_probe_calls':sum(probe.counts().values()),'probe_counts':probe.counts(),'setup_reference_value_calls':3,
             'initial_horizontal_velocities':velocities.tolist(),'initial_kinetic':float(.5*np.sum(velocities*velocities)),
             'velocity_regime':'moving_explorers_K.6_P0' if kinetic else 'all_at_rest'}
    _json(output/'contexts'/(name+'.json'),context);runs=[];actual_branches=0
    for policy in POLICIES:
        allocation=allocate(values,curvatures,policy,seed);state=initial_state(points,values,allocation,velocities)
        cfg=PopulationConfig(physics=GravityConfig(),rho=.24,collisions=False)
        method=IndependentEnergyPopulation(cfg);oracle=PopulationOracle(objective,total_budget=budget)
        oracle.values=allocation['charged_value_prefix'];oracle.gradients=allocation['charged_gradient_prefix']
        prefix=sum(oracle.counts().values());result=run_population(method,oracle,state,steps)
        key=name+'__'+policy
        save_trace(output,key,result.rows,result.events,method.map_attempts,
                   {'stop_reason':result.stop_reason,'last_committed_state':pack(result.final_state),'counts':result.counts,
                    'partial_intermediate_states':'Not returned by frozen event solver'})
        _json(output/'initial_states'/(key+'.json'),{**pack(state),'values':values,'allocation':allocation,'population_config':asdict(cfg),
                 'energy':result.rows[0]['energy'],'momentum':state.momentum.tolist(),'context_id':name,
                 'prefix_charge_is_derived_shared_probes':True})
        actual=result.counts['total_oracle_calls']-prefix;actual_branches+=actual
        runs.append({'run_id':key,'context_id':name,'condition':condition,'stiffness':stiffness,'seed':seed,'policy':policy,
             'failed':result.stop_reason.startswith(('contact_error:','numerical_error:')),'stop_reason':result.stop_reason,
             'total_oracle_calls':result.counts['total_oracle_calls'],'actual_branch_calls':actual,'charged_prefix_calls':prefix,
             'physical_time':result.rows[-1]['time'],'wall_seconds':result.wall_seconds,
             'initial_energy':result.rows[0]['energy'],'initial_clearance_potential':allocation['clearance_potential'],
             'initial_kinetic':state.kinetic,
             'selected_index':allocation['selected_index'],'heights':allocation['heights'],**score(result.rows,context,objective)})
    return {'contexts':[context],'runs':runs,'shared_probe_calls':context['actual_shared_probe_calls'],
            'actual_branch_calls':actual_branches,'setup_reference_values':3}


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    for name in ('traces','events','maps','uncommitted','initial_states','contexts'):(output/name).mkdir()
    manifest=snapshot(output);kinetic=args.stage.startswith('kinetic');preflight=args.stage in ('preflight','kinetic_preflight')
    seeds=[1430 if kinetic else 1390] if preflight else list(range(1440,1460) if kinetic else range(1400,1420))
    budget=1200 if preflight else 6000;steps=80 if preflight else 500
    config={'phase':'E22','stage':args.stage,'seeds':seeds,'conditions':list(CONDITIONS),'stiffnesses':list(STIFFNESSES),
            'policies':list(POLICIES),'query_cap':budget,'max_steps':steps,'h':.05,'gravity':.5,'binary_interactions':False,
            'initial_kinetic':.6 if kinetic else 0.,'velocity_regime':'moving_explorers_K.6_P0' if kinetic else 'all_at_rest',
            'high_height':.8,'low_height':.02,'curvature_product_cap':.2,'probe_step':1e-5,'initial_displacement':.001,
            'source_sha256':manifest,'source_archive_sha256':hashlib.sha256((output/'source.zip').read_bytes()).hexdigest(),
            'launch_argv':sys.argv,'cwd':str(Path.cwd()),'launch_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
            'numpy':np.__version__,'platform':platform.platform(),'workers':args.workers,
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'accounting_note':'Shared20 probes executed once/context; fixed policies charge4, curvature policies20, then all pay initialization revalidation and trajectory calls. Setup/scoring references separately counted.'}
    _json(output/'config.json',config);contexts=[];runs=[];costs={'shared_probe_calls':0,'actual_branch_calls':0,'setup_reference_values':0}
    def save():
        _json(output/'contexts.json',contexts);_json(output/'runs.json',runs)
        if runs:_csv(output/'runs.csv',runs)
        _json(output/'execution_counts.json',{**costs,'contexts':len(contexts),'trajectory_executions':len(runs),
             'actual_solver_queries':costs['shared_probe_calls']+costs['actual_branch_calls'],
             'actual_queries_including_setup':sum(costs.values()),'charged_policy_queries':sum(r['total_oracle_calls'] for r in runs),
             'independent_seed_groups':len(seeds)})
    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(task,str(output),condition,k,seed,budget,steps,kinetic) for seed in seeds for condition in CONDITIONS for k in STIFFNESSES]
        try:
            for i,future in enumerate(as_completed(futures),1):
                result=future.result();contexts.extend(result['contexts']);runs.extend(result['runs'])
                for key in costs:costs[key]+=result[key]
                save()
                if i%5==0 or i==len(futures):print(f'{args.stage}: {i}/{len(futures)} contexts, {len(runs)} trajectories',flush=True)
        except BaseException:
            _json(output/'task_failure.json',{'traceback':traceback.format_exc(),'active_task_cost':'Unknown if not returned; incremental artifacts preserved'})
            for future in futures:future.cancel()
            raise
    print(json.dumps(json.loads((output/'execution_counts.json').read_text())),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=('preflight','formal','kinetic_preflight','kinetic'),required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.workers<1:p.error('Positive workers required.')
    run(a)


if __name__=='__main__':main()
