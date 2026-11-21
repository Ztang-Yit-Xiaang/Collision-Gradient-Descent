"""E21 empirical trajectory isolation and return-map batches."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,os,platform,sys,traceback,zipfile
from pathlib import Path
import numpy as np

from .cooling import CoolingPopulation
from .event_experiment import SetupObjective,save_trace
from .event_replay import pack
from .gravity import GravityConfig
from .population import PopulationConfig,PopulationOracle,run_population
from .population_experiment import make_objective
from .orbit_stability import (CURVATURES,PRODUCTS,ROLLOUT_PRODUCTS,PERTURBATIONS,
                        rosen_basis,controlled_initial,bowl_returns,empirical_matrices)
from .strategic_experiment import _json,_csv

ROOT=Path(__file__).resolve().parents[2]
SPEC='docs/superpowers/specs/2026-11-09-collision-stability-design.md'
PLAN='docs/superpowers/plans/2026-11-09-collision-stability-plan.md'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(output):
    paths=set(ROOT.glob('*.py'))
    for name in ('src','tests'):paths.update(p for p in (ROOT/name).rglob('*.py') if '__pycache__' not in p.parts)
    paths.update(ROOT/p for p in ('AGENTS.md','pyproject.toml',SPEC,PLAN));manifest={}
    with zipfile.ZipFile(output/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            data=path.read_bytes();name=str(path.relative_to(ROOT));z.writestr(name,data)
            manifest[name]=hashlib.sha256(data).hexdigest()
    _json(output/'source_manifest.json',manifest);return manifest


def prepare_rosen(output,preflight=False):
    parents={};provenance=[]
    for seed in ((880,) if preflight else (880,881,882)):
        path=ROOT/'output/collision_event_sensitivity_long/initial_states'/f'rosenbrock__s{seed}__h0.05__tight0.json'
        destination=output/'parents'/f'cloud_{seed}.json';destination.write_bytes(path.read_bytes())
        parents[str(seed)]=json.loads(path.read_text())
        provenance.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'preserved_copy':str(destination.relative_to(output))})
    obj,_,_=make_objective('rosenbrock');optimum=np.array(parents['880']['positions'][0][:-1])
    basis=rosen_basis(obj,optimum);height=parents['880']['positions'][0][-1]-basis['baseline_value']
    setup={'basis':basis,'original_height':height,'parent_inputs':provenance,
           'setup_value_calls':basis['setup_value_calls'],'setup_gradient_calls':basis['setup_gradient_calls']}
    _json(output/'setup.json',setup);return setup


def rosen_task(output,layout,seed,height_label,height,h,perturbation,horizon):
    output=Path(output);setup=json.loads((output/'setup.json').read_text());basis=setup['basis']
    parent=json.loads((output/'parents'/f'cloud_{seed or 880}.json').read_text())
    obj,_,_=make_objective('rosenbrock');counter=SetupObjective(obj)
    initial=controlled_initial(parent,basis,height,perturbation,layout,counter)
    initial_values=[counter.value(p.x) for p in initial.particles]
    physics=GravityConfig(h=h);cfg=PopulationConfig(physics=physics,rho=.24,collisions=False)
    method=CoolingPopulation(cfg,schedule='ramp',rule='elastic')
    result=run_population(method,PopulationOracle(obj,total_budget=100000),initial,round(horizon/h))
    name=f'rosen__{layout}__s{seed or 0}__H{height_label}__h{h}__{perturbation}'
    save_trace(output,name,result.rows,result.events,method.map_attempts,
               {'stop_reason':result.stop_reason,'last_committed_state':pack(result.final_state),
                'counts':result.counts,'partial_intermediate_states':'Unavailable in frozen solver'})
    _json(output/'initial_states'/(name+'.json'),{**pack(initial),'values':initial_values,
        'population_config':asdict(cfg),'height':height,'optimum':basis['optimum'],
        'perturbation':perturbation,'layout':layout,'spectator_seed':seed,'reference_loss':basis['baseline_value']})
    rise=[r['values'][0]-basis['baseline_value'] for r in result.rows]
    distances=[float(np.linalg.norm(np.array(r['positions'][0][:-1])-basis['optimum'])) for r in result.rows]
    record={'run_id':name,'case':'rosenbrock','kind':'rosen_trajectory','layout':layout,'spectator_seed':seed,
            'height_label':height_label,'height':height,'h':h,'perturbation':perturbation,
            'stop_reason':result.stop_reason,'failed':result.stop_reason.startswith(('contact_error:','numerical_error:')),
            'time':result.rows[-1]['time'],'final_loss':result.rows[-1]['values'][0],
            'best_loss':min(r['values'][0] for r in result.rows),'initial_loss':initial_values[0],
            'max_loss_rise':max(0.,max(rise)),'final_loss_rise':rise[-1],
            'max_displacement':max(distances),'final_displacement':distances[-1],
            'first_time_above_1e-4':next((r['time'] for r,d in zip(result.rows,rise) if d>1e-4),None),
            'total_oracle_calls':result.counts['total_oracle_calls'],'setup_value_calls':counter.value_calls,
            'setup_gradient_calls':0,'wall_seconds':result.wall_seconds}
    return {'runs':[record],'solver_queries':record['total_oracle_calls'],'setup_value_calls':counter.value_calls}


def map_cell_task(output,kappa,product,preflight=False):
    output=Path(output);height=product/kappa;prefix=f'bowl__k{kappa}__r{product}';probes=[];runs=[]
    requests=[('zero',0.,0)]+[(axis,eps,sign) for eps in (1e-6,1e-7) for axis in ('x','vx') for sign in (-1,1)]
    for axis,eps,sign in requests:
        x=sign*eps if axis=='x' else 0.;vx=sign*eps if axis=='vx' else 0.
        result=bowl_returns(kappa,height,x=x,vx=vx,bounces=1)
        name=prefix+f'__probe_{axis}_{eps}_{sign}'
        _json(output/'return_traces'/(name+'.json'),result)
        last=result['rows'][-1]
        probes.append({'run_id':name,'case':'parabola','kappa':kappa,'height':height,'product':product,
             'axis':axis,'epsilon':eps,'sign':sign,'input_x':x,'input_vx':vx,
             'return_coordinates':[last['x'],last['vx']],'completed_bounces':len(result['events']),
             'failed':result['failed'],'stop_reason':result['stop_reason'],**result['counts']})
    matrices=[{'cell_id':prefix,'kappa':kappa,'height':height,'product':product,**m} for m in empirical_matrices(probes)]
    if product in ROLLOUT_PRODUCTS or preflight:
        for x in (0.,1e-7,-1e-7):
            n=5 if preflight else 30;result=bowl_returns(kappa,height,x=x,bounces=n)
            name=prefix+f'__rollout_{x}'
            _json(output/'return_traces'/(name+'.json'),result)
            rows=result['rows'];norms=[r['phase_norm'] for r in rows]
            runs.append({'run_id':name,'case':'parabola','kind':'return_rollout','kappa':kappa,
                'height':height,'product':product,'initial_x':x,'completed_bounces':len(result['events']),
                'failed':result['failed'],'stop_reason':result['stop_reason'],
                'final_loss':rows[-1]['loss'],'best_loss':min(r['loss'] for r in rows),
                'max_loss':max(r['loss'] for r in rows),'max_phase_norm':max(norms),
                'final_phase_norm':norms[-1],'max_amplification':max(norms)/abs(x) if x else None,
                'final_amplification':norms[-1]/abs(x) if x else None,**result['counts']})
    return {'runs':runs,'return_probes':probes,'return_matrices':matrices,
            'solver_queries':sum(r['total_oracle_calls'] for r in runs+probes)}


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    for d in ('parents','traces','events','maps','uncommitted','initial_states','return_traces'):(output/d).mkdir()
    preflight=args.stage=='preflight';setup=prepare_rosen(output,preflight) if args.stage!='map' else {}
    manifest=archive(output)
    config={'phase':'E21','stage':args.stage,'cases':['rosenbrock','parabola'] if preflight else ['parabola'] if args.stage=='map' else ['rosenbrock'],
            'rosen_horizon':2. if preflight else 25.,'rosen_query_cap':100000,'return_query_cap':2000,
            'curvatures':[1.] if preflight else list(CURVATURES),'products':[.2,.7] if preflight else list(PRODUCTS),
            'workers':args.workers,'launch_argv':sys.argv,'cwd':str(Path.cwd()),
            'launch_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
            'numeric_thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'source_sha256':manifest,'source_archive_sha256':sha(output/'source.zip'),
            'parent_inputs':setup.get('parent_inputs',[]),'no_theorem_or_online_policy':True}
    _json(output/'config.json',config)
    tables={'runs':[],'return_probes':[],'return_matrices':[]}
    costs={'solver_queries':0,'setup_value_calls':setup.get('setup_value_calls',0),
           'setup_gradient_calls':setup.get('setup_gradient_calls',0)}
    def save():
        for key,records in tables.items():
            _json(output/(key+'.json'),records)
            if records:_csv(output/(key+'.csv'),records)
        _json(output/'execution_counts.json',{**costs,
            'actual_queries_including_setup':sum(costs.values()),
            'rosen_trajectories':sum(r['kind']=='rosen_trajectory' for r in tables['runs']),
            'return_rollouts':sum(r['kind']=='return_rollout' for r in tables['runs']),
            'one_return_executions':len(tables['return_probes']),
            'derived_matrices':len(tables['return_matrices']),
            'executions':len(tables['runs'])+len(tables['return_probes'])})
    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[]
        if args.stage!='map':
            heights=[('original',setup['original_height'])] if preflight else [(str(h),h) for h in (.002,.02,.08,.32)]+[('original',setup['original_height'])]
            for label,height in heights:
                for h in ((.05,) if preflight else (.05,.0125)):
                    for perturb in (('zero','high_plus') if preflight else PERTURBATIONS):
                        futures.append(pool.submit(rosen_task,str(output),'single',None,label,height,h,perturb,config['rosen_horizon']))
            for seed in ((880,) if preflight else (880,881,882)):
                for h in ((.05,) if preflight else (.05,.0125)):
                    for perturb in (('zero','high_plus') if preflight else ('zero','high_plus','high_minus')):
                        futures.append(pool.submit(rosen_task,str(output),'joint',seed,'original',setup['original_height'],h,perturb,config['rosen_horizon']))
        if args.stage!='rosen':
            for kappa in config['curvatures']:
                for product in config['products']:futures.append(pool.submit(map_cell_task,str(output),kappa,product,preflight))
        try:
            for i,f in enumerate(as_completed(futures),1):
                result=f.result()
                for key in tables:tables[key].extend(result.get(key,[]))
                for key in costs:costs[key]+=result.get(key,0)
                save()
                if i%5==0 or i==len(futures):print(f'{args.stage}: {i}/{len(futures)} tasks',flush=True)
        except BaseException:
            _json(output/'task_failure.json',{'traceback':traceback.format_exc(),'active_work_cost':'Unknown if not returned; completed tables preserved'})
            for f in futures:f.cancel()
            raise
    print(json.dumps(json.loads((output/'execution_counts.json').read_text())),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=('preflight','rosen','map'),required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.workers<1:p.error('Positive workers required.')
    run(a)


if __name__=='__main__':main()
