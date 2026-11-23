"""Independent E21 saved-state, external-impulse and finite-probe checks."""
import argparse,csv,gzip,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from collision_optim.population_experiment import make_objective


def read(p):return json.loads(p.read_text())


def audit(output):
    p=Path(output);cfg=read(p/'config.json');runs=read(p/'runs.json');probes=read(p/'return_probes.json');matrices=read(p/'return_matrices.json')
    counts=read(p/'execution_counts.json');stage=cfg['stage']
    expected={'preflight':(4,6,18,4),'rosen':(58,0,0,0),'map':(0,36,270,60)}[stage]
    rr=[r for r in runs if r['kind']=='rosen_trajectory'];br=[r for r in runs if r['kind']=='return_rollout']
    assert (len(rr),len(br),len(probes),len(matrices))==expected,'Incomplete batch'
    assert hashlib.sha256((p/'source.zip').read_bytes()).hexdigest()==cfg['source_archive_sha256']
    with zipfile.ZipFile(p/'source.zip') as z:
        for name,digest in cfg['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    peaks=dict(energy=0.,momentum=0.,reflection=0.,flight=0.,matrix=0.);physical_rows=return_events=0;validation_values=validation_gradients=0
    if rr:
        setup=read(p/'setup.json');basis=setup['basis'];obj,_,_=make_objective('rosenbrock');opt=np.array(basis['optimum'])
        for probe in basis['probes']:
            d=np.eye(2)[probe['axis']]*probe['delta']
            for sign,key in ((1,'plus_gradient'),(-1,'minus_gradient')):
                assert np.max(abs(obj.gradient(opt+sign*d)-probe[key]))<1e-12;validation_gradients+=1
        for parent in setup['parent_inputs']:
            assert hashlib.sha256((p/parent['preserved_copy']).read_bytes()).hexdigest()==parent['sha256']
    for record in rr:
        initial=read(p/'initial_states'/(record['run_id']+'.json'))
        with gzip.open(p/'traces'/(record['run_id']+'.csv.gz'),'rt') as f:rows=list(csv.DictReader(f))
        events=read(p/'events'/(record['run_id']+'.json'));bystep={}
        for e in events:assert e['kind']=='wall';bystep.setdefault(e['step'],[]).append(e)
        previous=None;heat=impact=0.;e0=None;rise=[];dist=[]
        for row in rows:
            q=np.array(json.loads(row['positions']));v=np.array(json.loads(row['velocities']));vals=json.loads(row['values'])
            k=.5*np.sum(v*v);momentum=v.sum(axis=0);energy=k+.5*q[:,-1].sum()
            true=[obj.value(x[:-1]) for x in q];validation_values+=len(q)
            assert np.max(abs(np.array(true)-vals))<1e-10 and min(q[:,-1]-true)>-1e-8
            assert abs(energy-float(row['energy']))<1e-10
            if previous is None:
                e0=energy;assert q.tolist()==initial['positions'] and v.tolist()==initial['velocities']
                assert abs(q[0,-1]-vals[0]-record['height'])<1e-12
                if record['layout']=='joint':
                    parent=read(p/'parents'/f'cloud_{record["spectator_seed"]}.json')
                    assert q[1:].tolist()==parent['positions'][1:] and v[1:].tolist()==parent['velocities'][1:]
            else:
                h=record['h'];step=int(row['step']);base=-np.log(.995)/.05
                rate=base+(.5-base)*min((step-.5)*h/25,1);beta=np.exp(-rate*h)
                assert abs(float(row['drag_beta'])-beta)<1e-14
                delta=(beta*beta-1)*previous[0];assert abs(delta-float(row['damping_energy']))<1e-9;heat+=delta
                wall=sum((np.array(e['after_i'])-e['before_i'] for e in bystep.get(step,[])),np.zeros(3))
                gravity=np.array([0.,0.,-len(q)*.5*h]);error=float(np.linalg.norm(momentum-beta*previous[1]-wall-gravity))
                peaks['momentum']=max(peaks['momentum'],error);assert error<1e-9
                impact+=float(row['impulse_energy'])
            error=abs(energy-e0-heat-impact);peaks['energy']=max(peaks['energy'],error);assert error<1e-8
            assert sum(int(row[k]) for k in ('value_calls','gradient_calls','path_calls'))==int(row['total_oracle_calls'])
            rise.append(vals[0]-initial['reference_loss']);dist.append(float(np.linalg.norm(q[0,:-1]-initial['optimum'])))
            previous=k,momentum;physical_rows+=1
        assert abs(max(0.,max(rise))-record['max_loss_rise'])<1e-12
        assert abs(max(dist)-record['max_displacement'])<1e-12
        assert record['total_oracle_calls']>=int(rows[-1]['total_oracle_calls'])
        assert abs(float(rows[-1]['time'])-cfg['rosen_horizon'])<1e-12 or record['stop_reason']!='completed'
    for record in br+probes:
        raw=read(p/'return_traces'/(record['run_id']+'.json'));rows=raw['rows'];kappa=record['kappa'];height=record['height']
        for i,row in enumerate(rows):
            velocity=np.array([row['vx'],row['vz']]);energy=.5*(velocity@velocity)+.5*row['z']
            assert abs(.5*kappa*row['x']**2-row['loss'])<1e-10
            error=abs(energy-.5*height);peaks['energy']=max(peaks['energy'],error);assert error<1e-8
            assert sum(row[k] for k in ('value_calls','gradient_calls','path_calls'))==row['total_oracle_calls']
            if i:
                prev=rows[i-1];event=raw['events'][i-1];dt=event['duration']
                incoming=np.array([prev['vx'],prev['vz']-.5*dt]);point=np.array([prev['x']+dt*prev['vx'],prev['z']+dt*prev['vz']-.25*dt*dt])
                error=float(np.max(np.abs(point-event['point'])));peaks['flight']=max(peaks['flight'],error);assert error<1e-9
                normal=np.array([-kappa*row['x'],1.]);normal/=np.linalg.norm(normal)
                after=incoming-2*(incoming@normal)*normal
                error=float(np.max(abs(after-velocity)));peaks['reflection']=max(peaks['reflection'],error);assert error<1e-8
                wall=np.array(event['wall_impulse']);gravity=np.array([0.,-.5*dt]);before=np.array([prev['vx'],prev['vz']])
                error=float(np.linalg.norm(velocity-before-wall-gravity));peaks['momentum']=max(peaks['momentum'],error);assert error<1e-9
                assert abs(row['time']-prev['time']-dt)<1e-10
                return_events+=1
            physical_rows+=1
        assert raw['counts']['total_oracle_calls']==record['total_oracle_calls']
        if 'return_coordinates' in record:assert record['return_coordinates']==[rows[-1]['x'],rows[-1]['vx']]
    for m in matrices:
        if not m['available']:continue
        cell=[r for r in probes if r['kappa']==m['kappa'] and r['product']==m['product'] and r['epsilon']==m['epsilon']]
        columns=[]
        for axis in ('x','vx'):
            plus=next(r for r in cell if r['axis']==axis and r['sign']==1)
            minus=next(r for r in cell if r['axis']==axis and r['sign']==-1)
            columns.append((np.array(plus['return_coordinates'])-minus['return_coordinates'])/(2*m['epsilon']))
        matrix=np.column_stack(columns);error=float(np.max(abs(matrix-m['matrix'])));peaks['matrix']=max(peaks['matrix'],error);assert error<1e-12
        assert abs(max(abs(np.linalg.eigvals(matrix)))-m['spectral_radius'])<1e-10
    assert counts['solver_queries']==sum(r['total_oracle_calls'] for r in runs+probes)
    assert counts['executions']==len(runs)+len(probes)
    result={'passed':True,'trajectories_checked':len(runs),'one_return_executions_checked':len(probes),
            'derived_matrices_checked':len(matrices),'physical_rows_checked':physical_rows,'return_events_checked':return_events,
            'failures_retained':sum(r['failed'] for r in runs+probes),'max_errors':peaks,
            'solver_queries':counts['solver_queries'],'posthoc_value_calls':validation_values,'posthoc_gradient_calls':validation_gradients}
    (p/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');audit(p.parse_args().output)
