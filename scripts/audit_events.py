"""Independent raw-state E20 physics, contact reconstruction and cost audit."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from collision_optim.population_experiment import make_objective


def read(path):
    return json.loads(path.read_text())


def trace(path):
    with gzip.open(path, 'rt') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('positions','velocities','values'):
            row[key] = json.loads(row[key])
    return rows


def audit(output):
    output = Path(output); cfg = read(output/'config.json')
    assert hashlib.sha256((output/'source.zip').read_bytes()).hexdigest()==cfg['source_archive_sha256']
    with zipfile.ZipFile(output/'source.zip') as z:
        for name, digest in cfg['source_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
    sources = read(output/'sources.json'); runs = read(output/'runs.json')
    contexts = read(output/'contexts.json'); sensitivity = read(output/'sensitivity_runs.json')
    if cfg['stage']=='sensitivity':
        assert len(sensitivity)==len(cfg['seeds'])*6, 'Batch still incomplete'
    else:
        assert len(sources)==len(cfg['seeds'])*len(cfg['cases']), 'Batch still incomplete'
        assert len(runs)==6*len(contexts), 'Incomplete paired replay set'
    peaks = dict(energy=0., momentum=0., heat=0., pair=0., reconstruction=0., feature=0.)
    checked = walls = pair_events = audit_values = audit_gradients = 0; cache = {}; event_cache = {}
    def physical(record, h=.05, replay=False):
        nonlocal checked, walls, pair_events, audit_values
        name = record['run_id']; rows = trace(output/'traces'/(name+'.csv.gz'))
        events = read(output/'events'/(name+'.json')); bystep = {}
        for event in events:
            bystep.setdefault(event['step'], []).append(event)
        obj, _, _ = make_objective(record['case']); previous = None; initial = None; heat = 0.; impact_error = 0.
        for row in rows:
            q = np.array(row['positions']); v = np.array(row['velocities'])
            k = .5*np.sum(v*v); p = v.sum(axis=0); energy = k+.5*q[:,-1].sum()
            assert abs(energy-float(row['energy']))<1e-9
            true_values = [obj.value(x[:-1]) for x in q]; audit_values += len(q)
            assert np.max(np.abs(np.array(true_values)-row['values']))<1e-9
            assert min(q[:,-1]-true_values)>-1e-8
            if previous is None:
                initial = energy
            else:
                step = int(row['step']); duration = float(row['duration']) if replay else h
                clock = float(row['absolute_start_time']) if replay else (step-1)*h
                partial = replay and row['partial_source_step']=='True'
                rate = 0. if partial else -np.log(.995)/.05+(.5+np.log(.995)/.05)*min((clock+duration/2)/25,1)
                beta = 1. if partial else np.exp(-rate*duration)
                assert abs(beta-float(row['drag_beta']))<1e-13
                expected_heat = (beta*beta-1)*previous[0]
                err = abs(expected_heat-float(row['damping_energy'])); peaks['heat']=max(peaks['heat'],err)
                assert err<1e-9
                wall = sum((np.array(e['after_i'])-e['before_i'] for e in bystep.get(step,[]) if e['kind']=='wall'),np.zeros(v.shape[1]))
                gravity = np.zeros(v.shape[1]); gravity[-1] = -len(v)*.5*duration
                err = float(np.linalg.norm(p-beta*previous[1]-wall-gravity))
                peaks['momentum']=max(peaks['momentum'],err); assert err<1e-9
                heat += expected_heat; impact_error += float(row['impulse_energy'])
            err = abs(energy-initial-heat-impact_error); peaks['energy']=max(peaks['energy'],err); assert err<1e-8
            total = sum(int(row[k]) for k in ('value_calls','gradient_calls','path_calls'))
            assert total==int(row['total_oracle_calls'])
            if replay:
                assert int(row['actual_branch_calls'])==total-6
            previous = k,p; checked += 1
        for e in events:
            if e['kind']=='wall':
                walls += 1
            else:
                a,b,c,d = (np.array(e[k]) for k in ('before_i','before_j','after_i','after_j'))
                err = max(abs(.5*(c@c+d@d-a@a-b@b)),float(np.linalg.norm(c+d-a-b)))
                peaks['pair']=max(peaks['pair'],err); assert err<1e-10; pair_events += 1
        assert record['total_oracle_calls']>=int(rows[-1]['total_oracle_calls'])
        cache[name]=rows; event_cache[name]=events
        return rows
    for source in sources:
        physical(source)
    for record in sensitivity:
        rows = physical(record,record['h']); initial = read(output/'initial_states'/(record['run_id']+'.json'))
        assert rows[0]['positions']==initial['positions'] and rows[0]['velocities']==initial['velocities']
        rise = [r['values'][0]-initial['values'][0] for r in rows]
        assert abs(max(0.,max(rise))-record['protected_max_rise'])<1e-12
        assert abs(rise[-1]-record['protected_final_rise'])<1e-12
        assert abs(float(rows[-1]['time'])-cfg.get('sensitivity_time',10.))<1e-12 or record['stop_reason']!='completed'
        assert not any(e['kind']=='pair' for e in event_cache[record['run_id']])
    for context in contexts:
        rows = cache[context['source_run']]; events = event_cache[context['source_run']]
        step = context['source_step']; q = np.array(rows[step-1]['positions'],float)
        v = np.array(rows[step-1]['velocities'],float)*float(rows[step]['drag_beta']); clock=(step-1)*.05
        local_error = 0.; found=False
        for index,e in enumerate(events):
            if e['step']!=step:
                continue
            dt=max(0.,e['time']-clock); q += v*dt; q[:,-1] -= .25*dt*dt; v[:,-1] -= .5*dt; clock=e['time']
            for suffix, particle in [('i',e['i']),('j',e.get('j'))]:
                if particle is not None:
                    local_error=max(local_error,float(np.max(np.abs(q[particle]-e['point_'+suffix]))),float(np.max(np.abs(v[particle]-e['before_'+suffix]))))
                    q[particle]=e['point_'+suffix];v[particle]=e['before_'+suffix]
            if index==context['source_event_index']:
                local_error=max(local_error,float(np.max(np.abs(q-context['incoming']['positions']))),float(np.max(np.abs(v-context['incoming']['velocities']))));found=True
            for suffix,particle in [('i',e['i']),('j',e.get('j'))]:
                if particle is not None:v[particle]=e['after_'+suffix]
        dt=max(0.,step*.05-clock);q+=v*dt;q[:,-1]-=.25*dt*dt;v[:,-1]-=.5*dt
        local_error=max(local_error,float(np.max(np.abs(q-rows[step]['positions']))),float(np.max(np.abs(v-rows[step]['velocities']))))
        peaks['reconstruction']=max(peaks['reconstruction'],local_error);assert found and local_error<1e-9
        obj,_,_=make_objective(context['case']);q=np.array(context['incoming']['positions']);v=np.array(context['incoming']['velocities'])
        vals=[obj.value(x[:-1]) for x in q];audit_values+=len(q)
        i,j=context['i'],context['j'];n=np.array(context['normal']);a,b=v[i],v[j]
        gradients=[obj.gradient(q[k,:-1]) for k in (i,j)];audit_gradients+=2
        gi,gj=[np.r_[g,0.] for g in gradients];impulse=-((a-b)@n)*n
        def cos(a,b):
            d=np.linalg.norm(a)*np.linalg.norm(b)
            return float(a@b/d) if d>1e-12 else 0.
        f={'source_time':context['source_time'],'pair_loss':min(vals[i],vals[j])/max(1,abs(context['initial_best'])),
           'pair_kinetic':.5*(a@a+b@b),'relative_speed':np.linalg.norm(a-b),'center_speed':np.linalg.norm((a+b)/2),
           'impulse_norm':np.linalg.norm(impulse),'abs_energy_transfer':abs(.5*((a+impulse)@(a+impulse)-a@a)),
           'gradient_norm_i':np.linalg.norm(gi),'gradient_norm_j':np.linalg.norm(gj),
           'gradient_zero_i':float(np.linalg.norm(gi)<=1e-12),'gradient_zero_j':float(np.linalg.norm(gj)<=1e-12),
           'gradient_impulse_cos_i':cos(gi,impulse),'gradient_impulse_cos_j':cos(gj,-impulse),
           'velocity_impulse_cos_i':cos(a,impulse),'velocity_impulse_cos_j':cos(b,-impulse),
           'normal_gradient_cos_i':cos(n,gi),'normal_gradient_cos_j':cos(-n,gj)}
        assert set(f)==set(context['features'])
        error=max(abs(float(f[k])-context['features'][k]) for k in f)
        peaks['feature']=max(peaks['feature'],error);assert error<1e-9
        assert context['feature_probe_calls']==6
    ctx={c['context_id']:c for c in contexts}
    for record in runs:
        rows=physical(record,replay=True);c=ctx[record['context_id']]
        mapping=read(output/'maps'/(record['run_id']+'.json'))
        assert mapping['before']==c['incoming']
        assert rows[0]['positions']==mapping['after']['positions'] and rows[0]['velocities']==mapping['after']['velocities']
        a=np.array(mapping['before']['velocities']);b=np.array(mapping['after']['velocities'])
        assert abs(.5*np.sum(b*b-a*a))<1e-10 and np.linalg.norm(b.sum(axis=0)-a.sum(axis=0))<1e-10
        assert not any(e['kind']=='pair' for e in event_cache[record['run_id']])
        for key in ('end','budget1000','boundary10','boundary50','boundary100'):
            view=record['views'][key]
            if view is None:continue
            points=[r for r in rows if int(r['total_oracle_calls'])<=1000] if key=='budget1000' else rows[:view['step']+1]
            pair=[min(r['values'][c['i']],r['values'][c['j']]) for r in points]
            assert abs(view['pair_final_loss']-pair[-1])<1e-12
            assert abs(view['pair_best_loss']-min(pair))<1e-12
        assert record['actual_branch_calls']==record['total_oracle_calls']-6
    counts=read(output/'execution_counts.json')
    assert counts['actual_source_calls']==sum(r['total_oracle_calls'] for r in sources)
    assert counts['actual_probe_calls']==6*len(contexts)
    assert counts['actual_replay_calls']==sum(r['actual_branch_calls'] for r in runs)
    assert counts['actual_sensitivity_calls']==sum(r['total_oracle_calls'] for r in sensitivity)
    assert counts['actual_solver_queries']==sum(counts[k] for k in ('actual_source_calls','actual_probe_calls','actual_replay_calls','actual_sensitivity_calls'))
    result={'passed':True,'physical_rows_checked':checked,'source_pair_events_checked':pair_events,
            'wall_events_checked':walls,'contexts_checked':len(contexts),'replays_checked':len(runs),
            'sensitivity_checked':len(sensitivity),'max_errors':peaks,
            'failures_retained':sum(r['failed'] for r in sources+runs+sensitivity),
            'unavailable_contexts':len(read(output/'unavailable.json')),
            'actual_solver_queries':counts['actual_solver_queries'],
            'posthoc_validation_value_calls':audit_values,'posthoc_validation_gradient_calls':audit_gradients}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');audit(p.parse_args().output)
