"""Independent E22 allocation, paid probes, physics and raw metric audit."""
import argparse,csv,gzip,hashlib,json,zipfile
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())


def raw_metrics(values,positions,times,c):
    rotation=np.array(c['rotation']);coordinates=positions[:,:,:2]@rotation.T
    left=coordinates[:,:,0]<c['boundary'];hits=(values<=c['target'])&left
    p0=values[:,0];initial=np.array(c['values']);minimum=c['minimum']
    return {'final_value':float(values[-1].min()),'best_value':float(values.min()),
       'final_hit':bool(hits[-1].any()),'historical_hit':bool(hits.any()),'ever_left':bool(left.any()),
       'first_left_time':float(times[np.flatnonzero(left.any(axis=1))[0]]) if left.any() else None,
       'relative_final_gap':float((values[-1].min()-minimum)/(initial.min()-minimum)) if c['condition']=='bad_only' else None,
       'particle0_final_value':float(p0[-1]),'particle0_final_rise':float(p0[-1]-initial[0]),
       'particle0_max_rise':max(0.,float(p0.max()-initial[0])),'particle0_harmed':bool(p0.max()-initial[0]>1e-4),
       'particle0_max_above_global':float(p0.max()-minimum),
       'particle0_final_distance':float(np.linalg.norm(positions[-1,0,:-1]-c['reference_point0']))}


def audit(output):
    p=Path(output);cfg=read(p/'config.json');contexts=read(p/'contexts.json');runs=read(p/'runs.json');counts=read(p/'execution_counts.json')
    expected=len(cfg['seeds'])*len(cfg['conditions'])*len(cfg['stiffnesses'])
    assert len(contexts)==expected and len(runs)==5*expected,'Batch still incomplete'
    assert hashlib.sha256((p/'source.zip').read_bytes()).hexdigest()==cfg['source_archive_sha256']
    with zipfile.ZipFile(p/'source.zip') as z:
        for name,digest in cfg['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    peaks=dict(energy=0.,momentum=0.,probe=0.,allocation=0.,equal_initial_energy=0.,metrics=0.,reflection=0.)
    rows_checked=walls=probes_checked=0;common=[];validation_values=validation_gradients=0
    for c in contexts:
        Q=np.array(c['rotation']);k=c['stiffness'];x=np.array(c['points']);coordinates=x@Q.T
        value=lambda r:.25*(r[0]**2-1)**2+.05*r[0]+.5*k*r[1]**2
        gradient=lambda point:Q.T@np.array([(Q@point)[0]**3-(Q@point)[0]+.05,k*(Q@point)[1]])
        vals=np.array([value(r) for r in coordinates]);validation_values+=4
        assert np.max(abs(vals-c['values']))<1e-12
        for i,record in enumerate(c['curvature_probes']):
            columns=[]
            for probe in record['probes']:
                displacement=np.eye(2)[probe['axis']]*probe['step']
                a=gradient(x[i]+displacement);b=gradient(x[i]-displacement);validation_gradients+=2
                err=max(float(np.max(abs(a-probe['plus_gradient']))),float(np.max(abs(b-probe['minus_gradient']))))
                peaks['probe']=max(peaks['probe'],err);assert err<1e-9
                columns.append((np.array(probe['plus_gradient'])-probe['minus_gradient'])/(2*probe['step']));probes_checked+=2
            matrix=np.column_stack(columns);matrix=(matrix+matrix.T)/2
            assert np.max(abs(matrix-record['hessian']))<1e-12
            assert abs(max(np.linalg.eigvalsh(matrix))-c['curvatures'][i])<1e-12
            assert abs(max(3*coordinates[i,0]**2-1,k)-c['curvatures'][i])<1e-6
        assert c['actual_shared_probe_calls']==20 and c['probe_counts']=={'value_calls':4,'gradient_calls':16,'path_calls':0}
        group=[r for r in runs if r['context_id']==c['context_id']];cache={};initial_energies={}
        for rec in group:
            policy=rec['policy'];caps=np.minimum(.8,.2/np.maximum(c['curvatures'],1e-6));pivot=None
            if policy=='fixed_high':H=np.full(4,.8)
            elif policy=='fixed_low':H=np.full(4,.02)
            elif policy=='cap_all':H=caps.copy()
            else:
                pivot=int(np.argmin(vals)) if policy=='reserve_best' else int(np.random.default_rng(np.random.SeedSequence([20261109,c['seed']])).integers(0,4))
                H=np.full(4,.8+(.8-caps[pivot])/3);H[pivot]=caps[pivot]
            err=float(np.max(abs(H-rec['heights'])));peaks['allocation']=max(peaks['allocation'],err);assert err<1e-12
            assert rec['selected_index']==pivot
            prefix=4 if policy.startswith('fixed_') else 20
            assert rec['charged_prefix_calls']==prefix and rec['actual_branch_calls']==rec['total_oracle_calls']-prefix
            initial=read(p/'initial_states'/(rec['run_id']+'.json'))
            with gzip.open(p/'traces'/(rec['run_id']+'.csv.gz'),'rt') as f:rows=list(csv.DictReader(f))
            events=read(p/'events'/(rec['run_id']+'.json'));by_step={}
            for e in events:
                assert e['kind']=='wall';by_step.setdefault(e['step'],[]).append(e)
                point=np.array(e['point_i']);normal=np.r_[-gradient(point[:2]),1.];normal/=np.linalg.norm(normal)
                before=np.array(e['before_i']);after=np.array(e['after_i'])
                err=max(float(np.max(abs(normal-e['normal']))),float(np.max(abs(after-(before-2*(before@normal)*normal)))))
                peaks['reflection']=max(peaks['reflection'],err);assert err<1e-9
                assert abs(point[-1]-value(Q@point[:2]))<1e-8
                validation_gradients+=1;validation_values+=1
            previous=None;e0=None;heat=impact=0.;allq=[];allv=[];allvalues=[];alltimes=[];allcalls=[]
            for row in rows:
                q=np.array(json.loads(row['positions']));v=np.array(json.loads(row['velocities']));f=np.array(json.loads(row['values']))
                true=np.array([value(r) for r in q[:,:2]@Q.T]);validation_values+=4
                assert np.max(abs(true-f))<1e-9 and min(q[:,-1]-true)>-1e-8
                K=.5*np.sum(v*v);P=v.sum(axis=0);E=K+.5*q[:,-1].sum()
                assert abs(E-float(row['energy']))<1e-9
                if previous is None:
                    e0=E;initial_energies[policy]=E
                    assert q.tolist()==initial['positions'] and v.tolist()==initial['velocities']
                    assert np.max(abs(q[:,:2]-x))<1e-14 and np.max(abs(q[:,-1]-vals-H))<1e-12
                    expected_velocities=np.array(c.get('initial_horizontal_velocities',np.zeros((4,2))))
                    assert np.max(abs(v[:,:2]-expected_velocities))<1e-14 and np.max(abs(v[:,-1]))==0.
                    assert np.max(abs(P))<1e-12 and abs(K-c.get('initial_kinetic',0.))<1e-12
                    assert int(row['total_oracle_calls'])==prefix+4
                else:
                    step=int(row['step']);h=.05;base=-np.log(.995)/.05;rate=base+(.5-base)*min((step-.5)*h/25,1);beta=np.exp(-rate*h)
                    assert abs(beta-float(row['drag_beta']))<1e-14
                    loss=(beta*beta-1)*previous[0];assert abs(loss-float(row['damping_energy']))<1e-9;heat+=loss
                    wall=sum((np.array(e['after_i'])-e['before_i'] for e in by_step.get(step,[])),np.zeros(3))
                    err=float(np.linalg.norm(P-beta*previous[1]-wall-np.array([0.,0.,-.1])));peaks['momentum']=max(peaks['momentum'],err);assert err<1e-9
                    impact+=float(row['impulse_energy'])
                err=abs(E-e0-heat-impact);peaks['energy']=max(peaks['energy'],err);assert err<1e-8
                calls=sum(int(row[z]) for z in ('value_calls','gradient_calls','path_calls'));assert calls==int(row['total_oracle_calls'])
                allq.append(q);allv.append(v);allvalues.append(f);alltimes.append(float(row['time']));allcalls.append(calls)
                previous=K,P;rows_checked+=1
            walls+=len(events)
            assert rec['total_oracle_calls']>=allcalls[-1] and rec['total_oracle_calls']<=cfg['query_cap']
            if cfg['stage'] in ('formal','kinetic') and not rec['failed']:assert rec['total_oracle_calls']==6000
            q=np.array(allq);values=np.array(allvalues);times=np.array(alltimes);metrics=raw_metrics(values,q,times,c)
            for key,value_ in metrics.items():
                if isinstance(value_,bool) or value_ is None:assert rec[key]==value_
                else:
                    err=abs(rec[key]-value_);peaks['metrics']=max(peaks['metrics'],err);assert err<1e-8
            cache[policy]={'q':q,'values':values,'times':times,'calls':allcalls,'record':rec}
        for policy in ('reserve_best','reserve_random'):
            err=abs(initial_energies[policy]-initial_energies['fixed_high']);peaks['equal_initial_energy']=max(peaks['equal_initial_energy'],err);assert err<1e-12
        last=min(len(raw['times']) for raw in cache.values())
        for policy,raw in cache.items():
            metrics=raw_metrics(raw['values'][:last],raw['q'][:last],raw['times'][:last],c)
            common.append({'context_id':c['context_id'],'condition':c['condition'],'stiffness':k,'seed':c['seed'],'policy':policy,
                'physical_time':float(raw['times'][last-1]),'charged_calls_at_view':raw['calls'][last-1],
                'source_run_id':raw['record']['run_id'],'failed':raw['record']['failed'],**metrics})
    assert counts['shared_probe_calls']==20*len(contexts)
    assert counts['actual_branch_calls']==sum(r['actual_branch_calls'] for r in runs)
    assert counts['charged_policy_queries']==sum(r['total_oracle_calls'] for r in runs)
    assert counts['actual_solver_queries']==counts['shared_probe_calls']+counts['actual_branch_calls']
    assert counts['setup_reference_values']==3*len(contexts)
    (p/'common_time.json').write_text(json.dumps(common,indent=2)+'\n')
    result={'passed':True,'contexts_checked':len(contexts),'trajectories_checked':len(runs),'physical_rows_checked':rows_checked,
            'wall_events_checked':walls,'curvature_gradient_probes_checked':probes_checked,'max_errors':peaks,
            'failures_retained':sum(r['failed'] for r in runs),'actual_solver_queries':counts['actual_solver_queries'],
            'posthoc_value_calls':validation_values,'posthoc_gradient_calls':validation_gradients,'common_time_views_are_derived':True}
    (p/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');audit(p.parse_args().output)
