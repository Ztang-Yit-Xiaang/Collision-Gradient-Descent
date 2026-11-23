"""Independent E19 saved-state, cost, physical and baseline update audit."""

import argparse,csv,gzip,hashlib,json,zipfile
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def trace(p):
    with gzip.open(p,'rt') as f:return list(csv.DictReader(f))


def audit(output):
    output=Path(output);cfg=read(output/'config.json');runs=read(output/'runs.json');baselines=read(output/'baselines.json')
    safety=cfg['stage']=='safety';peaks={'energy':0.,'momentum':0.,'heat':0.,'pair':0.,'baseline':0.};checked=events_n=updates=0
    assert hashlib.sha256((output/'source.zip').read_bytes()).hexdigest()==cfg['source_archive_sha256']
    with zipfile.ZipFile(output/'source.zip') as z:
        for name,h in cfg['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    expected=len(cfg['cases'])*len(cfg['seeds'])*(5 if safety else 9);assert len(runs)==expected
    cache={};common=[]
    def physical(name,offset):
        nonlocal checked,events_n
        rows=trace(output/'traces'/(name+'.csv.gz'));events=read(output/'events'/(name+'.json'));maps=read(output/'maps'/(name+'.json'))
        bystep={}
        for e in events:bystep.setdefault(e['step'],[]).append(e)
        pairs=[e for e in events if e['kind']=='pair'];assert sum(m['committed'] for m in maps)==len(pairs)
        prev=None;initial=None;heat=0.
        for t in rows:
            q=np.array(json.loads(t['positions']));v=np.array(json.loads(t['velocities']));k=.5*np.sum(v*v);p=v.sum(axis=0);energy=k+.5*sum(q[:,-1])
            assert abs(energy-float(t['energy']))<1e-10 and abs(k-float(t['kinetic']))<1e-10
            if prev is None:initial=energy
            else:
                step=int(t['step']);base=-np.log(.995)/.05;rate=base+(.5-base)*min((offset+step-.5)*.05/25,1)
                beta=np.exp(-rate*.05);assert abs(beta-float(t['drag_beta']))<1e-15
                delta=(beta**2-1)*prev[0];heat+=delta;err=abs(delta-float(t['damping_energy']));assert err<1e-10;peaks['heat']=max(peaks['heat'],float(err))
                wall=sum((np.array(e['after_i'])-np.array(e['before_i']) for e in bystep.get(step,[]) if e['kind']=='wall'),np.zeros(3))
                err=float(np.linalg.norm(p-beta*prev[1]-wall-np.array([0,0,-.1])));assert err<1e-10;peaks['momentum']=max(peaks['momentum'],err)
            err=abs(energy-initial-heat-float(t['cumulative_impulse_error'])-float(t['cumulative_residual']));assert err<1e-10;peaks['energy']=max(peaks['energy'],float(err))
            assert int(t['total_oracle_calls'])==sum(int(t[k]) for k in ('value_calls','gradient_calls','path_calls'))
            prev=k,p;checked+=1
        for e in pairs:
            a,b,c,d=[np.array(e[k]) for k in ('before_i','before_j','after_i','after_j')]
            err=max(abs(.5*(c@c+d@d-a@a-b@b)),np.linalg.norm(c+d-a-b));assert err<1e-10;peaks['pair']=max(peaks['pair'],float(err));events_n+=1
        cache[name]=rows;return rows,events
    fulls={r['full_id']:r for r in runs if r.get('checkpoint') is None and r.get('trace_id')}
    for full_id,r in fulls.items():
        rows,events=physical(r['trace_id'],0);init=read(output/'initial_states'/(full_id+'.json'))
        assert json.loads(rows[0]['positions'])==[p['position'] for p in init['particles']]
        assert json.loads(rows[0]['velocities'])==[p['velocity'] for p in init['particles']]
        assert abs(float(rows[0]['kinetic'])-.6)<1e-12
        if safety:assert np.linalg.norm(init['momentum'])<1e-12
    checkpoints=read(output/'checkpoints.json')
    for cp in checkpoints:
        name=cp['source_full_id'];full=cache[name+'__full'];t=full[cp['step']]
        assert int(t['total_oracle_calls'])==cp['calls']==cp['feature_last_observation_calls']
        assert json.loads(t['positions'])==[p['position'] for p in cp['particles']]
        assert json.loads(t['velocities'])==[p['velocity'] for p in cp['particles']]
        assert cp['history_step_range']==[max(1,cp['step']-19),cp['step']]
        init=read(output/'initial_states'/(name+'.json'));scale=max(1,abs(min(init['values'])))
        assert abs(cp['features']['current_loss']-min(json.loads(t['values']))/scale)<1e-12
        start=full[cp['history_step_range'][0]]
        assert abs(cp['features']['recent_loss_improvement']-(min(json.loads(start['values']))-min(json.loads(t['values'])))/scale)<1e-12
        assert set(cp['features'])=={'budget_fraction','current_loss','recent_loss_improvement','kinetic','clearance_potential','center_speed','relative_speed_rms','position_dispersion','recent_pair_count','mean_abs_kinetic_transfer'}
    for r in runs:
        if r.get('trace_id') is None:assert r['failed'];continue
        full=cache[r['full_id']+'__full'];init=read(output/'initial_states'/(r['full_id']+'.json'))
        if r.get('checkpoint'):
            cp=read(output/'checkpoints'/(r['checkpoint']+'.json'));suffix,events=physical(r['trace_id'],cp['step'])
            assert not any(e['kind']=='pair' for e in events)
            assert json.loads(suffix[0]['positions'])==json.loads(full[cp['step']]['positions'])
            assert json.loads(suffix[0]['velocities'])==json.loads(full[cp['step']]['velocities'])
            assert int(suffix[0]['total_oracle_calls'])==cp['calls']+4
            assert r['total_oracle_calls']-cp['calls']==r['actual_suffix_calls']
            rows=full[:cp['step']+1]+suffix
            j=min(len(suffix)-1,len(full)-cp['step']-1);a,b=full[cp['step']+j],suffix[j]
            scale=max(min(init['values'])-init['minimum'],1e-12)
            common.append({'case':r['case'],'seed':r['seed'],'rule':r['rule'],'fraction':r['fraction'],
                'physical_time':(cp['step']+j)*.05,'off_minus_on_gap':(min(json.loads(b['values']))-min(json.loads(a['values'])))/scale if not safety else None,
                'on_calls':int(a['total_oracle_calls']),'off_calls':int(b['total_oracle_calls'])})
        else:rows=full
        assert r['total_oracle_calls']<=cfg['oracle_budget'] and int(rows[-1]['total_oracle_calls'])<=r['total_oracle_calls']
        values=[json.loads(t['values']) for t in rows]
        assert r['final_value']==min(values[-1]) and r['best_value']==min(min(v) for v in values)
        if safety:
            rise=max(v[0] for v in values)-init['values'][0]
            assert abs(rise-r['protected_max_rise'])<1e-12 and r['protected_harmed']==(rise>1e-4)
        else:
            flags=[any(v<=init['target'] and (init['boundary'] is None or p[0]<init['boundary']) for v,p in zip(vs,json.loads(t['positions']))) for vs,t in zip(values,rows)]
            assert flags[-1]==r['final_target_hit'] and any(flags)==r['historical_hit']
    for r in baselines:
        rows=trace(output/'baseline_traces'/(r['trace_id']+'.csv.gz'))
        assert int(rows[0]['total_oracle_calls'])==4
        for prev,t in zip(rows,rows[1:]):
            before=np.array(json.loads(t['before']));g=np.array(json.loads(t['gradient']));buffer=np.array(json.loads(t['buffer_before']))
            direction=g if r['family']=='gd' else .9*buffer+g;after=np.array(json.loads(t['positions']))[int(t['particle'])]
            err=float(np.linalg.norm(after-(before-r['eta']*direction)));assert err<1e-12;peaks['baseline']=max(peaks['baseline'],err)
            assert int(t['total_oracle_calls'])==int(prev['total_oracle_calls'])+2;updates+=1
        assert r['final_value']==min(json.loads(rows[-1]['values'])) and r['total_oracle_calls']<=cfg['oracle_budget']
    counts=read(output/'execution_counts.json');actual=sum(r['total_oracle_calls'] for r in fulls.values())+sum(r['actual_suffix_calls'] for r in runs)+sum(r['total_oracle_calls'] for r in baselines)
    assert actual==counts['actual_query_calls']
    result={'passed':True,'policy_views':len(runs),'baselines':len(baselines),'failures_retained':sum(r['failed'] for r in runs+baselines),
        'physical_rows_checked':checked,'pair_events_checked':events_n,'baseline_updates_checked':updates,'max_errors':peaks,'actual_query_calls':actual}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');(output/'common_time.json').write_text(json.dumps(common,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');print(json.dumps(audit(p.parse_args().output),indent=2))
