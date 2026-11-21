"""Raw E18 accounting, state continuity, GD commits and common-time comparisons."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def load(path):return json.loads(path.read_text())
def trace(path):
    with gzip.open(path,"rt") as f:rows=list(csv.DictReader(f))
    # Early preflight stored the three counters but not their redundant sum.
    for row in rows:
        if 'total_oracle_calls' not in row:
            row['total_oracle_calls']=str(sum(int(row[k]) for k in ('value_calls','gradient_calls','path_calls')))
    return rows


def score(rows,minimum,f0,target,boundary,gd=False):
    points=[([float(r['value'])],[json.loads(r['x'])]) if gd else
            (json.loads(r['values']),json.loads(r['positions'])) for r in rows]
    hits=[any(v<=target and (boundary is None or p[0]<boundary) for v,p in zip(vs,ps)) for vs,ps in points]
    best=min(min(vs) for vs,ps in points);final=min(points[-1][0]);scale=max(f0-minimum,1e-12)
    return {'best':best,'final':final,'best_gap':max(0.,(best-minimum)/scale),
            'final_gap':max(0.,(final-minimum)/scale),'ever':any(hits),'hit':hits[-1]}


def audit(output):
    output=Path(output);cfg=load(output/'config.json');runs=load(output/'runs.json')
    prefixes=load(output/'prefixes.json');replicas=load(output/'replication_runs.json')
    n=len(cfg['seeds'])*len(cfg['cases'])
    assert (len(prefixes),len(runs),len(replicas))==(3*n,9*n,3*n)
    assert len({(r['case'],r['seed'],r['prefix_method'],r['suffix']) for r in runs})==len(runs)
    assert hashlib.sha256((output/'source.zip').read_bytes()).hexdigest()==cfg['source_archive_sha256']
    with zipfile.ZipFile(output/'source.zip') as archive:
        for name,h in cfg['source_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==h
    peaks={'heat':0.,'energy':0.,'momentum':0.,'pair':0.,'gd_update':0.};physical_rows=pair_events=gd_updates=0

    def physical(rows,events,attempts,offset,schedule):
        nonlocal physical_rows,pair_events
        bystep={}
        for e in events:bystep.setdefault(e['step'],[]).append(e)
        pairs=[e for e in events if e['kind']=='pair']
        assert len(pairs)==sum(a['committed'] for a in attempts)
        previous=None;h0=None;qtotal=0.
        for t in rows:
            p=np.array(json.loads(t['positions']));v=np.array(json.loads(t['velocities']))
            k=.5*np.sum(v*v);momentum=v.sum(axis=0);h=k+.5*np.sum(p[:,-1])
            assert abs(k-float(t['kinetic']))<1e-10 and abs(h-float(t['energy']))<1e-10
            if previous is None:h0=h
            else:
                index=int(t['step']);dt=.05;base=-np.log(.995)/dt
                rate=base if schedule=='weak' else (base+.5)/2 if schedule=='constant' else base+(.5-base)*min((offset+index-.5)*dt/25,1)
                beta=.995 if schedule=='weak' else np.exp(-rate*dt)
                assert abs(beta-float(t['drag_beta']))<1e-15
                heat=(beta*beta-1)*previous[0];qtotal+=heat
                e=abs(heat-float(t['damping_energy']));peaks['heat']=max(peaks['heat'],float(e));assert e<1e-10
                wall=sum((np.array(e['after_i'])-np.array(e['before_i']) for e in bystep.get(index,[]) if e['kind']=='wall'),np.zeros(3))
                e=np.linalg.norm(momentum-beta*previous[1]-wall-np.array([0,0,-.1]))
                peaks['momentum']=max(peaks['momentum'],float(e));assert e<1e-10
            e=abs(h-h0-qtotal-float(t['cumulative_impulse_error'])-float(t['cumulative_residual']))
            peaks['energy']=max(peaks['energy'],float(e));assert e<1e-10
            previous=(k,momentum);physical_rows+=1
        for i,e in enumerate(pairs):
            a,b,c,d=[np.array(e[k]) for k in ('before_i','before_j','after_i','after_j')]
            err=max(abs(.5*(c@c+d@d-a@a-b@b)),np.linalg.norm(c+d-a-b));assert err<1e-10
            assert np.array_equal(c,attempts[i]['after_i']) and np.array_equal(d,attempts[i]['after_j'])
            peaks['pair']=max(peaks['pair'],float(err));pair_events+=1

    prefix_data={}
    for r in prefixes:
        name=r['prefix_id'];rows=trace(output/'prefix_traces'/(name+'.csv.gz'))
        cp=load(output/'checkpoints'/(name+'.json'));initial=load(output/'initial_states'/f"{r['case']}__s{r['seed']}.json")
        assert int(rows[-1]['total_oracle_calls'])==r['total_oracle_calls'] if r['reached'] else True
        assert json.loads(rows[0]['positions'])==[p['position'] for p in initial['particles']]
        assert json.loads(rows[0]['velocities'])==[p['velocity'] for p in initial['particles']]
        assert json.loads(rows[-1]['positions'])==[p['position'] for p in cp['particles']]
        assert json.loads(rows[-1]['velocities'])==[p['velocity'] for p in cp['particles']]
        assert cp['committed_steps']==len(rows)-1 and cp['calls']==r['total_oracle_calls']<=cfg['oracle_budget']
        if r['reached']:assert cp['calls']>=cfg['handoff_threshold']
        physical(rows,load(output/'prefix_events'/(name+'.json')),load(output/'prefix_maps'/(name+'.json')),0,'ramp')
        prefix_data[name]=(rows,cp)
    physical_suffixes={}
    for r in runs:
        rows,cp=prefix_data[r['prefix_id']];name=r['prefix_id']+'__'+r['suffix']
        if not r['suffix_executed']:
            assert r['failed'] and not cp['reached'];continue
        suffix=trace(output/'suffix_traces'/(name+'.csv.gz'))
        assert r['prefix_calls']==cp['calls']
        assert r['total_oracle_calls']==r['prefix_calls']+r['suffix_calls']<=cfg['oracle_budget']
        suffix_calls=int(suffix[-1]['total_oracle_calls']);assert suffix_calls<=r['total_oracle_calls']
        if r['suffix']=='gd':
            interface=load(output/'interfaces'/(name+'.json'));i=int(np.argmin(cp['values']))
            assert interface['selected_index']==i and interface['selected_state']==cp['particles'][i]
            assert json.loads(suffix[0]['x'])==cp['particles'][i]['position'][:-1]
            assert float(suffix[0]['value'])==cp['values'][i] and int(suffix[0]['total_oracle_calls'])==cp['calls']
            energies=[.5*np.dot(p['velocity'],p['velocity'])+.5*p['position'][-1] for p in cp['particles']]
            momenta=np.array([p['velocity'] for p in cp['particles']])
            assert abs(interface['retained_energy']-energies[i])<1e-12
            assert abs(interface['exported_energy']-(sum(energies)-energies[i]))<1e-12
            assert np.linalg.norm(np.array(interface['exported_momentum'])-(momenta.sum(axis=0)-momenta[i]))<1e-12
            for previous,t in zip(suffix,suffix[1:]):
                before=np.array(json.loads(t['before_x']));g=np.array(json.loads(t['gradient']));after=np.array(json.loads(t['x']))
                assert np.array_equal(before,json.loads(previous['x']))
                err=float(np.linalg.norm(after-(before-cfg['gd_eta']*g)));assert err<1e-14
                peaks['gd_update']=max(peaks['gd_update'],err)
                assert int(t['total_oracle_calls'])==int(previous['total_oracle_calls'])+2;gd_updates+=1
            s=score(suffix,cp['minimum'],cp['initial_value'],cp['target'],cp['boundary'],True)
        else:
            assert json.loads(suffix[0]['positions'])==json.loads(rows[-1]['positions'])
            assert json.loads(suffix[0]['velocities'])==json.loads(rows[-1]['velocities'])
            assert int(suffix[0]['total_oracle_calls'])==cp['calls']+4
            events=load(output/'suffix_events'/(name+'.json'))
            if r['suffix']=='independent':assert not any(e['kind']=='pair' for e in events)
            physical(suffix,events,load(output/'suffix_maps'/(name+'.json')),cp['committed_steps'],'ramp')
            s=score(suffix,cp['minimum'],cp['initial_value'],cp['target'],cp['boundary'])
            physical_suffixes[r['prefix_id'],r['suffix']]=(suffix,r)
        p=score(rows,cp['minimum'],cp['initial_value'],cp['target'],cp['boundary'])
        assert r['final_value']==s['final'] and r['best_value']==min(p['best'],s['best'])
        assert r['final_target_hit']==s['hit'] and r['historical_hit']==(p['ever'] or s['ever'])
        assert abs(r['relative_final_gap']-s['final_gap'])<1e-12
    common=[]
    for name,(rows,cp) in prefix_data.items():
        if not cp['reached']:continue
        a,ra=physical_suffixes[name,'continue'];b,rb=physical_suffixes[name,'independent'];j=min(len(a),len(b))-1
        sa=score([a[j]],cp['minimum'],cp['initial_value'],cp['target'],cp['boundary'])
        sb=score([b[j]],cp['minimum'],cp['initial_value'],cp['target'],cp['boundary'])
        assert float(a[j]['time'])==float(b[j]['time'])
        common.append({'prefix_id':name,'case':cp['case'],'seed':cp['seed'],'prefix_method':cp['prefix_method'],
            'common_global_time':cp['handoff_time']+float(a[j]['time']),
            'continue_gap':sa['final_gap'],'independent_gap':sb['final_gap'],
            'continue_calls':int(a[j]['total_oracle_calls']),'independent_calls':int(b[j]['total_oracle_calls']),
            'independent_minus_continue_gap':sb['final_gap']-sa['final_gap']})
        if cp['prefix_method']=='independent':
            assert len(a)==len(b)
            for x,y in zip(a,b):assert x['positions']==y['positions'] and x['velocities']==y['velocities']
    for r in replicas:
        name=f"{r['case']}__s{r['seed']}__elastic__{r['schedule']}"
        rows=trace(output/'replication/traces'/(name+'.csv.gz'));init=load(output/'initial_states'/f"{r['case']}__s{r['seed']}.json")
        physical(rows,load(output/'replication/events'/(name+'.json')),load(output/'replication/map_attempts'/(name+'.json')),0,r['schedule'])
        s=score(rows,init['minimum'],init['initial_value'],init['target'],init['boundary'])
        assert r['final_value']==s['final'] and r['final_target_hit']==s['hit']
        assert r['population_target_hit']==s['ever']
    counts=load(output/'execution_counts.json')
    actual=sum(r['total_oracle_calls'] for r in prefixes)+sum(r['suffix_calls'] for r in runs)+sum(r['total_oracle_calls'] for r in replicas)
    assert actual==counts['actual_query_calls']
    result={'passed':True,'prefixes':len(prefixes),'suffixes':sum(r['suffix_executed'] for r in runs),
        'uninterrupted_runs':len(replicas),'failure_rows_retained':sum(r['failed'] for r in runs+replicas),
        'physical_rows_checked':physical_rows,'pair_events_checked':pair_events,'gd_updates_checked':gd_updates,
        'max_errors':peaks,'common_time_comparisons':len(common),'actual_query_calls':actual}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'common_time.json').write_text(json.dumps(common,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');print(json.dumps(audit(p.parse_args().output),indent=2))
