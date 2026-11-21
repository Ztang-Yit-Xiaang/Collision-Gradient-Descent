"""Postprocess E17 paired contrasts and verify secondary metrics from raw traces."""

from collections import Counter,defaultdict
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np


def diagnose(output):
    output=Path(output);load=lambda p:json.loads(p.read_text());runs=load(output/'runs.json')
    pairs=load(output/'paired.json');groups=defaultdict(list)
    for r in pairs:groups[r['case'],r['method'],r['schedule'],r['contrast']].append(r)
    contrasts=[]
    for (case,method,schedule,contrast),rows in sorted(groups.items()):
        counts=Counter(r['outcome'] for r in rows);valid=[r['final_gap_difference'] for r in rows if r['valid']]
        contrasts.append({'case':case,'method':method,'schedule':schedule,'contrast':contrast,
            **{k:counts[k] for k in ('better','worse','tie','unavailable')},
            'median_paired_final_gap_difference':float(np.median(valid)) if valid else None})
    contact_pairs=no_maps=0;peak_tail=0.;geometry=[]
    for r in runs:
        name=f"{r['case']}__s{r['seed']}__{r['method']}__{r['schedule']}"
        with gzip.open(output/'traces'/(name+'.csv.gz'),'rt') as f:trace=list(csv.DictReader(f))
        init=load(output/'initial_states'/f"{r['case']}__s{r['seed']}.json")
        hits=[];times=[]
        for t in trace:
            hits.append(any(v<=init['target'] and (init['boundary'] is None or p[0]<init['boundary'])
                for v,p in zip(json.loads(t['values']),json.loads(t['positions']))))
            times.append(float(t['time']))
        hit_times=[t for t,h in zip(times,hits) if h]
        assert r['first_observed_hit_time']==(min(hit_times) if hit_times else None)
        window_start=.8*times[-1];covered=0.;hit_time=0.
        for i in range(len(times)-1):
            dt=max(0.,times[i+1]-max(window_start,times[i]));covered+=dt
            if hits[i]:hit_time+=dt
        if covered:
            err=abs(hit_time/covered-r['tail_target_occupancy']);assert err<1e-12;peak_tail=max(peak_tail,err)
        else:assert r['tail_target_occupancy'] is None
        positions=np.array(json.loads(trace[-1]['positions']));values=np.array(json.loads(trace[-1]['values']))
        winner=int(np.argmin(values))
        geometry.append({'case':r['case'],'seed':r['seed'],'method':r['method'],'schedule':r['schedule'],
            'minimum_center_separation':min(float(np.linalg.norm(positions[i]-positions[j])) for i in range(4) for j in range(i+1,4)),
            'minimum_parameter_separation':min(float(np.linalg.norm(positions[i,:-1]-positions[j,:-1])) for i in range(4) for j in range(i+1,4)),
            'best_particle_clearance':float(positions[winner,-1]-values[winner]),
            'final_relative_gap':r['relative_final_gap']})
        if r['method']=='elastic':
            a=load(output/'map_attempts'/(name+'.json'))
            b=load(output/'map_attempts'/(name.replace('__elastic__','__random__')+'.json'))
            if not a and not b:no_maps+=1
            else:
                assert a and b
                for key in ('before_i','before_j','normal'):assert np.array_equal(a[0][key],b[0][key])
                contact_pairs+=1
    result={'secondary_metrics_verified_runs':len(runs),'max_tail_occupancy_error':peak_tail,
            'first_kernel_input_pairs_identical':contact_pairs,'paired_populations_without_map':no_maps,
            'contrasts':contrasts}
    (output/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'exploratory_geometry.json').write_text(json.dumps({'note':'Exploratory endpoint distances added after seeing completed quadratic results; no run settings/scoring changed. These associations do not isolate a crowding mechanism.', 'rows':geometry},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='contrasts'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');diagnose(p.parse_args().output)
