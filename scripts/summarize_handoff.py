"""Derived E18 contrasts, shared-time checks and split/unsplit equivalence."""

import argparse
from collections import defaultdict, Counter
import csv
import gzip
import json
from pathlib import Path
import numpy as np


def load(path):return json.loads(path.read_text())
def trace(path):
    with gzip.open(path,'rt') as stream:return list(csv.DictReader(stream))


def summarize(output):
    output=Path(output);data=[]
    for kind,name in [('handoff','paired'),('replication','replication_paired')]:
        for r in load(output/(name+'.json')):data.append({**r,'kind':kind})
    groups=defaultdict(list)
    for r in data:groups[r['kind'],r['case'],r['prefix_method'],r['arm'],r['contrast']].append(r)
    summaries=[]
    for (kind,case,mode,arm,contrast),rows in sorted(groups.items()):
        counts=Counter(r['outcome'] for r in rows);valid=[r['final_gap_difference'] for r in rows if r['valid']]
        summaries.append({'kind':kind,'case':case,'prefix_method':mode,'arm':arm,'contrast':contrast,
            **{k:counts[k] for k in ('better','worse','tie','unavailable')},
            'median_paired_gap_difference':float(np.median(valid)) if valid else None})
    common_groups=defaultdict(list)
    for r in load(output/'common_time.json'):common_groups[r['case'],r['prefix_method']].append(r)
    common=[]
    for (case,mode),rows in sorted(common_groups.items()):
        ds=[r['independent_minus_continue_gap'] for r in rows]
        common.append({'case':case,'prefix_method':mode,'pairs':len(rows),
            'independent_better':sum(d < -1e-6 for d in ds),'independent_worse':sum(d > 1e-6 for d in ds),
            'tie':sum(abs(d)<=1e-6 for d in ds),'median_independent_minus_continue_gap':float(np.median(ds))})
    prefixes=load(output/'prefixes.json');equivalent=checked=0;endpoint_loss=[]
    for p in prefixes:
        if p['prefix_method']!='elastic' or not p['reached']:continue
        name=p['prefix_id'];cp=load(output/'checkpoints'/(name+'.json'));index=cp['committed_steps']
        prefix=trace(output/'prefix_traces'/(name+'.csv.gz'))
        suffix=trace(output/'suffix_traces'/(name+'__continue.csv.gz'))
        replica=trace(output/'replication/traces'/f"{p['case']}__s{p['seed']}__elastic__ramp.csv.gz")
        for i in range(min(len(prefix),len(replica))):
            assert prefix[i]['positions']==replica[i]['positions'] and prefix[i]['velocities']==replica[i]['velocities']
            checked+=1
        for j in range(min(len(suffix),len(replica)-index)):
            assert suffix[j]['positions']==replica[index+j]['positions'] and suffix[j]['velocities']==replica[index+j]['velocities']
            assert int(suffix[j]['value_calls'])+int(suffix[j]['gradient_calls'])+int(suffix[j]['path_calls'])==int(replica[index+j]['total_oracle_calls'])+4
            checked+=1
        equivalent+=1;endpoint_loss.append(len(replica)-(index+len(suffix)))
    result={'contrasts':summaries,'common_physical_time':common,
        'split_vs_uninterrupted_ramp':{'paired_paths_identical_on_common_grid':equivalent,'observation_pairs_checked':checked,
            'extra_paid_handoff_validation_calls':4,'lost_final_observation_steps_min':min(endpoint_loss,default=0),
            'lost_final_observation_steps_max':max(endpoint_loss,default=0)},
        'counting':load(output/'execution_counts.json')}
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('contrasts','common_physical_time')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');summarize(p.parse_args().output)
