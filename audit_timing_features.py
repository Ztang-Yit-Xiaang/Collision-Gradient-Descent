"""Reconstruct past-only features and offline labels from independent raw views."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def trace(p):
    with gzip.open(p,'rt') as f:return list(csv.DictReader(f))


def audit(output):
    output=Path(output);cps=read(output/'checkpoints.json');cache={};peak=0.
    for cp in cps:
        name=cp['source_full_id']
        if name not in cache:cache[name]=(trace(output/'traces'/(name+'__full.csv.gz')),read(output/'events'/(name+'__full.json')))
        rows,events=cache[name];step=cp['step'];row=rows[step]
        v=np.array(json.loads(row['velocities']));q=np.array(json.loads(row['positions']));values=np.array(json.loads(row['values']))
        start=max(1,step-19);past=[e for e in events if e['kind']=='pair' and start<=e['step']<=step]
        scale=max(1.,abs(min(json.loads(rows[0]['values']))));center=np.mean(v,axis=0)
        transfer=[abs(.5*(np.dot(e['after_i'],e['after_i'])-np.dot(e['before_i'],e['before_i']))) for e in past]
        expected={'budget_fraction':cp['calls']/read(output/'config.json')['oracle_budget'],
            'current_loss':float(min(values)/scale),
            'recent_loss_improvement':float((min(json.loads(rows[start]['values']))-min(values))/scale),
            'kinetic':float(.5*np.sum(v*v)),
            'clearance_potential':float(.5*np.sum(q[:,-1]-values)),
            'center_speed':float(np.linalg.norm(center)),
            'relative_speed_rms':float(np.sqrt(np.sum((v-center)**2)/len(v))),
            'position_dispersion':float(np.sqrt(np.sum((q-np.mean(q,axis=0))**2)/len(q))),
            'recent_pair_count':len(past),'mean_abs_kinetic_transfer':float(np.mean(transfer)) if transfer else 0.}
        for key,value in expected.items():
            error=abs(value-cp['features'][key]);assert error<1e-10,(name,key,error);peak=max(peak,error)
    labels=read(output/'collision_value_labels.json') if (output/'collision_value_labels.json').exists() else []
    bykey={cp['source_full_id']+f"__off{cp['fraction']}":cp for cp in cps};label_peak=0.
    for label in labels:
        cp=bykey[label['checkpoint_id']];full=cache[cp['source_full_id']][0];off=trace(output/'traces'/(label['checkpoint_id']+'.csv.gz'))
        assert label['features']==cp['features'] and label['feature_last_calls']==cp['calls']
        cap=label['cap'];on=[r for r in full[cp['step']:] if int(r['total_oracle_calls'])+4<=cap][-1]
        other=[r for r in off if int(r['total_oracle_calls'])<=cap][-1]
        value=(min(json.loads(other['values']))-min(json.loads(on['values'])))/max(cp['initial_value']-cp['minimum'],1e-12)
        error=abs(value-label['value']);assert error<1e-12;label_peak=max(label_peak,error)
        assert label['charged_on_calls']==int(on['total_oracle_calls'])+4<=cap
        assert label['off_calls']==int(other['total_oracle_calls'])<=cap
    if (output/'frozen_analysis_inputs.json').exists():
        for name,h in read(output/'frozen_analysis_inputs.json')['files'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==h
    result={'passed':True,'checkpoints_checked':len(cps),'past_only_feature_max_error':peak,
        'derived_labels_checked':len(labels),'label_max_error':label_peak}
    (output/'feature_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');audit(p.parse_args().output)
