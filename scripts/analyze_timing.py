"""E19 development selection and frozen held-out offline signal evaluation."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from collision_optim.timing import BASE_FEATURES,PHYSICS_FEATURES,FRACTIONS,fit_ridge,predict_ridge


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def trace(p):
    with gzip.open(p,'rt') as f:return list(csv.DictReader(f))


def criterion(rows):
    return float(np.mean([np.log1p(1e6) if r['failed'] else np.log1p(r['relative_final_gap']) for r in rows]))


def select(runs,baselines):
    cases=sorted({r['case'] for r in runs});scores={}
    for f in FRACTIONS:
        scores[str(f)]=float(np.mean([criterion([r for r in runs if r['case']==c and r['rule']==('independent' if f==0 else 'elastic') and r['fraction']==f]) for c in cases]))
    best=min(scores.values());candidates=[f for f in FRACTIONS if scores[str(f)]<=best+1e-12]
    fraction=min(candidates,key=lambda f:(abs(f-.5),f));rates={}
    for c in cases:
        rates[c]={}
        for family in ('gd','momentum'):
            group=[r for r in baselines if r['case']==c and r['family']==family]
            options=sorted({r['eta'] for r in group})
            rates[c][family]=min(options,key=lambda eta:(criterion([r for r in group if r['eta']==eta]),eta))
    return {'fraction':fraction,'development_fraction_scores':scores,'baseline_rates':rates,
            'rule':'equal-case mean log1p final relative gap, failures penalized; global elastic fraction only'}


def labels(output):
    output=Path(output);cfg=read(output/'config.json');runs=read(output/'runs.json');lookup={r['trace_id']:r for r in runs if r.get('trace_id')}
    result=[]
    for cp in read(output/'checkpoints.json'):
        name=cp['source_full_id'];key=name+f"__off{cp['fraction']}"
        if lookup[name+'__full']['failed'] or lookup[key]['failed']:continue
        full=trace(output/'traces'/(name+'__full.csv.gz'));off=trace(output/'traces'/(key+'.csv.gz'))
        cap=min(cfg['oracle_budget'],cp['calls']+1000)
        raw_a=[r for r in full[cp['step']:] if int(r['total_oracle_calls'])<=cap][-1]
        # The reused on curve has no resume validation; equal-charge labels
        # reserve the same four-value overhead that the off fork actually pays.
        a=[r for r in full[cp['step']:] if int(r['total_oracle_calls'])+4<=cap][-1]
        b=[r for r in off if int(r['total_oracle_calls'])<=cap][-1]
        scale=max(cp['initial_value']-cp['minimum'],1e-12)
        loss=lambda r:min(json.loads(r['values']))
        horizon=min(20,len(full)-cp['step']-1,len(off)-1)
        result.append({'case':cp['case'],'seed':cp['seed'],'rule':cp['rule'],'fraction':cp['fraction'],
            'checkpoint_id':key,'features':cp['features'],'feature_last_calls':cp['calls'],
            'cap':cap,'source_on_calls':int(a['total_oracle_calls']),'charged_on_calls':int(a['total_oracle_calls'])+4,
            'off_calls':int(b['total_oracle_calls']),'derived_on_validation_charge':4,
            'value':(loss(b)-loss(a))/scale,'raw_budget_value':(loss(b)-loss(raw_a))/scale,'common_physical_steps':horizon,
            'physical_value':(loss(off[horizon])-loss(full[cp['step']+horizon]))/scale,
            'note':'Derived label with equal four-value fork validation charges; on source curve reused without executing extra queries. Raw budget value also retained; no online lookahead.'})
    return result


def evaluate(models,rows,training):
    report={};truth=np.array([r['value'] for r in rows]);active=np.abs(truth)>1e-6
    train_active=[r['value'] for r in training if abs(r['value'])>1e-6]
    majority=1 if sum(v>0 for v in train_active)>=len(train_active)/2 else -1
    for name,model in models.items():
        prediction=predict_ridge(model,rows);per_seed={}
        for seed in sorted({r['seed'] for r in rows}):
            mask=np.array([r['seed']==seed for r in rows]);per_seed[str(seed)]=float(np.mean(np.abs(prediction[mask]-truth[mask])))
        report[name]={'mae':float(np.mean(np.abs(prediction-truth))),
            'non_tie_sign_accuracy':float(np.mean((prediction[active]>0)==(truth[active]>0))) if active.any() else None,
            'per_seed_mae':per_seed,'predictions':prediction.tolist()}
    report['class_balance']={'rows':len(rows),'positive':int(sum(truth>1e-6)),'negative':int(sum(truth< -1e-6)),
        'ties':int(sum(~active)),'train_majority_sign':majority,
        'train_majority_sign_accuracy':float(np.mean((truth[active]>0)==(majority>0))) if active.any() else None}
    return report


def analyze(output,development=None):
    output=Path(output);cfg=read(output/'config.json');rows=labels(output);write(output/'collision_value_labels.json',rows)
    if cfg['stage']=='development':
        selection=select(read(output/'runs.json'),read(output/'baselines.json'))
        models={'base':fit_ridge(rows,BASE_FEATURES),'physics':fit_ridge(rows,BASE_FEATURES+PHYSICS_FEATURES)}
        write(output/'predictors.json',models)
        selection.update({'development_seeds':cfg['seeds'],'source_manifest':cfg['source_sha256'],
            'runs_sha256':hashlib.sha256((output/'runs.json').read_bytes()).hexdigest(),
            'baselines_sha256':hashlib.sha256((output/'baselines.json').read_bytes()).hexdigest(),
            'predictors_sha256':hashlib.sha256((output/'predictors.json').read_bytes()).hexdigest()})
        write(output/'selection.json',selection);print(json.dumps({k:selection[k] for k in ('fraction','baseline_rates','development_fraction_scores')},indent=2))
    elif cfg['stage']=='confirmation':
        development=Path(development);selection=read(development/'selection.json')
        assert not set(cfg['seeds'])&set(selection['development_seeds'])
        assert hashlib.sha256((development/'selection.json').read_bytes()).hexdigest()==cfg['selection_sha256']
        assert hashlib.sha256((development/'predictors.json').read_bytes()).hexdigest()==selection['predictors_sha256']
        models=read(development/'predictors.json');training=read(development/'collision_value_labels.json')
        result={'selected_fraction':selection['fraction'],'overall':evaluate(models,rows,training),'groups':{}}
        for case in cfg['cases']:
            for rule in ('elastic','random'):
                group=[r for r in rows if r['case']==case and r['rule']==rule]
                result['groups'][case+'__'+rule]=evaluate(models,group,training)
        write(output/'heldout_prediction.json',result)
        print(json.dumps({'selected_fraction':selection['fraction'],'class_balance':result['overall']['class_balance'],
            'base_mae':result['overall']['base']['mae'],'physics_mae':result['overall']['physics']['mae']},indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--development');a=p.parse_args();analyze(a.output,a.development)
