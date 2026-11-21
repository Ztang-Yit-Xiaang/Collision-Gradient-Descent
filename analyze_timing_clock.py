"""Predeclared clock control: physical signals must add more than elapsed time."""

import argparse,hashlib,json
from pathlib import Path
from analyze_timing import read,write,evaluate
from collision_optim.timing import BASE_FEATURES,PHYSICS_FEATURES,fit_ridge


def enrich(output):
    output=Path(output);rows=read(output/'collision_value_labels.json')
    for row in rows:
        cp=read(output/'checkpoints'/(row['checkpoint_id']+'.json'))
        row['features']={**row['features'],'physical_time':cp['step']*.05}
    return rows


def analyze(output,development=None):
    output=Path(output);cfg=read(output/'config.json');rows=enrich(output)
    if cfg['stage']=='development':
        models={'clock':fit_ridge(rows,BASE_FEATURES+('physical_time',)),
                'clock_physics':fit_ridge(rows,BASE_FEATURES+('physical_time',)+PHYSICS_FEATURES)}
        write(output/'clock_predictors.json',models)
        write(output/'clock_control_provenance.json',{'stage':'Development-only fit; declared before development results were inspected',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_seeds':cfg['seeds'],
            'model_sha256':hashlib.sha256((output/'clock_predictors.json').read_bytes()).hexdigest()})
    else:
        development=Path(development);models=read(development/'clock_predictors.json');training=enrich(development)
        provenance=read(development/'clock_control_provenance.json')
        assert not set(cfg['seeds'])&set(provenance['training_seeds'])
        assert hashlib.sha256((development/'clock_predictors.json').read_bytes()).hexdigest()==provenance['model_sha256']
        result={'overall':evaluate(models,rows,training),'groups':{}}
        for case in cfg['cases']:
            for rule in ('elastic','random'):
                result['groups'][case+'__'+rule]=evaluate(models,[r for r in rows if r['case']==case and r['rule']==rule],training)
        write(output/'clock_heldout.json',result)
        print(json.dumps({k:{metric:v[metric] for metric in ('mae','non_tie_sign_accuracy')} for k,v in result['overall'].items() if k!='class_balance'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--development');a=p.parse_args();analyze(a.output,a.development)
