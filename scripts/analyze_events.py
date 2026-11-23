"""Frozen E20 event effects, train-only predictors and diagnostic figures."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from collision_optim.event_replay import MAGNITUDE_KEYS,DIRECTION_KEYS
from collision_optim.timing import fit_ridge,predict_ridge


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(output):
    rows=read(output/'runs.json');contexts=read(output/'contexts.json');result=[];missing=[]
    for c in sorted(contexts,key=lambda c:c['context_id']):
        group=[r for r in rows if r['context_id']==c['context_id']]
        elastic=[r for r in group if r['rule']=='elastic'];none=[r for r in group if r['rule']=='none']
        random=sorted((r for r in group if r['rule']=='random'),key=lambda r:r['replica'])
        if len(elastic)!=1 or len(none)!=1 or len(random)!=4 or any(r['failed'] for r in group):
            missing.append({'context_id':c['context_id'],'reason':'Missing or failed branch'});continue
        views={}
        for window in ('budget1000','boundary10','boundary50','boundary100','end'):
            if any(r['views'][window] is None for r in group):continue
            e=elastic[0]['views'][window];n=none[0]['views'][window];rs=[r['views'][window] for r in random]
            scale=max(c['initial_best']-c['minimum'],1e-12)
            views[window]={'geometry_value':float((np.mean([r['pair_final_loss'] for r in rs])-e['pair_final_loss'])/scale),
                'collision_value':float((n['pair_final_loss']-e['pair_final_loss'])/scale),
                'elastic':e,'none':n,'random':rs,
                'random_pair_loss_sd':float(np.std([r['pair_final_loss'] for r in rs]))}
        if 'budget1000' not in views:
            missing.append({'context_id':c['context_id'],'reason':'Unavailable primary budget window'});continue
        result.append({'context_id':c['context_id'],'case':c['case'],'seed':c['seed'],'features':c['features'],
                       'value':views['budget1000']['geometry_value'],'views':views})
    write(output/'labels.json',result);write(output/'unavailable_labels.json',missing);return result


def effect_summary(rows):
    result=[]
    for case in sorted({r['case'] for r in rows}):
        subset=[r for r in rows if r['case']==case]
        for window in ('budget1000','boundary10','boundary50','boundary100'):
            available=[r for r in subset if window in r['views']]
            for field in ('geometry_value','collision_value'):
                values=[r['views'][window][field] for r in available]
                if not values:continue
                byseed={s:float(np.mean([r['views'][window][field] for r in available if r['seed']==s])) for s in sorted({r['seed'] for r in available})}
                result.append({'case':case,'window':window,'effect':field,'contexts':len(values),'seeds':len(byseed),
                    'better':sum(v>1e-6 for v in values),'worse':sum(v< -1e-6 for v in values),'tie':sum(abs(v)<=1e-6 for v in values),
                    'mean':float(np.mean(values)),'median':float(np.median(values)),
                    'seed_equal_weight_mean':float(np.mean(list(byseed.values()))),'per_seed_mean':byseed})
    return result


def evaluate(model,rows):
    y=np.array([r['value'] for r in rows]);p=predict_ridge(model,rows);mask=np.abs(y)>1e-6
    return {'mae':float(np.mean(abs(y-p))),
            'non_tie_sign_accuracy':float(np.mean(np.sign(y[mask])==np.sign(p[mask]))) if mask.any() else None,
            'per_seed_mae':{str(s):float(np.mean([abs(r['value']-p[i]) for i,r in enumerate(rows) if r['seed']==s])) for s in sorted({r['seed'] for r in rows})},
            'predictions':p.tolist()}


def plot_effects(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=('quadratic','double_well','rosenbrock');fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,case in zip(axes,cases):
        subset=[r for r in rows if r['case']==case];seeds=sorted({r['seed'] for r in subset})
        for x,field in enumerate(('geometry_value','collision_value')):
            means=[np.mean([r['views']['budget1000'][field] for r in subset if r['seed']==s]) for s in seeds]
            ax.scatter(np.full(len(means),x),means,alpha=.65,s=20)
            if means:ax.plot([x-.2,x+.2],[np.mean(means)]*2,color='black',lw=2)
        ax.axhline(0,color='grey',lw=1);ax.set_xticks([0,1],['vs random mean','vs no impulse']);ax.set_title(case.replace('_',' '))
        ax.set_ylabel('Elastic benefit / initial gap\n(one dot per seed)')
    fig.tight_layout();fig.savefig(output/'event_effects.png',dpi=160);plt.close(fig)


def sensitivity_analysis(output):
    records=read(output/'sensitivity_runs.json');common=[]
    for r in records:
        with gzip.open(output/'traces'/(r['run_id']+'.csv.gz'),'rt') as f:rows=list(csv.DictReader(f))
        for t in rows:
            time=float(t['time'])
            if abs(time/.05-round(time/.05))<1e-8:
                values=json.loads(t['values'])
                common.append({'run_id':r['run_id'],'seed':r['seed'],'h':r['h'],'tight':r['tight'],'time':time,
                               'protected_loss_rise':values[0]-r['protected_initial_loss']})
    write(output/'sensitivity_common_time.json',common)
    summaries=[]
    for h in (.05,.025,.0125):
        for tight in (False,True):
            subset=[r for r in records if r['h']==h and r['tight']==tight]
            if not subset:continue
            summaries.append({'h':h,'tight':tight,'runs':len(subset),
                'median_final_rise':float(np.median([r['protected_final_rise'] for r in subset])),
                'median_all_grid_max_rise':float(np.median([r['protected_max_rise'] for r in subset])),
                'median_common_grid_max_rise':float(np.median([max(x['protected_loss_rise'] for x in common if x['run_id']==r['run_id']) for r in subset]))})
    write(output/'sensitivity_summary.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for r in summaries:
        points=[p for p in common if p['h']==r['h'] and p['tight']==r['tight']]
        times=sorted({round(p['time'],8) for p in points})
        losses=[np.median([p['protected_loss_rise'] for p in points if abs(p['time']-t)<1e-8]) for t in times]
        ax.plot(times,np.maximum(losses,1e-20),label=f'h={r["h"]}, '+('tight' if r['tight'] else 'default'),ls='--' if r['tight'] else '-')
    ax.set_yscale('log');ax.set_xlabel('Physical time');ax.set_ylabel('Protected loss rise (median of 3 seeds)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'sensitivity.png',dpi=160);plt.close(fig)


def analyze(output,development=None):
    output=Path(output);cfg=read(output/'config.json');assert read(output/'audit.json')['passed']
    if cfg['stage']=='sensitivity':
        sensitivity_analysis(output);return
    rows=labels(output);write(output/'effects.json',effect_summary(rows));plot_effects(output,rows)
    if cfg['stage']=='development':
        if not rows:raise ValueError('No available development labels.')
        models={'magnitude':fit_ridge(rows,MAGNITUDE_KEYS),
                'direction':fit_ridge(rows,MAGNITUDE_KEYS+DIRECTION_KEYS)}
        positive=sum(r['value']>1e-6 for r in rows);negative=sum(r['value']< -1e-6 for r in rows)
        models['training_majority_sign']=1 if positive>negative else -1
        write(output/'predictors.json',models)
        names=['predictors.json','labels.json','runs.json','contexts.json','source_manifest.json']
        path=output/'frozen_inputs.json'
        if path.exists():raise FileExistsError('Never overwrite frozen development inputs.')
        write(path,{'phase':'E20','frozen_before_confirmation':True,'development_seeds':cfg['seeds'],
                    'confirmation_seeds':list(range(1240,1260)),'files':{n:digest(output/n) for n in names}})
        print('Development predictors frozen:',len(rows),'contexts')
    elif cfg['stage']=='confirmation':
        if not development:raise ValueError('Development folder required.')
        development=Path(development);frozen=read(development/'frozen_inputs.json')
        assert digest(development/'frozen_inputs.json')==cfg['development_frozen_sha256']
        for name,h in frozen['files'].items():assert digest(development/name)==h
        models=read(development/'predictors.json');metrics={k:evaluate(models[k],rows) for k in ('magnitude','direction')}
        available=[r for r in rows if abs(r['value'])>1e-6]
        base=metrics['magnitude'];new=metrics['direction'];seeds=base['per_seed_mae']
        improvements=sum(new['per_seed_mae'][s]<seeds[s] for s in seeds)
        metrics.update(contexts=len(rows),independent_seeds=len(seeds),
            positive=sum(r['value']>1e-6 for r in rows),negative=sum(r['value']< -1e-6 for r in rows),ties=len(rows)-len(available),
            training_majority_sign=models['training_majority_sign'],
            training_majority_accuracy=float(np.mean([np.sign(r['value'])==models['training_majority_sign'] for r in available])) if available else None,
            seed_mae_improvements=improvements,mae_relative_reduction=1-new['mae']/base['mae'] if base['mae'] else None,
            promising_screen_passed=bool(new['mae']<=.95*base['mae'] and improvements>=14 and len(seeds)==20),
            frozen_inputs_unchanged=True,online_policy_evaluated=False)
        write(output/'heldout_prediction.json',metrics)
        print(json.dumps({k:v for k,v in metrics.items() if k not in ('magnitude','direction')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--development');a=p.parse_args();analyze(a.output,a.development)
