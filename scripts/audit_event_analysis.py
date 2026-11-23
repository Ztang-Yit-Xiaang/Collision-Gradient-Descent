"""Independent map, effect-label and frozen linear-inference checks for E20."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())


def audit(output,development=None):
    p=Path(output);contexts=read(p/'contexts.json');runs=read(p/'runs.json');labels=read(p/'labels.json')
    assert read(p/'audit.json')['passed'];ctx={c['context_id']:c for c in contexts};max_map=max_label=0.
    for r in runs:
        c=ctx[r['context_id']];m=read(p/'maps'/(r['run_id']+'.json'));v=np.array(m['before']['velocities'])
        i,j=c['i'],c['j'];n=np.array(c['normal']);a,b=v[i].copy(),v[j].copy()
        if r['rule']=='elastic':
            rel=(a-b)@n;a,b=a-rel*n,b+rel*n
        elif r['rule']=='random':
            rng=np.random.default_rng(np.random.SeedSequence(m['rng_entropy']));d=rng.normal(size=len(a));d/=np.linalg.norm(d)
            if d@n<0:d=-d
            center=(a+b)/2;relative=np.linalg.norm((a-b)/2)*d;a,b=center+relative,center-relative
        v[i]=a;v[j]=b;err=float(np.max(np.abs(v-m['after']['velocities'])));max_map=max(max_map,err);assert err<1e-12
    for label in labels:
        c=ctx[label['context_id']];group=[r for r in runs if r['context_id']==c['context_id']]
        scale=max(c['initial_best']-c['minimum'],1e-12);assert label['features']==c['features']
        for key,view in label['views'].items():
            records=[r['views'][key] for r in group];assert all(v is not None for v in records)
            e=next(r['views'][key] for r in group if r['rule']=='elastic')
            n=next(r['views'][key] for r in group if r['rule']=='none')
            rs=[r['views'][key] for r in group if r['rule']=='random'];assert len(rs)==4
            expected=(sum(r['pair_final_loss'] for r in rs)/4-e['pair_final_loss'])/scale
            err=abs(expected-view['geometry_value']);max_label=max(max_label,err);assert err<1e-12
            assert abs((n['pair_final_loss']-e['pair_final_loss'])/scale-view['collision_value'])<1e-12
            if key.startswith('boundary'):
                assert max(v['time'] for v in records)-min(v['time'] for v in records)<1e-12
            if key=='budget1000':
                assert all(v['charged_calls']<=1000 for v in records)
                assert abs(expected-label['value'])<1e-12
    checks={'passed':True,'maps_checked':len(runs),'labels_checked':len(labels),'max_map_error':max_map,'max_label_error':max_label}
    if development:
        d=Path(development);frozen=read(d/'frozen_inputs.json');cfg=read(p/'config.json')
        assert hashlib.sha256((d/'frozen_inputs.json').read_bytes()).hexdigest()==cfg['development_frozen_sha256']
        assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in frozen['files'].items())
        models=read(d/'predictors.json');training=read(d/'labels.json');heldout=read(p/'heldout_prediction.json');error=0.
        for key in ('magnitude','direction'):
            m=models[key];x=np.array([[r['features'][k] for k in m['keys']] for r in training])
            assert np.max(np.abs(x.mean(axis=0)-m['mean']))<1e-12
            scale=x.std(axis=0);scale[scale<1e-12]=1.
            assert np.max(np.abs(scale-m['scale']))<1e-12
            test=np.array([[r['features'][k] for k in m['keys']] for r in labels])
            prediction=np.c_[np.ones(len(test)),(test-np.array(m['mean']))/np.array(m['scale'])]@np.array(m['coef'])
            err=float(np.max(np.abs(prediction-heldout[key]['predictions'])));error=max(error,err);assert err<1e-12
            mae=float(np.mean(np.abs(prediction-np.array([r['value'] for r in labels]))))
            assert abs(mae-heldout[key]['mae'])<1e-12
        checks.update(frozen_inputs_unchanged=True,training_standardization_only=True,max_prediction_error=error)
    (p/'analysis_audit.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--development');a=parser.parse_args();audit(a.output,a.development)
