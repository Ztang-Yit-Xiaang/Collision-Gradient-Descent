"""Reproduction, symmetry and cross-scale checks from E21 raw artifacts."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def trace(p):
    with gzip.open(p,'rt') as f:return list(csv.DictReader(f))


def check(rosen,bowl):
    r=Path(rosen);p=Path(bowl);runs=read(r/'runs.json');replicas=[]
    for rec in runs:
        if rec['layout']!='joint' or rec['perturbation']!='zero':continue
        old=Path('output/collision_event_sensitivity_long/traces')/f'rosenbrock__s{rec["spectator_seed"]}__h{rec["h"]}__tight0.csv.gz'
        a=trace(old);b=trace(r/'traces'/(rec['run_id']+'.csv.gz'));assert len(a)==len(b)
        error=0.;exact=True
        for x,y in zip(a,b):
            for k in ('positions','velocities','values'):
                u,v=json.loads(x[k]),json.loads(y[k]);exact=exact and u==v
                error=max(error,float(np.max(np.abs(np.array(u)-v))))
        assert error<1e-9
        replicas.append({'run_id':rec['run_id'],'parent_trace':str(old),'rows':len(a),'states_values_exact':exact,'max_error':error})
    records=read(p/'runs.json');symmetry=[];zero=[]
    for kappa in (1.,4.,16.):
        for product in (.2,.49,.51,.7):
            group=[x for x in records if x['kappa']==kappa and x['product']==product]
            plus=next(x for x in group if x['initial_x']>0);minus=next(x for x in group if x['initial_x']<0);ref=next(x for x in group if x['initial_x']==0)
            a=read(p/'return_traces'/(plus['run_id']+'.json'))['rows'];b=read(p/'return_traces'/(minus['run_id']+'.json'))['rows']
            err=max(max(abs(x['x']+y['x']),abs(x['vx']+y['vx']),abs(x['loss']-y['loss']),abs(x['time']-y['time'])) for x,y in zip(a,b))
            assert len(a)==len(b)==31 and err<1e-10
            c=read(p/'return_traces'/(ref['run_id']+'.json'))['rows'];assert all(x['x']==x['vx']==x['loss']==0 for x in c)
            symmetry.append({'kappa':kappa,'product':product,'mirror_error':err});zero.append(ref['run_id'])
    consistency=read(p/'probe_consistency.json');spread=[]
    for product in sorted({x['product'] for x in consistency}):
        vals=[x['spectral_radius'] for x in consistency if x['product']==product]
        spread.append({'product':product,'cross_curvature_radius_range':max(vals)-min(vals)})
    assert all(c['passed'] for c in consistency)
    legacy=read(Path('output/collision_event_confirmation/source_manifest.json'))
    names=[f for f in legacy if f.startswith(('src/','tests/'))]
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==legacy[f] for f in names)
    result={'passed':True,'legacy_model_test_files_restored_or_unchanged':len(names),'baseline_reproductions':replicas,
            'mirror_rollout_pairs':symmetry,'exact_vertical_reference_runs':zero,'cross_scale_radius_spread':spread,
            'all_30_probe_size_consistency_screens_passed':True,'no_stability_theorem_claimed':True}
    (r/'empirical_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'baseline_reproductions':len(replicas),'baseline_exact':all(x['states_values_exact'] for x in replicas),
          'mirror_pairs':len(symmetry),'exact_vertical_references':len(zero),'max_cross_scale_radius_range':max(x['cross_curvature_radius_range'] for x in spread)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rosen');p.add_argument('bowl');a=p.parse_args();check(a.rosen,a.bowl)
