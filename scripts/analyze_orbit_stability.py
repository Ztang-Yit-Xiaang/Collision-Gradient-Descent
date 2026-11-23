"""Empirical E21 comparisons and plots; no analytic stability formula fitted."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def rosen_analysis(p,runs):
    cache={};common=[]
    for r in runs:
        with gzip.open(p/'traces'/(r['run_id']+'.csv.gz'),'rt') as f:rows=list(csv.DictReader(f))
        points=[]
        for row in rows:
            t=float(row['time'])
            if abs(t/.05-round(t/.05))>1e-8:continue
            q=json.loads(row['positions'])[0];v=json.loads(row['velocities'])[0];values=json.loads(row['values'])
            point={'run_id':r['run_id'],'time':t,'grid_index':round(t/.05),'x':q[:-1],'velocity':v[:-1],
                   'loss':values[0],'layout':r['layout'],'height_label':r['height_label'],'height':r['height'],
                   'h':r['h'],'perturbation':r['perturbation'],'spectator_seed':r['spectator_seed']}
            points.append(point);common.append(point)
        cache[r['run_id']]={x['grid_index']:x for x in points}
    write(p/'rosen_common_time.json',common)
    perturbations=[];layouts=[];height_table=[]
    for r in runs:
        points=cache[r['run_id']];initial=read(p/'initial_states'/(r['run_id']+'.json'));opt=np.array(initial['optimum'])
        height_table.append({**r,'common_max_displacement':max(float(np.linalg.norm(np.array(x['x'])-opt)) for x in points.values()),
                             'common_max_loss_rise':max(0.,max(x['loss']-initial['reference_loss'] for x in points.values()))})
        if r['perturbation']!='zero':
            zero=next(z for z in runs if z['layout']==r['layout'] and z['h']==r['h'] and z['height_label']==r['height_label']
                      and z['spectator_seed']==r['spectator_seed'] and z['perturbation']=='zero')
            baseline=cache[zero['run_id']];grid=sorted(points.keys()&baseline.keys())
            deviation=[float(np.linalg.norm(np.array(points[t]['x'])-baseline[t]['x'])) for t in grid]
            perturbations.append({'run_id':r['run_id'],'baseline_run':zero['run_id'],'height':r['height'],'h':r['h'],
                'layout':r['layout'],'perturbation':r['perturbation'],'spectator_seed':r['spectator_seed'],
                'max_deviation':max(deviation),'final_deviation':deviation[-1],
                'max_amplification':max(deviation)/1e-8,'final_amplification':deviation[-1]/1e-8,
                'first_time_deviation_above_1e-6':next((t*.05 for t,d in zip(grid,deviation) if d>1e-6),None)})
        if r['layout']=='joint':
            single=next(s for s in runs if s['layout']=='single' and s['height_label']=='original' and s['h']==r['h'] and s['perturbation']==r['perturbation'])
            baseline=cache[single['run_id']];grid=sorted(points.keys()&baseline.keys())
            delta=[float(np.linalg.norm(np.array(points[t]['x'])-baseline[t]['x'])) for t in grid]
            layouts.append({'joint_run':r['run_id'],'single_run':single['run_id'],'h':r['h'],'perturbation':r['perturbation'],
                            'spectator_seed':r['spectator_seed'],'max_position_difference':max(delta),'final_position_difference':delta[-1],
                            'first_time_difference_above_1e-6':next((t*.05 for t,d in zip(grid,delta) if d>1e-6),None)})
    write(p/'height_summary.json',height_table);write(p/'perturbation_growth.json',perturbations);write(p/'layout_comparisons.json',layouts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for r in sorted((r for r in runs if r['layout']=='single' and r['h']==.0125 and r['perturbation']=='high_plus'),key=lambda r:r['height']):
        points=list(cache[r['run_id']].values());ax.plot([x['time'] for x in points],np.maximum([x['loss'] for x in points],1e-30),label=f'H={r["height"]:.4g}')
    ax.set_yscale('log');ax.set_xlabel('Physical time');ax.set_ylabel('Current loss, +1e-8 high-curvature perturbation');ax.legend();fig.tight_layout();fig.savefig(p/'height_sweep.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    opt=np.array(read(p/'setup.json')['basis']['optimum'])
    for ax,perturb in zip(axes,('zero','high_plus')):
        for r in (r for r in runs if r['height_label']=='original' and r['h']==.0125 and r['perturbation']==perturb):
            points=list(cache[r['run_id']].values());distance=[float(np.linalg.norm(np.array(x['x'])-opt)) for x in points]
            ax.plot([x['time'] for x in points],np.maximum(distance,1e-30),label='single' if r['layout']=='single' else f'joint {r["spectator_seed"]}',ls='-' if r['layout']=='single' else '--')
        ax.set_yscale('log');ax.set_title(perturb);ax.set_xlabel('Physical time');ax.set_ylabel('Distance from good parameter');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(p/'layout_comparison.png',dpi=160);plt.close(fig)


def map_analysis(p,runs,matrices):
    groups={}
    for m in matrices:groups.setdefault(m['cell_id'],[]).append(m)
    consistency=[]
    for cell,group in sorted(groups.items()):
        a,b=sorted(group,key=lambda m:m['epsilon'],reverse=True)
        if not a['available'] or not b['available']:
            consistency.append({'cell_id':cell,'passed':False,'reason':'Unavailable probe'});continue
        relative=float(np.linalg.norm(np.array(a['matrix'])-b['matrix'])/max(np.linalg.norm(b['matrix']),1e-12))
        difference=abs(a['spectral_radius']-b['spectral_radius'])
        consistency.append({'cell_id':cell,'kappa':b['kappa'],'height':b['height'],'product':b['product'],
             'matrix_relative_difference':relative,'radius_difference':difference,'passed':bool(relative<1e-3 and difference<1e-3),
             'spectral_radius':b['spectral_radius'],'screen':b['screen']})
    write(p/'probe_consistency.json',consistency)
    byscale=[]
    for kappa in sorted({r['kappa'] for r in consistency if 'kappa' in r}):
        rows=[r for r in consistency if r.get('kappa')==kappa and r['passed']]
        non=[r['product'] for r in rows if r['screen']=='nonexpanding'];amp=[r['product'] for r in rows if r['screen']=='amplifying']
        byscale.append({'kappa':kappa,'consistent_cells':len(rows),'largest_nonexpanding_product':max(non) if non else None,
                        'smallest_amplifying_product':min(amp) if amp else None})
    write(p/'cross_scale_summary.json',byscale)
    write(p/'rollout_summary.json',runs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kappa in sorted({r.get('kappa') for r in consistency if 'kappa' in r}):
        rows=sorted((r for r in consistency if r.get('kappa')==kappa),key=lambda r:r['product'])
        axes[0].plot([r['product'] for r in rows],[r['spectral_radius'] for r in rows],marker='o',label=f'curvature {kappa:g}')
    axes[0].axhline(1.01,ls='--',color='grey',lw=1);axes[0].set_xlabel('Curvature x clearance height');axes[0].set_ylabel('Measured local return amplification');axes[0].legend(fontsize=8)
    representative=4. if any(r['kappa']==4. for r in runs) else 1.
    for r in sorted((r for r in runs if r['kappa']==representative and r['initial_x']>0),key=lambda r:r['product']):
        raw=read(p/'return_traces'/(r['run_id']+'.json'));axes[1].plot([x['bounce'] for x in raw['rows']],[max(x['phase_norm']/abs(r['initial_x']),1e-12) for x in raw['rows']],label=f'product {r["product"]:g}')
    axes[1].set_yscale('log');axes[1].set_xlabel('Number of natural returns');axes[1].set_ylabel('Actual small-perturbation amplification');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(p/'return_amplification.png',dpi=160);plt.close(fig)


def analyze(output):
    p=Path(output);assert read(p/'audit.json')['passed'];runs=read(p/'runs.json');matrices=read(p/'return_matrices.json')
    rosen=[r for r in runs if r['kind']=='rosen_trajectory'];bowl=[r for r in runs if r['kind']=='return_rollout']
    if rosen:rosen_analysis(p,rosen)
    if matrices:map_analysis(p,bowl,matrices)
    print('Analysis complete:',len(rosen),'Rosenbrock trajectories;',len(bowl),'return rollouts;',len(matrices),'derived matrices')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');analyze(p.parse_args().output)
