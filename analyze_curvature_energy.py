"""Prespecified E22 protection/escape/energy comparisons, no tuning."""
import argparse,csv,json
from pathlib import Path
import numpy as np

POLICIES=('fixed_high','fixed_low','cap_all','reserve_best','reserve_random')


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def aggregate(rows):
    result=[]
    for condition in ('good_present','bad_only'):
        for stiffness in (1.,8.,32.):
            for policy in POLICIES:
                group=[r for r in rows if r['condition']==condition and r['stiffness']==stiffness and r['policy']==policy]
                if not group:continue
                valid=[r for r in group if not r['failed']]
                def median(key):
                    data=[r[key] for r in valid if r.get(key) is not None]
                    return float(np.median(data)) if data else None
                result.append({'condition':condition,'stiffness':stiffness,'policy':policy,'runs':len(group),'valid':len(valid),
                    'failures':len(group)-len(valid),'final_hits':sum(r['final_hit'] and not r['failed'] for r in group),
                    'historical_hits_observed':sum(r['historical_hit'] for r in group),'ever_left_observed':sum(r['ever_left'] for r in group),
                    'particle0_harmed':sum(r['particle0_harmed'] for r in group),
                    'particle0_harmed_or_failed':sum(r['particle0_harmed'] or r['failed'] for r in group),
                    'median_final_value':median('final_value'),'median_best_value':median('best_value'),
                    'median_final_gap':median('relative_final_gap'),'median_particle0_max_rise':median('particle0_max_rise'),
                    'median_particle0_final_rise':median('particle0_final_rise'),'median_physical_time':median('physical_time'),
                    'mean_initial_clearance_potential':float(np.mean([r['initial_clearance_potential'] for r in group])) if 'initial_clearance_potential' in group[0] else None})
    return result


def contrasts(rows):
    result=[]
    for condition in ('good_present','bad_only'):
        for stiffness in (1.,8.,32.):
            candidates=[r for r in rows if r['condition']==condition and r['stiffness']==stiffness and r['policy']=='reserve_best']
            for reference in ('fixed_high','fixed_low','cap_all','reserve_random'):
                controls={r['seed']:r for r in rows if r['condition']==condition and r['stiffness']==stiffness and r['policy']==reference}
                for key in ('final_value','particle0_max_rise'):
                    pairs=[r[key]-controls[r['seed']][key] for r in candidates if r['seed'] in controls and not r['failed'] and not controls[r['seed']]['failed']]
                    result.append({'condition':condition,'stiffness':stiffness,'candidate':'reserve_best','reference':reference,'metric':key,
                         'better':sum(d< -1e-6 for d in pairs),'worse':sum(d>1e-6 for d in pairs),'tie':sum(abs(d)<=1e-6 for d in pairs),
                         'unavailable':len(candidates)-len(pairs),'median_paired_difference':float(np.median(pairs)) if pairs else None})
    return result


def screen(summary,failures):
    checks=[]
    for k in (1.,8.,32.):
        def row(condition,policy):return next(r for r in summary if r['condition']==condition and r['stiffness']==k and r['policy']==policy)
        good_high=row('good_present','fixed_high');good_best=row('good_present','reserve_best')
        bad_high=row('bad_only','fixed_high');bad_best=row('bad_only','reserve_best')
        drop=(good_high['particle0_harmed_or_failed']-good_best['particle0_harmed_or_failed'])/good_high['runs']
        checks.append({'stiffness':k,'good_harm_rate_reduction':drop,
                       'good_protection_screen':drop>=.25,
                       'bad_final_hit_change':bad_best['final_hits']-bad_high['final_hits'],
                       'bad_escape_screen':bad_best['final_hits']>=bad_high['final_hits']-2})
    return {'checks':checks,'promising_screen_passed':bool(sum(r['good_protection_screen'] for r in checks)>=2 and all(r['bad_escape_screen'] for r in checks) and failures==0),
            'criterion':'At least2/3 settings reduce harm>=.25, no setting loses>2/20 bad-only final hits, no solver failures. A screen, not a theorem.',
            'online_control_evaluated':False,'binary_collision_policy_evaluated':False}


def plot(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=('High','Low','All capped','Reserve best','Reserve random');fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(3);width=.15
    for j,(policy,label) in enumerate(zip(POLICIES,labels)):
        for ax,condition,key in [(axes[0],'good_present','particle0_harmed_or_failed'),(axes[1],'bad_only','final_hits')]:
            rows=[next(r for r in summary if r['condition']==condition and r['stiffness']==k and r['policy']==policy) for k in (1.,8.,32.)]
            ax.bar(x+(j-2)*width,[r[key]/r['runs'] for r in rows],width,label=label)
    axes[0].set_title('Good particle: transient loss rise > 1e-4 (lower is better)');axes[1].set_title('Bad-only final target (higher is better)')
    for ax in axes:
        ax.set_xticks(x,['1','8','32']);ax.set_xlabel('Transverse stiffness');ax.set_ylim(0,1.05);ax.set_ylabel('Fraction of20 paired seeds')
    handles,leg=axes[0].get_legend_handles_labels();fig.legend(handles,leg,loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(output/'protection_escape.png',dpi=160);plt.close(fig)


def analyze(output):
    p=Path(output);cfg=read(p/'config.json');assert read(p/'audit.json')['passed']
    runs=read(p/'runs.json');summary=aggregate(runs);common=aggregate(read(p/'common_time.json'))
    write(p/'summary.json',summary);write(p/'common_time_summary.json',common)
    for name,rows in [('summary',summary),('common_time_summary',common)]:
        with (p/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(p/'paired_contrasts.json',contrasts(runs));write(p/'common_time_contrasts.json',contrasts(read(p/'common_time.json')))
    if cfg['stage'] in ('formal','kinetic'):
        s=screen(summary,sum(r['failed'] for r in runs))
        if cfg['stage']=='kinetic':
            sensitive=[{'stiffness':k,'fixed_high_final_hits':next(r['final_hits'] for r in summary if r['condition']=='bad_only' and r['policy']=='fixed_high' and r['stiffness']==k)} for k in (1.,8.,32.)]
            s['exploration_sensitivity']=sensitive
            s['joint_evidence_interpretable']=bool(s['promising_screen_passed'] and all(r['fixed_high_final_hits']>=5 for r in sensitive))
            s['additional_gate']='New follow-up only: baseline>=5/20 final hits in every stiffness; otherwise original non-inferiority check is not sufficient.'
        write(p/'joint_screen.json',s);print(json.dumps(s,indent=2))
        plot(p,summary)
    print('Analysis complete:',len(runs),'executed trajectories;',len(summary),'derived policy summaries')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');analyze(p.parse_args().output)
