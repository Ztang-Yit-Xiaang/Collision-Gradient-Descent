"""E23 frozen per-regime comparisons, derived from audited raw outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_curvature_energy import aggregate as parent_aggregate

POLICIES=('fixed_high','fixed_low','cap_all','reserve_best','reserve_random')
SCALES=(.5,1.,2.)
ENERGIES=(0.,1e-5,1e-3)


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def aggregate(rows):
    summaries=[]
    for a in SCALES:
        for e in ENERGIES:
            subset=[r for r in rows if r['longitudinal_scale']==a and r['incumbent_energy']==e]
            for s in parent_aggregate(subset):
                group=[r for r in subset if r['condition']==s['condition'] and r['policy']==s['policy']]
                s.update(longitudinal_scale=a,incumbent_energy=e,
                    particle0_final_hits=sum(r['particle0_final_hit'] and not r['failed'] for r in group))
                summaries.append(s)
    return summaries


def comparisons(rows):
    result=[]
    for a in SCALES:
        for e in ENERGIES:
            for condition in ('good_present','bad_only'):
                def group(policy):
                    return {r['seed']:r for r in rows if r['longitudinal_scale']==a and r['incumbent_energy']==e
                            and r['condition']==condition and r['policy']==policy}
                best=group('reserve_best')
                for policy in ('fixed_high','fixed_low','cap_all','reserve_random'):
                    ref=group(policy);assert ref.keys()==best.keys()
                    available=[s for s in best if not best[s]['failed'] and not ref[s]['failed']]
                    for metric in ('particle0_max_rise','particle0_final_rise','final_value'):
                        differences=[best[s][metric]-ref[s][metric] for s in available]
                        result.append({'condition':condition,'longitudinal_scale':a,'incumbent_energy':e,
                            'reference':policy,'candidate':'reserve_best','metric':metric,'paired_seeds':len(best),
                            'unavailable':len(best)-len(available),'better':sum(d< -1e-6 for d in differences),
                            'worse':sum(d>1e-6 for d in differences),'tie':sum(abs(d)<=1e-6 for d in differences),
                            'median_paired_difference':float(np.median(differences)) if differences else None})
                    result.append({'condition':condition,'longitudinal_scale':a,'incumbent_energy':e,
                        'reference':policy,'candidate':'reserve_best','metric':'final_hit','paired_seeds':len(best),
                        'both_final_hit':sum(best[s]['final_hit'] and ref[s]['final_hit'] and s in available for s in best),
                        'only_candidate_final_hit':sum(best[s]['final_hit'] and not ref[s]['final_hit'] and s in available for s in best),
                        'only_reference_final_hit':sum(not best[s]['final_hit'] and ref[s]['final_hit'] and s in available for s in best),
                        'neither_final_hit':sum(not best[s]['final_hit'] and not ref[s]['final_hit'] and s in available for s in best),
                        'unavailable':len(best)-len(available)})
    return result


def screens(summaries):
    result=[]
    for e in ENERGIES:
        checks=[]
        for a in SCALES:
            def get(condition,policy):
                return next(s for s in summaries if s['incumbent_energy']==e and s['longitudinal_scale']==a
                            and s['condition']==condition and s['policy']==policy)
            gh,gb=get('good_present','fixed_high'),get('good_present','reserve_best')
            bh,bb=get('bad_only','fixed_high'),get('bad_only','reserve_best')
            drop=(gh['particle0_harmed_or_failed']-gb['particle0_harmed_or_failed'])/gh['runs']
            change=bb['final_hits']-bh['final_hits']
            checks.append({'longitudinal_scale':a,'good_harm_rate_reduction':drop,'good_protection_screen':drop>=.25,
                'bad_final_hit_change':change,'bad_escape_screen':change>=-2,'fixed_high_bad_final_hits':bh['final_hits'],
                'exploration_sensitive':bh['final_hits']>=5})
        fail=sum(s['failures'] for s in summaries if s['incumbent_energy']==e)
        mechanical=sum(c['good_protection_screen'] for c in checks)>=2 and all(c['bad_escape_screen'] for c in checks) and fail==0
        sensitive=all(c['exploration_sensitive'] for c in checks)
        result.append({'incumbent_energy':e,'checks':checks,'failures':fail,'mechanical_screen_passed':mechanical,
            'exploration_sensitive_all_scales':sensitive,'joint_evidence_interpretable':mechanical and sensitive,
            'statistical_noninferiority_established':False})
    return result


def plot(p,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharey=True)
    labels=('High','Low','All capped','Reserve best','Reserve random');x=np.arange(3);width=.15
    for i,e in enumerate(ENERGIES):
        for j,(condition,key,title) in enumerate((('good_present','particle0_harmed_or_failed','Good particle: transient loss rise > 1e-4'),
                ('good_present','particle0_final_hits','Good particle final target'),('bad_only','final_hits','Bad-only final target'))):
            ax=axes[i,j]
            for t,(policy,label) in enumerate(zip(POLICIES,labels)):
                rows=[next(s for s in summary if s['incumbent_energy']==e and s['longitudinal_scale']==a
                    and s['condition']==condition and s['policy']==policy) for a in SCALES]
                ax.bar(x+(t-2)*width,[s[key]/s['runs'] for s in rows],width,label=label)
            ax.set_title(title+f'\nParticle0 initial K={e:g}',fontsize=10)
            ax.set_xticks(x,[str(a) for a in SCALES]);ax.set_xlabel('Longitudinal scale');ax.set_ylim(0,1.05)
            if j==0:ax.set_ylabel('Fraction of20 paired seeds')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(p/'generalization.png',dpi=160);plt.close(fig)


def analyze(output):
    p=Path(output);assert read(p/'audit.json')['passed'];runs=read(p/'runs.json');views=read(p/'common_time.json')
    summary=aggregate(runs);common=aggregate(views)
    for name,rows in (('summary',summary),('common_time_summary',common)):
        write(p/(name+'.json'),rows)
        with (p/(name+'.csv')).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(p/'paired_contrasts.json',comparisons(runs));write(p/'common_time_contrasts.json',comparisons(views))
    # Independent count recomputation from the saved records, not extra oracle queries.
    for source,sums in ((runs,summary),(views,common)):
        for s in sums:
            g=[r for r in source if all(r[k]==s[k] for k in ('condition','longitudinal_scale','incumbent_energy','policy'))]
            assert len(g)==s['runs'] and sum(r['particle0_final_hit'] and not r['failed'] for r in g)==s['particle0_final_hits']
            assert sum(r['final_hit'] and not r['failed'] for r in g)==s['final_hits']
            assert sum(r['particle0_harmed'] or r['failed'] for r in g)==s['particle0_harmed_or_failed']
    counts=read(p/'execution_counts.json');contexts=read(p/'contexts.json')
    assert counts['actual_solver_queries']==20*len(contexts)+sum(r['actual_branch_calls'] for r in runs)
    assert counts['charged_policy_queries']-counts['actual_solver_queries']==48*len(contexts)
    for c in contexts:
        ts=[r['physical_time'] for r in views if r['context_id']==c['context_id']]
        assert len(ts)==5 and max(ts)-min(ts)<1e-12
    write(p/'analysis_validation.json',{'passed':True,'summary_groups_checked':len(summary),'common_time_groups_checked':len(common),
          'contexts_checked':len(contexts),'query_accounting_checked':True,'new_oracle_calls':0})
    if read(p/'config.json')['stage']=='formal':
        screening=screens(summary);write(p/'joint_screens.json',screening);plot(p,summary)
        print(json.dumps(screening,indent=2))
    print(f'Analyzed {len(runs)} actual trajectories; {len(summary)} derived cells; no fitting/tuning.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');analyze(parser.parse_args().output)
