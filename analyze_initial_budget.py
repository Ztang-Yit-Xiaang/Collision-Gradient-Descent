"""E24 fixed height/momentum factorial analysis; no tuning or proofs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from collision_optim.initial_budget import POLICIES,HEIGHT_MODES


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


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
                    'particle0_final_hits':sum(r['particle0_final_hit'] and not r['failed'] for r in group),
                    'particle0_harmed':sum(r['particle0_harmed'] for r in group),
                    'particle0_harmed_or_failed':sum(r['particle0_harmed'] or r['failed'] for r in group),
                    'median_final_value':median('final_value'),'median_best_value':median('best_value'),
                    'median_final_gap':median('relative_final_gap'),'median_particle0_max_rise':median('particle0_max_rise'),
                    'median_particle0_final_rise':median('particle0_final_rise'),'median_physical_time':median('physical_time'),
                    'mean_initial_clearance_potential':float(np.mean([r['initial_clearance_potential'] for r in group])) if 'initial_clearance_potential' in group[0] else None})
    return result


def contrasts(rows):
    out=[]
    pairs=[(h+'_transfer',h+'_keep') for h in HEIGHT_MODES]
    pairs += [('curve_transfer',h+'_transfer') for h in ('high','fixed_ref','fixed_safe')]
    for condition in ('good_present','bad_only'):
        for k in (1.,8.,32.):
            def branch(policy):return {r['seed']:r for r in rows if r['condition']==condition and r['stiffness']==k and r['policy']==policy}
            for candidate,reference in pairs:
                a,b=branch(candidate),branch(reference);assert a.keys()==b.keys()
                common={'condition':condition,'stiffness':k,'candidate':candidate,'reference':reference,'paired_seeds':len(a)}
                valid=[s for s in a if not a[s]['failed'] and not b[s]['failed']]
                for metric in ('particle0_max_rise','particle0_final_rise','final_value'):
                    d=[a[s][metric]-b[s][metric] for s in valid]
                    out.append({**common,'metric':metric,'better':sum(v< -1e-6 for v in d),'worse':sum(v>1e-6 for v in d),
                        'tie':sum(abs(v)<=1e-6 for v in d),'unavailable':len(a)-len(valid),
                        'median_paired_difference':float(np.median(d)) if d else None})
                for metric in ('final_hit','particle0_final_hit'):
                    switched=[(bool(a[s][metric] and not a[s]['failed']),bool(b[s][metric] and not b[s]['failed'])) for s in a]
                    out.append({**common,'metric':metric,'both':sum(x and y for x,y in switched),
                        'only_candidate':sum(x and not y for x,y in switched),'only_reference':sum(not x and y for x,y in switched),
                        'neither':sum(not x and not y for x,y in switched),'failures_as_unsuccessful':len(a)-len(valid)})
    return out


def screen(summary):
    checks=[]
    for k in (1.,8.,32.):
        def row(condition,policy):return next(r for r in summary if r['condition']==condition and r['stiffness']==k and r['policy']==policy)
        g0,g1=row('good_present','curve_keep'),row('good_present','curve_transfer')
        b0,b1=row('bad_only','curve_keep'),row('bad_only','curve_transfer')
        drop=(g0['particle0_harmed_or_failed']-g1['particle0_harmed_or_failed'])/g0['runs']
        change=b1['final_hits']-b0['final_hits']
        checks.append({'stiffness':k,'good_harm_rate_reduction':drop,'good_protection_screen':drop>=.25,
            'bad_final_hit_change':change,'bad_escape_screen':change>=-2,
            'baseline_bad_final_hits':b0['final_hits'],'exploration_sensitive':b0['final_hits']>=5})
    failures=sum(r['failures'] for r in summary)
    mechanical=sum(c['good_protection_screen'] for c in checks)>=2 and all(c['bad_escape_screen'] for c in checks) and failures==0
    sensitivity=all(c['exploration_sensitive'] for c in checks)
    return {'candidate':'curve_transfer','reference':'curve_keep','checks':checks,'solver_failures':failures,
        'mechanical_screen_passed':mechanical,'exploration_sensitive_all_stiffnesses':sensitivity,
        'joint_evidence_interpretable':mechanical and sensitivity,'statistical_noninferiority_established':False,
        'dynamical_collision_or_online_controller_evaluated':False,'curvature_necessity_established':False}


def plot(p,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.6));x=np.arange(3);width=.1
    for i,policy in enumerate(POLICIES):
        mode,motion=policy.rsplit('_',1)
        for ax,cond,key in zip(axes,('good_present','good_present','bad_only'),('particle0_harmed_or_failed','particle0_final_hits','final_hits')):
            rows=[next(r for r in summary if r['condition']==cond and r['stiffness']==k and r['policy']==policy) for k in (1.,8.,32.)]
            ax.bar(x+(i-3.5)*width,[r[key]/r['runs'] for r in rows],width,color=f'C{HEIGHT_MODES.index(mode)}',
                alpha=.45 if motion=='keep' else 1.,label=policy.replace('_',' '))
    for ax,title in zip(axes,('Good particle: transient loss rise > 1e-4','Good particle final target','Bad-only final target')):
        ax.set_title(title);ax.set_xticks(x,['1','8','32']);ax.set_xlabel('Transverse stiffness');ax.set_ylim(0,1.05)
        ax.set_ylabel('Fraction of20 paired seeds')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.tight_layout(rect=[0,.15,1,1]);fig.savefig(p/'initial_budget.png',dpi=160);plt.close(fig)


def analyze(output):
    p=Path(output);assert read(p/'audit.json')['passed'];runs=read(p/'runs.json');views=read(p/'common_time.json')
    summary=aggregate(runs);common=aggregate(views)
    for name,rows in (('summary',summary),('common_time_summary',common)):
        write(p/(name+'.json'),rows)
        with (p/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(p/'paired_contrasts.json',contrasts(runs));write(p/'common_time_contrasts.json',contrasts(views))
    for source,sums in ((runs,summary),(views,common)):
        for s in sums:
            g=[r for r in source if all(r[k]==s[k] for k in ('condition','stiffness','policy'))]
            assert len(g)==s['runs']
            assert sum(r['particle0_harmed'] or r['failed'] for r in g)==s['particle0_harmed_or_failed']
            assert sum(r['particle0_final_hit'] and not r['failed'] for r in g)==s['particle0_final_hits']
            assert sum(r['final_hit'] and not r['failed'] for r in g)==s['final_hits']
    counts=read(p/'execution_counts.json');contexts=read(p/'contexts.json')
    assert counts['actual_solver_queries']==20*len(contexts)+sum(r['actual_branch_calls'] for r in runs)
    assert counts['charged_policy_queries']-counts['actual_solver_queries']==44*len(contexts)
    for c in contexts:
        times=[v['physical_time'] for v in views if v['context_id']==c['context_id']]
        assert len(times)==8 and max(times)-min(times)<1e-12
    write(p/'analysis_validation.json',{'passed':True,'summary_groups_checked':len(summary),'common_time_groups_checked':len(common),
        'contexts_checked':len(contexts),'query_accounting_checked':True,'new_oracle_calls':0})
    if read(p/'config.json')['stage']=='formal':
        result=screen(summary);write(p/'joint_screen.json',result);plot(p,summary);print(json.dumps(result,indent=2))
    print(f'Analyzed {len(runs)} trajectories into {len(summary)} derived policy cells, no tuning.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');analyze(parser.parse_args().output)

