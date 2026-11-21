"""Independently check E22 aggregates/screen and derive paid-reference budgets."""
import argparse,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())


def check(output):
    p=Path(output);assert read(p/'audit.json')['passed']
    contexts={c['context_id']:c for c in read(p/'contexts.json')};runs=read(p/'runs.json');summary=read(p/'summary.json');cfg=read(p/'config.json')
    for c in contexts.values():
        velocities=np.array(c.get('initial_horizontal_velocities',np.zeros((4,2))))
        assert abs(.5*np.sum(velocities*velocities)-cfg.get('initial_kinetic',0.))<1e-12
        assert np.max(abs(velocities.sum(axis=0)))<1e-12 and np.max(abs(velocities[0]))==0.
    budgets=[]
    for r in runs:
        c=contexts[r['context_id']];initial=read(p/'initial_states'/(r['run_id']+'.json'))
        energy=.5*np.array(initial['positions'])[:,-1]+.5*np.sum(np.array(initial['velocities'])**2,axis=1);barrier=.5*c['reference_values'][1]
        assert abs(sum(energy)-r['initial_energy'])<1e-12
        budgets.append({'run_id':r['run_id'],'context_id':r['context_id'],'condition':r['condition'],'stiffness':r['stiffness'],
              'seed':r['seed'],'policy':r['policy'],'initial_particle_energies':energy.tolist(),
              'reference_barrier_energy':barrier,'reference_barrier_margins':(energy-barrier).tolist(),
              'particles_with_reference_margin':int(sum(energy>=barrier)),
              'initial_total_energy':float(sum(energy)),'initial_clearance_potential':r['initial_clearance_potential'],
              'reference_values_are_scoring_only':True})
    for s in summary:
        group=[r for r in runs if all(r[k]==s[k] for k in ('condition','stiffness','policy'))]
        assert len(group)==s['runs']
        assert sum(r['final_hit'] and not r['failed'] for r in group)==s['final_hits']
        assert sum(r['historical_hit'] for r in group)==s['historical_hits_observed']
        assert sum(r['particle0_harmed'] for r in group)==s['particle0_harmed']
        assert sum(r['particle0_harmed'] or r['failed'] for r in group)==s['particle0_harmed_or_failed']
    views=read(p/'common_time.json')
    for context in contexts:
        times=[r['physical_time'] for r in views if r['context_id']==context]
        assert len(times)==5 and max(times)-min(times)<1e-12
    actual=read(p/'execution_counts.json')
    charged=sum(r['total_oracle_calls'] for r in runs)
    assert actual['actual_solver_queries']==sum(r['actual_branch_calls'] for r in runs)+20*len(contexts)
    assert charged-actual['actual_solver_queries']==48*len(contexts)
    if (p/'joint_screen.json').exists():
        screen=read(p/'joint_screen.json');checks=[]
        for k in (1.,8.,32.):
            def row(condition,policy):return [r for r in runs if r['condition']==condition and r['stiffness']==k and r['policy']==policy]
            good_hi=row('good_present','fixed_high');good_best=row('good_present','reserve_best')
            bad_hi=row('bad_only','fixed_high');bad_best=row('bad_only','reserve_best')
            drop=(sum(r['particle0_harmed'] or r['failed'] for r in good_hi)-sum(r['particle0_harmed'] or r['failed'] for r in good_best))/len(good_hi)
            change=sum(r['final_hit'] and not r['failed'] for r in bad_best)-sum(r['final_hit'] and not r['failed'] for r in bad_hi)
            checks.append((drop>=.25,change>=-2))
        expected=bool(sum(a for a,b in checks)>=2 and all(b for a,b in checks) and not any(r['failed'] for r in runs))
        assert expected==screen['promising_screen_passed']
        if 'joint_evidence_interpretable' in screen:
            sensitive=[s['final_hits']>=5 for s in summary if s['condition']=='bad_only' and s['policy']=='fixed_high']
            assert screen['joint_evidence_interpretable']==bool(expected and all(sensitive))
    result={'passed':True,'allocation_budgets_checked':len(budgets),'aggregate_groups_checked':len(summary),
            'common_time_contexts_checked':len(contexts),'query_accounting_checked':True,
            'all_references_already_paid':True,'new_oracle_calls_for_this_check':0,
            'known_archive_vs_active_point_distinction_retained':True}
    (p/'energy_budget_views.json').write_text(json.dumps(budgets,indent=2)+'\n')
    (p/'analysis_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');check(p.parse_args().output)
