"""Seal E24 executions, allocation ledgers and independent screen checks."""
import hashlib,json,zipfile
import statistics
from pathlib import Path
from finalize_energy_generalization import physical_ledgers


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def create(p,data):
    with p.open('x') as f:f.write(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def allocation_ledger(root,runs):
    entries=[]
    for r in runs:
        state=read(root/'initial_states'/(r['run_id']+'.json'));a=state['allocation'];j=a['selected_index'];height=a['heights'][j]
        assert abs(a['total_kinetic_after']-a['total_kinetic_before'])<1e-12
        assert sum((x-y)**2 for x,y in zip(a['linear_momentum_before'],a['linear_momentum_after']))<1e-24
        entries.append({'run_id':r['run_id'],'condition':r['condition'],'stiffness':r['stiffness'],'seed':r['seed'],
            'policy':r['policy'],'selected_index':j,'selected_height':height,
            'selected_K_before':a['selected_kinetic_before'],'selected_K_after':a['selected_kinetic_after'],
            'selected_excess_energy_before_velocity_allocation':a['selected_kinetic_before']+.5*height,
            'selected_excess_energy_after_velocity_allocation':a['selected_kinetic_after']+.5*height,
            'total_K_before':a['total_kinetic_before'],'total_K_after':a['total_kinetic_after'],
            'linear_P_before':a['linear_momentum_before'],'linear_P_after':a['linear_momentum_after'],
            'angular_before_at_allocated_positions':a['angular_before_velocity_allocation'],
            'angular_after_at_allocated_positions':a['angular_after_velocity_allocation'],
            'angular_change_at_same_positions':a['angular_change_at_same_positions'],
            'runtime_collision_or_reset':False,'new_oracle_calls':0})
    create(root/'allocation_ledger.json',entries)


def check_screen(root,runs):
    screen=read(root/'joint_screen.json');good=0;sensitive=True;escape=True
    for c in screen['checks']:
        k=c['stiffness']
        def branch(condition,policy):return [r for r in runs if r['stiffness']==k and r['condition']==condition and r['policy']==policy]
        gh,gb=branch('good_present','curve_keep'),branch('good_present','curve_transfer')
        bh,bb=branch('bad_only','curve_keep'),branch('bad_only','curve_transfer')
        drop=(sum(r['particle0_harmed'] or r['failed'] for r in gh)-sum(r['particle0_harmed'] or r['failed'] for r in gb))/len(gh)
        hi=sum(r['final_hit'] and not r['failed'] for r in bh)
        change=sum(r['final_hit'] and not r['failed'] for r in bb)-hi
        assert abs(drop-c['good_harm_rate_reduction'])<1e-12 and change==c['bad_final_hit_change']
        assert (drop>=.25)==c['good_protection_screen'] and (change>=-2)==c['bad_escape_screen'] and (hi>=5)==c['exploration_sensitive']
        good+=drop>=.25;sensitive &= hi>=5;escape &= change>=-2
    mechanical=good>=2 and escape and not any(r['failed'] for r in runs)
    assert mechanical==screen['mechanical_screen_passed']
    assert sensitive==screen['exploration_sensitive_all_stiffnesses']
    assert (mechanical and sensitive)==screen['joint_evidence_interpretable']
    create(root/'independent_screen_validation.json',{'passed':True,'stiffnesses_checked':3,'new_oracle_calls':0})


def selection_diagnostics(root,runs):
    """Post hoc descriptive grouping only; no fitted/selected controller."""
    rows=[]
    for k in (1.,8.,32.):
        a={r['seed']:r for r in runs if r['condition']=='bad_only' and r['stiffness']==k and r['policy']=='curve_keep'}
        b={r['seed']:r for r in runs if r['condition']=='bad_only' and r['stiffness']==k and r['policy']=='curve_transfer'}
        for selected0 in (True,False):
            seeds=[s for s in a if (a[s]['selected_index']==0)==selected0]
            rows.append({'stiffness':k,'selected_template_particle0':selected0,'seeds':len(seeds),
                'median_selected_initial_K':statistics.median(a[s]['selected_kinetic_before'] for s in seeds) if seeds else None,
                'only_keep_final_success':sum(a[s]['final_hit'] and not b[s]['final_hit'] for s in seeds),
                'only_transfer_final_success':sum(not a[s]['final_hit'] and b[s]['final_hit'] for s in seeds)})
    create(root/'exploratory_selection_breakdown.json',{'role':'Post hoc description after primary outcomes; not independent confirmation or controller fit',
        'new_trajectories':0,'new_oracle_calls':0,'rows':rows})


def finalize():
    roots=[Path('output/collision_initial_budget_preflight'),Path('output/collision_initial_budget')]
    totals={};cost={'posthoc_value_calls':0,'posthoc_gradient_calls':0};ledgers={};failures=[]
    for p,expected in zip(roots,(48,960)):
        cfg=read(p/'config.json');counts=read(p/'execution_counts.json');runs=read(p/'runs.json');audit=read(p/'audit.json')
        assert audit['passed'] and read(p/'analysis_validation.json')['passed']
        assert len(runs)==expected==counts['trajectory_executions']
        assert counts['charged_policy_queries']-counts['actual_solver_queries']==44*counts['contexts']
        ledgers[str(p)]=physical_ledgers(p,runs);allocation_ledger(p,runs)
        for key,n in counts.items():
            if key!='independent_seed_groups':totals[key]=totals.get(key,0)+n
        for key in cost:cost[key]+=audit[key]
        failures.extend({'folder':str(p),'run_id':r['run_id'],'reason':r['stop_reason']} for r in runs if r['failed'])
        if cfg['stage']=='formal':
            assert all(r['total_oracle_calls']==6000 for r in runs if not r['failed'])
            check_screen(p,runs)
            selection_diagnostics(p,runs)
            create(p/'verification.json',{'date':'2026-11-10','full_tests':280,'test_seconds':32.30,'role':'Initialization ablation, not dynamic collision/controller',
                'audit':audit,'analysis_validation':read(p/'analysis_validation.json')})
        assert sha(p/'source.zip')==cfg['source_archive_sha256']
        with zipfile.ZipFile(p/'source.zip') as z:
            for n,h in read(p/'source_manifest.json').items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        files=[p/name for name in ('config.json','contexts.json','runs.json','runs.csv','execution_counts.json','source.zip','source_manifest.json')]
        files.extend(q for q in p.rglob('*') if q.is_file() and q.parent!=p)
        files.extend(p/name for name in ('source_consistency.json','completion_observation.json') if (p/name).exists())
        create(p/'raw_artifact_manifest.json',{'files':{str(q.relative_to(p)):{'sha256':sha(q),'bytes':q.stat().st_size} for q in sorted(files)}})
        print(f'Sealed {p}: {len(runs)} trajectories/{len(files)} raw files',flush=True)
    assert totals['trajectory_executions']==1008 and totals['contexts']==126
    completion={'passed':True,'date':'2026-11-10','counts':totals,'solver_failures':failures,'physical_ledgers':ledgers,
        'allocation_ledgers':1008,'audit_oracle_costs_separate_from_solver':cost,
        'formal_independent_seed_groups':20,'preflight_seed_groups_excluded_from_efficacy':1,
        'no_tuning_or_extra_parameter_search':True,'dynamic_binary_collisions':False,'online_control':False,
        'angular_conservation_claimed_for_initial_velocity_map':False,'new_neural_training_runs':0,
        'joint_screen':read(roots[1]/'joint_screen.json')}
    sources=[Path(n) for n in (
        'src/collision_optim/initial_budget.py','src/collision_optim/initial_budget_experiment.py','tests/test_initial_budget.py',
        'run_initial_budget.py','audit_initial_budget.py','analyze_initial_budget.py','finalize_initial_budget.py','register_initial_budget.py',
        'src/collision_optim/energy_generalization.py','src/collision_optim/curvature_energy.py','finalize_energy_generalization.py',
        'register_energy_generalization.py','register_curvature_energy.py','register_orbit_stability.py','register_events.py',
        'register_timing.py','register_handoff.py','refresh_experiment_inventory.py',
        'docs/superpowers/specs/2026-11-10-initial-budget-design.md','docs/superpowers/plans/2026-11-10-initial-budget-plan.md',
        'docs/collision_initial_budget_results.md','docs/experiment_log.md','README.md','AGENTS.md')]
    for p in roots:
        create(p/'completion_validation.json',completion)
        archive=p/'postprocessing_sources.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for f in sources:z.write(f,str(f))
        create(p/'postprocessing_sources.json',{'date':'2026-11-10','archive':str(archive),'archive_sha256':sha(archive),
            'files':{str(f):sha(f) for f in sources},'launch_command':'python3 finalize_initial_budget.py',
            'note':'Final audits/ledgers/report sources. Original launch source.zip unchanged.'})
    print(json.dumps(completion,indent=2))


if __name__=='__main__':finalize()
