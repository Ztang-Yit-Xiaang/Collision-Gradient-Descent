"""Seal E23 evidence, keeping executions separate from derived comparisons."""
import hashlib
import csv
import gzip
import json
from pathlib import Path
import zipfile


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def create(p,obj):
    with p.open('x') as f:f.write(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def physical_ledgers(root,runs):
    """Export committed-state energy accounting without any oracle evaluations."""
    result=[]
    for run in runs:
        with gzip.open(root/'traces'/(run['run_id']+'.csv.gz'),'rt') as stream:
            rows=list(csv.DictReader(stream))
        initial=read(root/'initial_states'/(run['run_id']+'.json'))
        last=rows[-1];q=json.loads(last['positions']);v=json.loads(last['velocities'])
        K0=.5*sum(w*w for vector in initial['velocities'] for w in vector)
        K1=.5*sum(w*w for vector in v for w in vector)
        E0=float(rows[0]['energy']);E1=float(last['energy'])
        heat=-sum(float(r['damping_energy']) for r in rows[1:])
        impulse_energy=sum(float(r['impulse_energy']) for r in rows[1:])
        gravity_work=-.5*sum(b[-1]-a[-1] for a,b in zip(initial['positions'],q))
        residual=E1-E0+heat-impulse_energy
        assert abs(residual)<1e-8 and abs(K1-K0-gravity_work+heat-impulse_energy)<1e-8
        result.append({'run_id':run['run_id'],'condition':run['condition'],'longitudinal_scale':run['longitudinal_scale'],
            'incumbent_energy':run['incumbent_energy'],'seed':run['seed'],'policy':run['policy'],
            'initial_energy':E0,'final_energy':E1,'initial_kinetic':K0,'final_kinetic':K1,
            'heat_removed_by_damping':heat,'gravity_work':gravity_work,'impulse_energy':impulse_energy,
            'energy_balance_residual':residual,'post_initialization_control_supply':0.,'binary_interventions':0,
            'physical_time':run['physical_time'],'committed_steps':len(rows)-1,
            'scope':'Committed trajectory; partial-step oracle costs paid but unavailable intermediate physical states excluded',
            'wall_and_gravity_momentum':'Separately reconstructed in raw audit; saved events/traces retain individual impulses'})
    create(root/'physical_ledgers.json',{'new_oracle_calls':0,'runs':result})
    return {'runs':len(result),'total_heat_removed':sum(r['heat_removed_by_damping'] for r in result),
            'max_energy_balance_residual':max(abs(r['energy_balance_residual']) for r in result)}


def finalize():
    roots=[Path('output/collision_energy_generalization_preflight'),Path('output/collision_energy_generalization')]
    totals={};audits={'posthoc_value_calls':0,'posthoc_gradient_calls':0};failures=[];ledgers={}
    for p,expected in zip(roots,(90,1800)):
        cfg=read(p/'config.json');counts=read(p/'execution_counts.json');runs=read(p/'runs.json');audit=read(p/'audit.json')
        assert audit['passed'] and read(p/'analysis_validation.json')['passed']
        assert len(runs)==expected==counts['trajectory_executions']
        assert counts['charged_policy_queries']-counts['actual_solver_queries']==48*counts['contexts']
        ledgers[str(p)]=physical_ledgers(p,runs)
        for key,n in counts.items():
            if key!='independent_seed_groups':totals[key]=totals.get(key,0)+n
        for key in audits:audits[key]+=audit[key]
        failures.extend({'folder':str(p),'run_id':r['run_id'],'reason':r['stop_reason']} for r in runs if r['failed'])
        if cfg['stage']=='formal':
            assert all(r['total_oracle_calls']==6000 for r in runs if not r['failed'])
            checked=[]
            for screen in read(p/'joint_screens.json'):
                e=screen['incumbent_energy'];good=0;sensitive=True;escape=True
                for item in screen['checks']:
                    a=item['longitudinal_scale']
                    def branch(condition,policy):
                        return [r for r in runs if r['incumbent_energy']==e and r['longitudinal_scale']==a
                                and r['condition']==condition and r['policy']==policy]
                    gh,gb=branch('good_present','fixed_high'),branch('good_present','reserve_best')
                    bh,bb=branch('bad_only','fixed_high'),branch('bad_only','reserve_best')
                    drop=(sum(r['particle0_harmed'] or r['failed'] for r in gh)-sum(r['particle0_harmed'] or r['failed'] for r in gb))/len(gh)
                    hi=sum(r['final_hit'] and not r['failed'] for r in bh)
                    change=sum(r['final_hit'] and not r['failed'] for r in bb)-hi
                    assert abs(drop-item['good_harm_rate_reduction'])<1e-12 and change==item['bad_final_hit_change']
                    assert (hi>=5)==item['exploration_sensitive'] and (drop>=.25)==item['good_protection_screen']
                    good+=drop>=.25;sensitive &= hi>=5;escape &= change>=-2
                failed=any(r['failed'] for r in runs if r['incumbent_energy']==e)
                mechanical=good>=2 and escape and not failed
                assert mechanical==screen['mechanical_screen_passed']
                assert sensitive==screen['exploration_sensitive_all_scales']
                assert (mechanical and sensitive)==screen['joint_evidence_interpretable']
                checked.append(e)
            create(p/'independent_screen_validation.json',{'passed':True,'kinetic_levels_checked':checked,'new_oracle_calls':0})
            create(p/'verification.json',{'date':'2026-11-10','full_tests':272,'test_seconds':21.46,
                'role':'Fixed-policy synthetic robustness/energy allocation; no neural training or binary interactions',
                'audit':audit,'analysis_validation':read(p/'analysis_validation.json')})
        assert sha(p/'source.zip')==cfg['source_archive_sha256']
        with zipfile.ZipFile(p/'source.zip') as z:
            for name,h in read(p/'source_manifest.json').items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        raw=[p/name for name in ('config.json','contexts.json','runs.json','runs.csv','execution_counts.json','source.zip','source_manifest.json')]
        raw.extend(q for q in p.rglob('*') if q.is_file() and q.parent!=p)
        raw.extend(p/name for name in ('audit_failure.json','failed_auditor_source.py','source_consistency.json','ledger_export_failure.json','failed_ledger_source.py') if (p/name).exists())
        create(p/'raw_artifact_manifest.json',{'date':'2026-11-10','files':{
            str(q.relative_to(p)):{'sha256':sha(q),'bytes':q.stat().st_size} for q in sorted(raw)}})
        print(f'Sealed {p}: {len(runs)} trajectories, {len(raw)} raw files',flush=True)
    assert totals['trajectory_executions']==1890 and totals['contexts']==378
    completion={'passed':True,'date':'2026-11-10','counts':totals,'solver_failures':failures,
        'physical_ledger_exports':ledgers,
        'audit_oracle_calls_separate_from_solver':audits,'failed_audit_partial_cost':'Unknown, retained audit_failure.json',
        'formal_independent_seed_groups':20,'preflight_seed_groups_excluded_from_efficacy':1,
        'no_tuning_or_extra_amplitude_search':True,'binary_interactions':False,'online_energy_controller':False,
        'neural_training_executions':0,'joint_screens_per_kinetic_level':read(roots[1]/'joint_screens.json'),
        'statistical_noninferiority_established':False}
    files=[Path(f) for f in (
        'src/collision_optim/energy_generalization.py','src/collision_optim/energy_generalization_experiment.py',
        'tests/test_energy_generalization.py','scripts/run_energy_generalization.py','scripts/audit_energy_generalization.py',
        'scripts/analyze_energy_generalization.py','scripts/register_energy_generalization.py','scripts/finalize_energy_generalization.py',
        'src/collision_optim/curvature_energy.py','scripts/analyze_curvature_energy.py','scripts/register_curvature_energy.py',
        'scripts/register_orbit_stability.py','scripts/register_events.py','scripts/register_timing.py','scripts/register_handoff.py','scripts/refresh_experiment_inventory.py',
        'docs/superpowers/specs/2026-11-10-energy-generalization-design.md',
        'docs/superpowers/plans/2026-11-10-energy-generalization-plan.md',
        'docs/collision_energy_generalization_results.md','docs/experiment_log.md','README.md','AGENTS.md')]
    for p in roots:
        create(p/'completion_validation.json',completion)
        archive=p/'postprocessing_sources.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for f in files:z.write(f,str(f))
        create(p/'postprocessing_sources.json',{'date':'2026-11-10','archive':str(archive),'archive_sha256':sha(archive),
            'files':{str(f):sha(f) for f in files},'launch_command':'python3 scripts/finalize_energy_generalization.py',
            'note':'Final audits/report versions; original launch source archives and failed preflight auditor preserved.'})
    print(json.dumps(completion,indent=2))


if __name__=='__main__':finalize()
