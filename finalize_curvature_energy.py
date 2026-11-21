"""Seal E22 derived evidence without replacing any execution artifacts.

Run after all four audits/analyses/checks, before register_curvature_energy.py.
This is bookkeeping, not another experiment or optimizer evaluation.
"""
import hashlib
import json
from pathlib import Path
import zipfile


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def finalize():
    names = ['collision_curvature_energy_preflight', 'collision_curvature_energy',
             'collision_curvature_energy_kinetic_preflight', 'collision_curvature_energy_kinetic']
    totals = {}
    audit_calls = {'posthoc_value_calls': 0, 'posthoc_gradient_calls': 0}
    paired = []
    for name in names:
        root = Path('output') / name
        audit = read(root / 'audit.json')
        assert audit['passed'] and read(root / 'analysis_validation.json')['passed']
        cfg = read(root / 'config.json')
        runs = read(root / 'runs.json')
        counts = read(root / 'execution_counts.json')
        assert not any(r['failed'] for r in runs)
        assert len(runs) == counts['trajectory_executions']
        for key, value in counts.items():
            if key != 'independent_seed_groups':
                totals[key] = totals.get(key, 0) + value
        for key in audit_calls:
            audit_calls[key] += audit[key]
        if cfg['stage'] in ('formal', 'kinetic'):
            assert len(runs) == 600 and all(r['total_oracle_calls'] == 6000 for r in runs)
            for stiffness in (1., 8., 32.):
                def branch(policy):
                    return {r['seed']: r for r in runs if r['condition'] == 'bad_only'
                            and r['stiffness'] == stiffness and r['policy'] == policy}
                high, best = branch('fixed_high'), branch('reserve_best')
                assert high.keys() == best.keys() and len(high) == 20
                paired.append({'stage': cfg['stage'], 'stiffness': stiffness, 'paired_seeds': 20,
                               'both_final_hit': sum(high[s]['final_hit'] and best[s]['final_hit'] for s in high),
                               'only_reserve_best_final_hit': sum(not high[s]['final_hit'] and best[s]['final_hit'] for s in high),
                               'only_fixed_high_final_hit': sum(high[s]['final_hit'] and not best[s]['final_hit'] for s in high),
                               'neither_final_hit': sum(not high[s]['final_hit'] and not best[s]['final_hit'] for s in high)})
            create_json(root / 'verification.json', {
                'date': '2026-11-10', 'full_tests': 264 if cfg['stage'] == 'kinetic' else 262,
                'test_seconds': 14.79 if cfg['stage'] == 'kinetic' else 12.86,
                'role': 'Separate synthetic initialization regime; not neural training or interacting balls',
                'audit': audit, 'analysis_validation': read(root / 'analysis_validation.json')})
        # Execution source is immutable. Validate archive even if current postprocessing evolved.
        with zipfile.ZipFile(root / 'source.zip') as archive:
            for path, digest in read(root / 'source_manifest.json').items():
                assert hashlib.sha256(archive.read(path)).hexdigest() == digest
        assert sha(root / 'source.zip') == cfg['source_archive_sha256']
        raw = [root / f for f in ('config.json', 'contexts.json', 'runs.json', 'runs.csv',
                                   'execution_counts.json', 'source.zip', 'source_manifest.json')]
        raw += sorted(p for p in root.rglob('*') if p.is_file() and p.parent != root)
        create_json(root / 'raw_artifact_manifest.json', {
            'date': '2026-11-10', 'role': 'Immutable execution records, not extra runs',
            'files': {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in raw}})
        print(f'{name}: sealed {len(runs)} trajectories and {len(raw)} raw files', flush=True)
    assert totals['trajectory_executions'] == 1260 and totals['contexts'] == 252
    assert totals['actual_solver_queries'] == 7255240
    assert totals['actual_queries_including_setup'] == 7255996
    assert totals['charged_policy_queries'] - totals['actual_solver_queries'] == 48 * 252
    assert not read(Path('output/collision_curvature_energy/joint_screen_interpretation.json'))['joint_efficacy_established']
    assert read(Path('output/collision_curvature_energy_kinetic/joint_screen.json'))['joint_evidence_interpretable']
    completion = {
        'passed': True, 'date': '2026-11-10', 'counts': totals, 'audit_oracle_calls_separate_from_solver': audit_calls,
        'paired_final_success_tables_derived_not_new_runs': paired,
        'formal_regimes_not_pooled': True, 'formal_seed_groups_per_regime': 20,
        'preflights_excluded_from_efficacy': True, 'binary_interactions_evaluated': False,
        'online_control_evaluated': False, 'new_neural_training_runs': 0,
        'statistical_noninferiority_established': False,
        'note': 'Kinetic follow-up passes descriptive screen; common-time exploration remains mixed. '
                'Good incumbent has zero horizontal initial velocity. No further speed search performed.'}
    files = [Path(f) for f in (
        'src/collision_optim/curvature_energy.py', 'src/collision_optim/curvature_energy_experiment.py',
        'tests/test_curvature_energy.py', 'run_curvature_energy.py', 'audit_curvature_energy.py',
        'analyze_curvature_energy.py', 'check_curvature_energy.py', 'finalize_curvature_energy.py',
        'register_curvature_energy.py', 'register_orbit_stability.py', 'register_events.py',
        'register_timing.py', 'register_handoff.py', 'refresh_experiment_inventory.py',
        'docs/superpowers/specs/2026-11-09-curvature-energy-design.md',
        'docs/superpowers/plans/2026-11-09-curvature-energy-plan.md',
        'docs/research/2026-11-10-energy-exploration-amendment.md',
        'docs/collision_curvature_energy_results.md', 'docs/experiment_log.md', 'README.md', 'AGENTS.md')]
    for name in names:
        root = Path('output') / name
        create_json(root / 'completion_validation.json', completion)
        archive = root / 'postprocessing_sources.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as output:
            for p in files:
                output.write(p, str(p))
        create_json(root / 'postprocessing_sources.json', {
            'date': '2026-11-10', 'archive': str(archive), 'archive_sha256': sha(archive),
            'files': {str(p): sha(p) for p in files},
            'launch_command': 'python3 finalize_curvature_energy.py',
            'note': 'Final audit/analysis/report sources. Original launch source.zip unchanged; '
                    'original zero-velocity raw screen preserved and separately interpreted.'})
    print(json.dumps(completion, indent=2))


if __name__ == '__main__':
    finalize()
