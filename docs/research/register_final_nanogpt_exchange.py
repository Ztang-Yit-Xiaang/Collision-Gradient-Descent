"""Seal the final NanoGPT artifacts and append their actual training counts."""

import csv
from pathlib import Path
import zipfile

from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_overlap_transfer import archive_check, check_links


DOCS = Path('docs')
PHASE = 'final_nanogpt_exchange'


def register():
    registry_path = DOCS/'experiment_registry.json'
    inventory_path = DOCS/'experiment_inventory.csv'
    validation_path = DOCS/'recordkeeping_validation.json'
    registry = read(registry_path)
    validation = read(validation_path)
    with inventory_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)

    roots = [Path('output/final_nanogpt_exchange_preflight'), Path('output/final_nanogpt_exchange')]
    previous = {e['folder'] for e in registry['entries']}
    new = set(map(str, roots))
    assert not new & previous and new | previous == {str(p) for p in Path('output').iterdir() if p.is_dir()}
    previous_artifacts = [item for entry in registry['entries'] for item in entry['top_level_artifacts']]
    for item in previous_artifacts:
        assert artifact(Path(item['path']))['sha256'] == item['sha256'], item['path']
    configs = [read(root/'config.json') for root in roots]
    assert configs[0]['source_sha256'] == configs[1]['source_sha256']
    links = check_links()
    for root, config, classification in zip(roots, configs, ('engineering_preflight', 'formal_neural_equal_gradient_comparison')):
        result = read(root/'audit.json')
        assert result['passed']
        paths = [Path(p) for p in ('scripts/audit_final_nanogpt_exchange.py',
                                   'docs/research/register_final_nanogpt_exchange.py',
                                   'docs/research/nanogpt_exchange_proofs.lean',
                                   'docs/final_nanogpt_exchange_results.md',
                                   'docs/final_nanogpt_exchange_table.tex',
                                   'docs/experiment_log.md')]
        archive = root/'postprocessing_sources.zip'
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
            for path in paths:
                stream.write(path, str(path))
        members = {str(p): artifact(p)['sha256'] for p in paths}
        write(root/'postprocessing_sources.json', {'archive': str(archive),
              'archive_sha256': artifact(archive)['sha256'], 'files': members})
        raw = sorted(p for p in root.rglob('*') if p.is_file())
        write(root/'raw_artifact_manifest.json', {'files': [artifact(p) for p in raw]})
        validation['source_archive_checks'].extend(
            [archive_check(root/'source.zip', config['source_sha256']), archive_check(archive, members)])
        tables = {
            'runs': {'path': str(root/'runs.json'), 'records': result['trained_replicas'],
                     'role': 'actual_replica_trainings'},
            'paired_outcomes': {'path': str(root/'paired_outcomes.json'), 'records': result['method_outcomes'],
                               'role': 'one_selected_model_per_method_seed'}
        }
        entry = {'phase_id': PHASE, 'folder': str(root), 'classification': classification,
                 'run_date': '2026-11-16', 'config': config, 'config_path': str(root/'config.json'),
                 'record_tables': tables, 'execution_counts': result['counts'], 'failures': result['failed_runs'],
                 'run_rows_are_derived': False,
                 'counting_note': 'One run row per trained replica. Method outcomes and CSV views summarize the same runs.',
                 'top_level_artifacts': [artifact(path) for path in sorted(root.iterdir()) if path.is_file()],
                 'nested_artifact_counts': {'checkpoints': {'files': result['trained_replicas'],
                     'bytes': sum(p.stat().st_size for p in (root/'checkpoints').iterdir())}}}
        registry['entries'].append(entry)
        row = dict.fromkeys(headers, 0)
        row.update({k: entry[k] for k in ('phase_id', 'folder', 'classification', 'config_path', 'run_rows_are_derived')})
        row.update(runs_records=result['trained_replicas'], full_execution_records=result['trained_replicas'],
                   checkpoint_records=result['checkpoints_verified'], derived_records=result['method_outcomes'])
        inventory.append(row)

    registry['entries'].sort(key=lambda e: e['folder'])
    registry['as_of'] = '2026-11-16'
    validation.update({
        'date': '2026-11-16', 'passed': True, 'local_document_links_checked': True, 'local_document_link_count': links,
        'local_document_link_errors': [], 'retained_output_directories': len(inventory),
        'root_run_rows_retained': sum(int(r['runs_records']) for r in inventory),
        'top_level_artifact_hashes_checked': sum(len(e['top_level_artifacts']) for e in registry['entries']),
        'historical_output_artifact_hashes_unchanged_for_final_nanogpt_exchange': len(previous_artifacts),
        'source_archive_members_checked': sum(a['members_checked'] for a in validation['source_archive_checks']),
        'new_optimization_runs': 99, 'new_model_benchmark_runs': 90,
        'new_optimization_runs_note': 'Final neural comparison. One run per trained replica, including preflight.',
        'new_run_breakdown': {'formal_replica_trainings': 90, 'engineering_replica_trainings': 9,
                              'formal_paired_seed_groups': 10, 'formal_method_outcomes': 30},
        'new_audit_oracle_calls': 0, 'new_execution_segments_with_complete_tables': 99,
        'new_accepted_execution_segments': 90, 'new_formal_mechanism_execution_segments': 0,
        'excluded_engineering_executions': 9, 'latest_full_test_count': 290, 'latest_full_test_seconds': 78.09
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row['folder']))
    print(f'Registered 99 actual replica trainings. Verified {links} links and {len(previous_artifacts)} historical artifacts.')


if __name__ == '__main__':
    register()
