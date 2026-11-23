"""Register E18 shared-prefix views separately from executed trajectory segments."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path

from refresh_experiment_inventory import read, write, refresh


def register():
    registry_path=Path('docs/experiment_registry.json');registry=read(registry_path)
    entries={e['folder']:e for e in registry['entries']}
    for folder,kind in [('collision_handoff','paired_handoff_views_and_uninterrupted_replication'),
                        ('collision_handoff_preflight','development_short_preflight'),
                        ('collision_handoff_resume_preflight','development_full_budget_resume_preflight'),
                        ('collision_handoff_interrupted','excluded_incomplete_engineering_failure')]:
        path=Path('output')/folder
        entries.setdefault(str(path),{'phase_id':'E18','folder':str(path),'classification':kind,'run_date':'2026-11-08',
            'counting_note':'Root runs rows are shared-prefix full-pipeline comparison views, not separately reexecuted complete trajectories or independent seeds. Prefixes and suffix executions and uninterrupted replicas are separately counted. Interrupted raw-file counts are not complete run counts.'})
    registry['entries']=list(entries.values());write(registry_path,registry)
    with redirect_stdout(io.StringIO()):refresh()
    registry=read(registry_path)
    known_prefixes=known_suffixes=known_replicas=0
    for e in registry['entries']:
        if e['phase_id']!='E18':continue
        path=Path(e['folder']);e['run_rows_are_derived']=True
        for name in ('prefixes','replication_runs'):
            f=path/(name+'.json')
            if f.exists():e['record_tables'][name]={'path':str(f),'records':len(read(f)),
                'role':'executed_prefixes' if name=='prefixes' else 'executed_uninterrupted_trajectories'}
        if 'runs' in e['record_tables']:e['record_tables']['runs']['role']='derived_full_pipeline_views_with_shared_prefix'
        e['failures']=[]
        for name in ('prefixes','runs','replication_runs'):
            if name not in e['record_tables']:continue
            for i,r in enumerate(read(Path(e['record_tables'][name]['path']))):
                if r['failed']:e['failures'].append({'table':name,'index':i,'case':r['case'],'seed':r['seed'],
                    'method':r.get('prefix_method',r.get('method')),'arm':r.get('suffix',r.get('schedule')),'reason':r['stop_reason']})
        if (f:=path/'execution_counts.json').exists():
            e['execution_counts']=read(f)
            known_prefixes+=e['execution_counts']['prefix_executions'];known_suffixes+=e['execution_counts']['suffix_executions'];known_replicas+=e['execution_counts']['uninterrupted_executions']
        if (f:=path/'interruption.json').exists():e['incomplete_work_provenance']=read(f)
        if (f:=path/'task_failure.json').exists():
            task=read(f);e['recorded_task_failure']=task
            e['failures'].append({'table':'task_failure','index':task['task_index'],'reason':task['exception'],
                'scope':'One recorded task error; not a reconstructed count of all unfinished or uncollected work.'})
    write(registry_path,registry)
    headers=['phase_id','folder','classification','runs_records','tuning_records','mechanism_records','derived_records',
             'prefix_execution_records','suffix_execution_records','uninterrupted_records','run_rows_are_derived',
             'recorded_numerical_or_contact_failures','config_path']
    with Path('docs/experiment_inventory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader()
        for e in registry['entries']:
            c={k:v['records'] for k,v in e['record_tables'].items()};counts=e.get('execution_counts',{})
            writer.writerow({'phase_id':e['phase_id'],'folder':e['folder'],'classification':e['classification'],
                'runs_records':c.get('runs',0),'tuning_records':c.get('tuning',0),
                'mechanism_records':c.get('mechanisms',0)+c.get('mechanism_maps',0),'derived_records':c.get('metrics',0),
                'prefix_execution_records':c.get('prefixes',0),'suffix_execution_records':counts.get('suffix_executions',0),
                'uninterrupted_records':c.get('replication_runs',0),'run_rows_are_derived':e.get('run_rows_are_derived',False),
                'recorded_numerical_or_contact_failures':len(e.get('failures',[])),'config_path':e.get('config_path','')})
    validation=read(Path('docs/recordkeeping_validation.json'))
    validation.update({'latest_full_test_count':225,'latest_full_test_seconds':78.75,'new_optimization_runs':known_replicas,
        'new_optimization_runs_note':'Only uninterrupted E18 trajectories; executed phases and derived pipeline views counted separately below.',
        'new_execution_segments_with_complete_tables':known_prefixes+known_suffixes+known_replicas,
        'new_run_breakdown':{'prefix_executions':known_prefixes,'suffix_executions':known_suffixes,'uninterrupted_trajectories':known_replicas,
            'formal_split_pipeline_views_not_independent_executions':540,'formal_uninterrupted_trajectories':180,
            'formal_seed_count':20,'development_preflights':2,'interrupted_work':'Saved files retained, active work/total actual queries unknown; excluded'},
        'table_records_checked':True})
    for e in registry['entries']:
        for t in e['record_tables'].values():
            f=Path(t['path'])
            if f.suffix=='.json':assert len(read(f))==t['records']
            else:
                with f.open() as stream:assert len(list(csv.DictReader(stream)))==t['records']
    write(Path('docs/recordkeeping_validation.json'),validation)
    print(json.dumps({k:validation[k] for k in ('passed','retained_output_directories','root_run_rows_retained','root_tuning_rows_retained',
        'new_execution_segments_with_complete_tables','top_level_artifact_hashes_checked','local_document_link_count')},indent=2))


if __name__=='__main__':register()
