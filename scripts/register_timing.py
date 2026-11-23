"""E19 records, retaining E18's separate phase/view accounting."""
from contextlib import redirect_stdout
import csv,io,json
from pathlib import Path
from refresh_experiment_inventory import read,write
from register_handoff import register as register_previous


def register():
    path=Path('docs/experiment_registry.json');registry=read(path);entries={e['folder']:e for e in registry['entries']}
    folders={'collision_timing_preflight':'development_engineering_preflight','collision_timing_safety_preflight':'protection_engineering_preflight',
        'collision_timing_development':'development_selection_and_tuning','collision_timing_confirmation':'heldout_confirmation',
        'collision_timing_safety':'constructed_protection_stress_diagnostic'}
    for folder,kind in folders.items():
        p=Path('output')/folder;entries.setdefault(str(p),{'phase_id':'E19','folder':str(p),'classification':kind,'run_date':'2026-11-09',
            'counting_note':'Root runs are policy views sharing all-on prefixes. Full executions, suffixes, gradient baselines and derived checkpoint/label tables are distinguished; JSON/CSV not added together.'})
    registry['entries']=list(entries.values());write(path,registry)
    with redirect_stdout(io.StringIO()):register_previous()
    registry=read(path);counts={'full':0,'suffix':0,'baselines':0,'policies':0};dev_tuning=0
    for e in registry['entries']:
        if e['phase_id']!='E19':continue
        p=Path(e['folder']);cfg=read(p/'config.json');e['run_rows_are_derived']=True;e['failures']=[]
        for name in ('baselines','checkpoints','collision_value_labels'):
            f=p/(name+'.json')
            if f.exists():e['record_tables'][name]={'path':str(f),'records':len(read(f)),
                'role':'development_tuning' if name=='baselines' and cfg['stage']=='development' else 'executed_baselines' if name=='baselines' else 'derived_observations'}
        for name in ('runs','baselines'):
            for i,r in enumerate(read(p/(name+'.json'))):
                if r['failed']:e['failures'].append({'table':name,'index':i,'case':r['case'],'seed':r['seed'],'method':r.get('rule',r.get('family')),'reason':r['stop_reason']})
        c=read(p/'execution_counts.json');e['execution_counts']=c
        for key,field in [('full','full_runs'),('suffix','suffix_runs'),('baselines','baseline_runs'),('policies','policy_views')]:counts[key]+=c[field]
        if cfg['stage']=='development':dev_tuning+=c['baseline_runs']
    registry['as_of']='2026-11-09';write(path,registry)
    headers=['phase_id','folder','classification','runs_records','run_rows_are_derived','tuning_records','baseline_records','mechanism_records','derived_records',
        'prefix_records','checkpoint_records','uninterrupted_replication_records','full_execution_records','suffix_execution_records','recorded_failures','config_path']
    with Path('docs/experiment_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader()
        for e in registry['entries']:
            t={k:v['records'] for k,v in e['record_tables'].items()};c=e.get('execution_counts',{})
            tuning=t.get('tuning',0)+(t.get('baselines',0) if e['classification']=='development_selection_and_tuning' else 0)
            w.writerow({'phase_id':e['phase_id'],'folder':e['folder'],'classification':e['classification'],'runs_records':t.get('runs',0),
                'run_rows_are_derived':e.get('run_rows_are_derived',False),'tuning_records':tuning,'baseline_records':t.get('baselines',0),
                'mechanism_records':t.get('mechanisms',0)+t.get('mechanism_maps',0),'derived_records':t.get('metrics',0),
                'prefix_records':t.get('prefixes',0),'checkpoint_records':t.get('checkpoints',0),'uninterrupted_replication_records':t.get('replication_runs',0),
                'full_execution_records':c.get('full_runs',0),'suffix_execution_records':c.get('suffix_runs',c.get('suffix_executions',0)),
                'recorded_failures':len(e.get('failures',[])),'config_path':e.get('config_path','')})
    v=read(Path('docs/recordkeeping_validation.json'));v.update({'date':'2026-11-09','latest_full_test_count':234,'latest_full_test_seconds':100.61,
        'new_optimization_runs':counts['full']+counts['baselines'],'new_optimization_runs_note':'E19 full all-on/independent and local baselines only; suffixes and policy views separately listed.',
        'new_execution_segments_with_complete_tables':counts['full']+counts['suffix']+counts['baselines'],
        'new_run_breakdown':counts,'root_tuning_rows_retained':235+dev_tuning,'new_model_benchmark_runs':0})
    write(Path('docs/recordkeeping_validation.json'),v)
    print(json.dumps({k:v[k] for k in ('passed','retained_output_directories','root_run_rows_retained','root_tuning_rows_retained','new_execution_segments_with_complete_tables','top_level_artifact_hashes_checked','local_document_link_count')},indent=2))


if __name__=='__main__':register()
