"""E20 bookkeeping layered over preserved E18/E19 counting conventions."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
from register_timing import register as previous


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def register():
    path=Path('docs/experiment_registry.json');registry=read(path)
    entries={e['folder']:e for e in registry['entries']}
    folders={'collision_event_preflight':'engineering_mechanism_preflight',
             'collision_event_sensitivity':'prespecified_numerical_diagnostic',
             'collision_event_sensitivity_long':'outcome_motivated_long_window_diagnostic',
             'collision_event_development':'mechanism_development_and_offline_fit',
             'collision_event_confirmation':'heldout_conditional_contact_mechanism'}
    for folder,kind in folders.items():
        p=Path('output')/folder;assert p.exists()
        entries.setdefault(str(p),{'phase_id':'E20','folder':str(p),'classification':kind,
             'run_date':'2026-11-09','counting_note':'Root runs are actually executed isolated replays. Sources and sensitivity trajectories are additional executed tables; contexts, labels and windows are derived, not independent runs. Four random replicas share a contact. Long sensitivity repeats the same three seeds.'})
    registry['entries']=list(entries.values());write(path,registry)
    with redirect_stdout(io.StringIO()):previous()
    registry=read(path);counts={'sources':0,'replays':0,'sensitivity':0,'contexts':0,'solver_queries':0,'setup_values':0}
    for e in registry['entries']:
        if e['phase_id']!='E20':continue
        p=Path(e['folder']);e['run_rows_are_derived']=False;e['failures']=[]
        for name in ('sources','contexts','sensitivity_runs','labels','unavailable','unavailable_labels'):
            f=p/(name+'.json')
            if f.exists():
                e['record_tables'][name]={'path':str(f),'records':len(read(f)),
                     'role':'executed_source_trajectories' if name=='sources' else 'executed_numerical_diagnostics' if name=='sensitivity_runs' else 'derived_contexts_labels_or_unavailable'}
        for name in ('sources','runs','sensitivity_runs'):
            for i,r in enumerate(read(p/(name+'.json'))):
                if r['failed']:e['failures'].append({'table':name,'index':i,'case':r['case'],'seed':r['seed'],'reason':r['stop_reason']})
        e['execution_counts']=c=read(p/'execution_counts.json')
        e['executed_cases']=sorted({r['case'] for table in ('sources','runs','sensitivity_runs') for r in read(p/(table+'.json'))})
        e['configuration_note']='Sensitivity config cases is the shared terrain catalogue; executed_cases and sensitivity_runs specify Rosenbrock only. Source/replay limits do not apply to sensitivity, whose cap100000 and horizon are in the frozen spec/code.'
        e['engineering_events']=[str(f) for f in p.glob('*failure*.json')]
        for key,field in [('sources','source_executions'),('replays','replay_executions'),('sensitivity','sensitivity_executions'),
                          ('contexts','contact_contexts'),('solver_queries','actual_solver_queries'),('setup_values','setup_value_calls')]:counts[key]+=c[field]
    write(path,registry)
    headers=['phase_id','folder','classification','runs_records','run_rows_are_derived','tuning_records','baseline_records',
             'mechanism_records','derived_records','prefix_records','checkpoint_records','uninterrupted_replication_records',
             'full_execution_records','suffix_execution_records','source_execution_records','replay_execution_records',
             'sensitivity_execution_records','contact_context_records','recorded_failures','config_path']
    with Path('docs/experiment_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader()
        for e in registry['entries']:
            t={k:v['records'] for k,v in e['record_tables'].items()};c=e.get('execution_counts',{})
            tuning=t.get('tuning',0)+(t.get('baselines',0) if e['classification']=='development_selection_and_tuning' else 0)
            w.writerow({'phase_id':e['phase_id'],'folder':e['folder'],'classification':e['classification'],
                'runs_records':t.get('runs',0),'run_rows_are_derived':e.get('run_rows_are_derived',False),
                'tuning_records':tuning,'baseline_records':t.get('baselines',0),
                'mechanism_records':t.get('mechanisms',0)+t.get('mechanism_maps',0),
                'derived_records':t.get('metrics',0)+t.get('labels',0),'prefix_records':t.get('prefixes',0),
                'checkpoint_records':t.get('checkpoints',0),'uninterrupted_replication_records':t.get('replication_runs',0),
                'full_execution_records':c.get('full_runs',0),'suffix_execution_records':c.get('suffix_runs',c.get('suffix_executions',0)),
                'source_execution_records':c.get('source_executions',0),'replay_execution_records':c.get('replay_executions',0),
                'sensitivity_execution_records':c.get('sensitivity_executions',0),'contact_context_records':c.get('contact_contexts',0),
                'recorded_failures':len(e.get('failures',[])),'config_path':e.get('config_path','')})
    validation_path=Path('docs/recordkeeping_validation.json');v=read(validation_path)
    v.update(date='2026-11-09',latest_full_test_count=245,latest_full_test_seconds=35.24,
             new_optimization_runs=0,new_optimization_runs_note='E20 is a conditional-contact mechanism study and numerical diagnostic; no full-algorithm ranking or real training.',
             new_execution_segments_with_complete_tables=counts['sources']+counts['replays']+counts['sensitivity'],
             new_run_breakdown=counts,new_model_benchmark_runs=0)
    write(validation_path,v)
    print(json.dumps({k:v[k] for k in ('passed','retained_output_directories','root_run_rows_retained','root_tuning_rows_retained',
          'new_execution_segments_with_complete_tables','top_level_artifact_hashes_checked','local_document_link_count')},indent=2))


if __name__=='__main__':register()
