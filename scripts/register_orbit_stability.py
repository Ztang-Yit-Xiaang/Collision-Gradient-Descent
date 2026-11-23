"""E21 preserved execution/view counts on top of E00–E20 registry."""
from contextlib import redirect_stdout
import csv,io,json
from pathlib import Path
from register_events import register as previous


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def register():
    path=Path('docs/experiment_registry.json');registry=read(path);entries={e['folder']:e for e in registry['entries']}
    names={'collision_stability_preflight':'engineering_namespace_collision_preflight_excluded',
           'collision_stability_preflight_corrected':'accepted_engineering_preflight',
           'collision_stability_rosen':'controlled_empirical_stability_mechanism',
           'collision_stability_map':'empirical_bounce_map_and_finite_perturbations'}
    for name,kind in names.items():
        p=Path('output')/name;assert p.exists()
        entries.setdefault(str(p),{'phase_id':'E21','folder':str(p),'classification':kind,'run_date':'2026-11-09',
             'counting_note':'Root runs are Rosenbrock or multi-return trajectories. Each signed one-return probe is an extra execution; matrices/common-time pairings are derived. Spectator seeds do not change the target initial point. First namespace-collision preflight preserved and excluded from accepted mechanism sample.'})
    registry['entries']=list(entries.values());write(path,registry)
    with redirect_stdout(io.StringIO()):previous()
    registry=read(path);totals={k:0 for k in ('rosen_trajectories','return_rollouts','one_return_executions','derived_matrices','executions','solver_queries','setup_value_calls','setup_gradient_calls')}
    excluded=0
    for e in registry['entries']:
        if e['phase_id']!='E21':continue
        p=Path(e['folder']);e['run_rows_are_derived']=False;e['failures']=[]
        for name in ('return_probes','return_matrices'):
            f=p/(name+'.json');rows=read(f)
            e['record_tables'][name]={'path':str(f),'records':len(rows),'role':'executed_single_return_probes' if name=='return_probes' else 'derived_numeric_matrices'}
        for name in ('runs','return_probes'):
            for i,r in enumerate(read(p/(name+'.json'))):
                if r['failed']:e['failures'].append({'table':name,'index':i,'case':r['case'],'reason':r['stop_reason']})
        e['execution_counts']=c=read(p/'execution_counts.json');e['executed_cases']=sorted({r['case'] for t in ('runs','return_probes') for r in read(p/(t+'.json'))})
        e['engineering_events']=[str(f) for f in p.glob('*correction*.json')]
        for key in totals:totals[key]+=c[key]
        if e['classification'].endswith('excluded'):excluded+=c['executions']
    write(path,registry)
    headers=['phase_id','folder','classification','runs_records','run_rows_are_derived','tuning_records','baseline_records','mechanism_records','derived_records',
             'prefix_records','checkpoint_records','uninterrupted_replication_records','full_execution_records','suffix_execution_records',
             'source_execution_records','replay_execution_records','sensitivity_execution_records','contact_context_records',
             'rosen_trajectory_records','return_rollout_records','one_return_execution_records','derived_matrix_records','recorded_failures','config_path']
    with Path('docs/experiment_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader()
        for e in registry['entries']:
            t={k:v['records'] for k,v in e['record_tables'].items()};c=e.get('execution_counts',{})
            row={'phase_id':e['phase_id'],'folder':e['folder'],'classification':e['classification'],'runs_records':t.get('runs',0),
                 'run_rows_are_derived':e.get('run_rows_are_derived',False),'tuning_records':t.get('tuning',0)+(t.get('baselines',0) if e['classification']=='development_selection_and_tuning' else 0),
                 'baseline_records':t.get('baselines',0),'mechanism_records':t.get('mechanisms',0)+t.get('mechanism_maps',0),
                 'derived_records':t.get('metrics',0)+t.get('labels',0),'prefix_records':t.get('prefixes',0),'checkpoint_records':t.get('checkpoints',0),
                 'uninterrupted_replication_records':t.get('replication_runs',0),'full_execution_records':c.get('full_runs',0),
                 'suffix_execution_records':c.get('suffix_runs',c.get('suffix_executions',0)),'source_execution_records':c.get('source_executions',0),
                 'replay_execution_records':c.get('replay_executions',0),'sensitivity_execution_records':c.get('sensitivity_executions',0),
                 'contact_context_records':c.get('contact_contexts',0),'rosen_trajectory_records':c.get('rosen_trajectories',0),
                 'return_rollout_records':c.get('return_rollouts',0),'one_return_execution_records':c.get('one_return_executions',0),
                 'derived_matrix_records':c.get('derived_matrices',0),'recorded_failures':len(e.get('failures',[])),'config_path':e.get('config_path','')}
            w.writerow(row)
    p=Path('docs/recordkeeping_validation.json');v=read(p)
    v.update(date='2026-11-09',latest_full_test_count=253,latest_full_test_seconds=10.76,new_model_benchmark_runs=0,new_optimization_runs=0,
             new_optimization_runs_note='E21 empirical mechanism/finite-time diagnostics, not an optimizer or neural-training ranking.',
             new_execution_segments_with_complete_tables=totals['executions'],new_accepted_execution_segments=totals['executions']-excluded,
             new_formal_mechanism_execution_segments=364,new_run_breakdown=totals,excluded_engineering_executions=excluded)
    write(p,v)
    print(json.dumps({k:v[k] for k in ('passed','retained_output_directories','root_run_rows_retained','root_tuning_rows_retained',
          'new_execution_segments_with_complete_tables','new_accepted_execution_segments','top_level_artifact_hashes_checked','local_document_link_count')},indent=2))


if __name__=='__main__':register()
