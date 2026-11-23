"""E24 registry wrapper preserving E00–E23 execution distinctions."""
from contextlib import redirect_stdout
import csv,io,json
from pathlib import Path
from register_energy_generalization import register as previous


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def register():
    path=Path('docs/experiment_registry.json');registry=read(path);entries={e['folder']:e for e in registry['entries']}
    for folder,kind in [('collision_initial_budget_preflight','initial_budget_engineering_preflight'),
                        ('collision_initial_budget','prespecified_matched_initial_E_P_factorial')]:
        p=Path('output')/folder;assert p.exists()
        entries.setdefault(str(p),{'phase_id':'E24','folder':str(p),'classification':kind,'run_date':'2026-11-10',
             'counting_note':'Root rows are actual independent-cloud trajectories. Context probes execute once/shared but each policy pays its required prefix; common-time/aggregate/paired tables are derived, not extra trajectories or independent seeds.'})
    registry['entries']=list(entries.values());write(path,registry)
    with redirect_stdout(io.StringIO()):previous()
    registry=read(path);totals={k:0 for k in ('trajectory_executions','contexts','shared_probe_calls','actual_branch_calls','setup_reference_values','actual_solver_queries','charged_policy_queries')}
    for e in registry['entries']:
        if e['phase_id']!='E24':continue
        p=Path(e['folder']);e['run_rows_are_derived']=False;e['failures']=[]
        for name in ('contexts','common_time','summary','common_time_summary','paired_contrasts','common_time_contrasts','matched_height_initializations'):
            f=p/(name+'.json')
            if f.exists():e['record_tables'][name]={'path':str(f),'records':len(read(f)),'role':'shared_paid_initial_probe_contexts' if name=='contexts' else 'derived_comparisons'}
        for i,r in enumerate(read(p/'runs.json')):
            if r['failed']:e['failures'].append({'table':'runs','index':i,'condition':r['condition'],'stiffness':r['stiffness'],'longitudinal_scale':r['longitudinal_scale'],'incumbent_energy':r['incumbent_energy'],'seed':r['seed'],'policy':r['policy'],'reason':r['stop_reason']})
        e['execution_counts']=c=read(p/'execution_counts.json')
        if (p/'joint_screen_interpretation.json').exists():e['screen_interpretation']=read(p/'joint_screen_interpretation.json')
        if (p/'joint_screen.json').exists():e['momentum_allocation_screen']=read(p/'joint_screen.json')
        for k in totals:totals[k]+=c[k]
        e['engineering_events']=[str(f) for f in p.glob('*failure*.json')]
    registry['as_of']='2026-11-10';write(path,registry)
    headers=['phase_id','folder','classification','runs_records','run_rows_are_derived','tuning_records','baseline_records','mechanism_records','derived_records',
             'prefix_records','checkpoint_records','uninterrupted_replication_records','full_execution_records','suffix_execution_records',
             'source_execution_records','replay_execution_records','sensitivity_execution_records','contact_context_records',
             'rosen_trajectory_records','return_rollout_records','one_return_execution_records','derived_matrix_records','allocation_context_records','recorded_failures','config_path']
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
                 'derived_matrix_records':c.get('derived_matrices',0),'allocation_context_records':c.get('contexts',0) if e['phase_id'] in ('E22','E23','E24') else 0,
                 'recorded_failures':len(e.get('failures',[])),'config_path':e.get('config_path','')}
            w.writerow(row)
    p=Path('docs/recordkeeping_validation.json');v=read(p)
    v.update(date='2026-11-10',latest_full_test_count=280,latest_full_test_seconds=32.30,new_model_benchmark_runs=0,
             new_optimization_runs=totals['trajectory_executions'],new_optimization_runs_note='E24 initialization-only2x4 height/momentum factorial:960 formal trajectories with20 independent seed groups plus48 engineering preflight. All initial E/linear P matched; angular changes recorded, not a dynamic collision or online controller. No tuning/neural training.',
             new_execution_segments_with_complete_tables=totals['trajectory_executions'],new_accepted_execution_segments=totals['trajectory_executions'],
             new_formal_mechanism_execution_segments=960,new_run_breakdown=totals,excluded_engineering_executions=0)
    write(p,v)
    print(json.dumps({k:v[k] for k in ('passed','retained_output_directories','root_run_rows_retained','root_tuning_rows_retained',
          'new_execution_segments_with_complete_tables','top_level_artifact_hashes_checked','local_document_link_count')},indent=2))


if __name__=='__main__':register()
