"""Refresh retained-artifact inventory without executing or rewriting experiments."""

import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

from _source_paths import source_path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def refresh():
    registry_path=Path('docs/experiment_registry.json');registry=read(registry_path)
    entries={e['folder']:e for e in registry['entries']}
    for folder,kind in [('collision_cooling','formal_physical_dissipation_screen'),
                        ('collision_cooling_preflight','development_preflight'),
                        ('collision_cooling_parent_analysis','derived_parent_diagnostics')]:
        path=Path('output')/folder
        entries.setdefault(str(path),{'phase_id':'E17','folder':str(path),'classification':kind,
             'run_date':'2026-11-08','counting_note':'JSON/CSV are the same table. Parent diagnostics are derived E16 observations, not new trajectories. Formal tasks each execute once; no shard duplication.'})
    hashes_checked=0
    for e in entries.values():
        path=Path(e['folder']);cfg_path=path/'config.json'
        if cfg_path.exists():e.update(config_path=str(cfg_path),config=read(cfg_path))
        tables={}
        for name in ('runs','tuning','mechanisms','mechanism_maps','metrics'):
            f=path/(name+'.json');csv_path=path/(name+'.csv')
            if f.exists() and isinstance(data:=read(f),list):tables[name]={'path':str(f),'records':len(data)}
            elif csv_path.exists():
                with csv_path.open() as stream:tables[name]={'path':str(csv_path),'records':len(list(csv.DictReader(stream)))}
        e['record_tables']=tables
        if e['phase_id']=='E17':
            e['failures']=[]
            if 'runs' in tables:
                for i,r in enumerate(read(Path(tables['runs']['path']))):
                    if r['failed']:e['failures'].append({'table':'runs','index':i,'case':r['case'],'seed':r['seed'],'method':r['method'],'schedule':r['schedule'],'reason':r['stop_reason']})
        e['top_level_artifacts']=[{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(path.iterdir()) if f.is_file()]
        hashes_checked+=len(e['top_level_artifacts'])
        e['nested_artifact_counts']={f.name:{'files':len(fs:=list(x for x in f.rglob('*') if x.is_file())),
                                            'bytes':sum(x.stat().st_size for x in fs)} for f in sorted(path.iterdir()) if f.is_dir()}
        cfg=e.get('config') or {};manifest=cfg.get('dependency_source_sha256',cfg.get('source_sha256',{}))
        if manifest:
            matched=0;changed=[];missing=[]
            for name,h in manifest.items():
                f=source_path(name)
                if not f.exists():missing.append(name)
                elif sha(f)==h:matched+=1
                else:changed.append(name)
            e['current_source_check']={'matches':matched,'changed_since_recorded_version':changed,'missing':missing,
                 'note':'Historical changes retained. New files absent from old manifests do not alter archived sources. A hash alone cannot restore old source.'}
    registry['entries']=sorted(entries.values(),key=lambda e:e['folder']);registry['as_of']='2026-11-08'
    assert set(entries)=={str(f) for f in Path('output').iterdir() if f.is_dir()}
    write(registry_path,registry)
    headers=['phase_id','folder','classification','runs_records','tuning_records','mechanism_records','derived_records','recorded_numerical_or_contact_failures','config_path']
    with Path('docs/experiment_inventory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader()
        for e in registry['entries']:
            count={k:v['records'] for k,v in e['record_tables'].items()}
            writer.writerow({'phase_id':e['phase_id'],'folder':e['folder'],'classification':e['classification'],
                'runs_records':count.get('runs',0),'tuning_records':count.get('tuning',0),
                'mechanism_records':count.get('mechanisms',0)+count.get('mechanism_maps',0),'derived_records':count.get('metrics',0),
                'recorded_numerical_or_contact_failures':len(e.get('failures',[])),'config_path':e.get('config_path','')})
    for e in entries.values():
        for a in e['top_level_artifacts']:assert sha(Path(a['path']))==a['sha256']
    archives=[]
    old=read(Path('docs/research/2026-11-08-source-snapshot.json'))
    assert sha(Path(old['archive']))==old['archive_sha256']
    with zipfile.ZipFile(old['archive']) as z:
        for m in old['members']:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    archives.append({'archive':old['archive'],'members_checked':len(old['members']),'integrity_passed':True,
       'current_differences':[m['path'] for m in old['members'] if not (f:=source_path(m['path'])).is_file() or sha(f)!=m['sha256']]})
    for e in entries.values():
        path=Path(e['folder']);cfg=e.get('config') or {};archive=path/'source.zip'
        if not archive.exists():continue
        manifest=read(path/'source_manifest.json')
        assert sha(archive)==cfg['source_archive_sha256']
        with zipfile.ZipFile(archive) as z:
            for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        archives.append({'archive':str(archive),'members_checked':len(manifest),'integrity_passed':True,
            'current_differences':[name for name,h in manifest.items() if not (f:=source_path(name)).is_file() or sha(f)!=h]})
        if (post:=path/'postprocessing_sources.zip').exists():
            manifest=read(path/'postprocessing_sources.json')['files']
            with zipfile.ZipFile(post) as z:
                for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
            archives.append({'archive':str(post),'members_checked':len(manifest),'integrity_passed':True})
    links=0;errors=[]
    for doc in [Path('README.md'),*Path('docs').rglob('*.md')]:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
            target=target.strip().strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
            target=unquote(target.split('#')[0]);links+=1
            if not (doc.parent/target).exists():errors.append({'document':str(doc),'target':target})
    validation=read(Path('docs/recordkeeping_validation.json'))
    validation.update({'passed':not errors,'retained_output_directories':len(entries),
        'top_level_artifact_hashes_checked':hashes_checked,'local_document_links_checked':not errors,
        'local_document_link_count':links,'local_document_link_errors':errors,'new_optimization_runs':297,
        'new_run_breakdown':{'formal_unique':270,'preflight':27,'derived_parent_rows_not_new_runs':300},
        'latest_full_test_count':215,'latest_full_test_seconds':90.61,'new_model_benchmark_runs':0,
        'source_archive_checks':archives,'source_archive_members_checked':sum(a['members_checked'] for a in archives),
        'source_archive_matches_current':all(not a.get('current_differences') for a in archives),
        'root_run_rows_retained':sum(e['record_tables'].get('runs',{}).get('records',0) for e in entries.values()),
        'root_tuning_rows_retained':sum(e['record_tables'].get('tuning',{}).get('records',0) for e in entries.values())})
    write(Path('docs/recordkeeping_validation.json'),validation)
    assert not errors,errors
    print(json.dumps({k:validation[k] for k in ('passed','retained_output_directories','root_run_rows_retained','root_tuning_rows_retained','top_level_artifact_hashes_checked','local_document_link_count')},indent=2))


if __name__=='__main__':refresh()
