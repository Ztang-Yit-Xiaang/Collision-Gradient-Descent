"""Preserve/recheck raw long-window default/tight identity and crossing times."""
import argparse,csv,gzip,json
from pathlib import Path


def check(output):
    p=Path(output);records=json.loads((p/'sensitivity_runs.json').read_text());checks=[]
    for seed in (880,881,882):
        for h in (.05,.025,.0125):
            pair=[r for r in records if r['seed']==seed and r['h']==h];traces=[]
            for r in sorted(pair,key=lambda r:r['tight']):
                with gzip.open(p/'traces'/(r['run_id']+'.csv.gz'),'rt') as f:traces.append(list(csv.DictReader(f)))
            same=len(traces[0])==len(traces[1]) and all(a['positions']==b['positions'] and a['velocities']==b['velocities'] and a['time']==b['time'] for a,b in zip(*traces))
            checks.append({'seed':seed,'h':h,'default_tight_states_identical':same,
                           'first_time_above_1e-4':next((float(r['time']) for r in traces[0] if json.loads(r['values'])[0]>1e-4),None)})
    initials=[json.loads((p/'initial_states'/(r['run_id']+'.json')).read_text()) for r in records]
    same_initial=all(r['positions'][0]==initials[0]['positions'][0] and r['velocities'][0]==initials[0]['velocities'][0] for r in initials)
    result={'checks':checks,'protected_initial_state_identical_across_all_seeds_and_settings':same_initial,
            'interpretation':'Seeds change other independent particles. Protected initial state is shared; scheduler substep partition/roundoff can differ. Not three independently sampled good-particle starts.'}
    path=p/'numerical_pair_checks.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:
        with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'pairs_checked':len(checks),'all_default_tight_identical':all(c['default_tight_states_identical'] for c in checks),
                      'protected_initial_state_identical':same_initial}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');check(p.parse_args().output)
