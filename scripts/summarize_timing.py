"""E19 confirmation contrasts and complete curves, without reselection."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np


def summarize(output):
    output=Path(output);runs=json.loads((output/'runs.json').read_text());selection=json.loads((output/'selection_input.json').read_text())
    selected=selection['fraction'];lookup={(r['case'],r['seed'],r['rule'],r['fraction']):r for r in runs}
    baselines=json.loads((output/'baselines.json').read_text());bl={(r['case'],r['seed'],r['family']):r for r in baselines};pairs=[]
    for r in runs:
        if r['rule']!=('independent' if selected==0 else 'elastic') or r['fraction']!=selected:continue
        refs=[(f'fraction_{f}',lookup[r['case'],r['seed'],'independent' if f==0 else 'elastic',f]) for f in (0.,.5,1.)]
        refs.append(('matched_random',lookup[r['case'],r['seed'],'independent' if selected==0 else 'random',selected]))
        refs.extend((family,bl[r['case'],r['seed'],family]) for family in ('gd','momentum'))
        for name,ref in refs:
            valid=not(r['failed'] or ref['failed']);diff=r['relative_final_gap']-ref['relative_final_gap'] if valid else None
            pairs.append({'case':r['case'],'seed':r['seed'],'selected_fraction':selected,'reference':name,'valid':valid,
                'difference':diff,'outcome':'unavailable' if not valid else 'better' if diff< -1e-6 else 'worse' if diff>1e-6 else 'tie'})
    groups=defaultdict(list)
    for r in pairs:groups[r['case'],r['reference']].append(r)
    contrasts=[]
    for (case,ref),rows in sorted(groups.items()):
        counts=Counter(r['outcome'] for r in rows);values=[r['difference'] for r in rows if r['valid']]
        contrasts.append({'case':case,'reference':ref,**{k:counts[k] for k in ('better','worse','tie','unavailable')},
            'median_paired_difference':float(np.median(values)) if values else None})
    baseline_summary=[]
    for case in sorted({r['case'] for r in baselines}):
        for family in ('gd','momentum'):
            g=[r for r in baselines if r['case']==case and r['family']==family];v=[r for r in g if not r['failed']]
            baseline_summary.append({'case':case,'family':family,'eta':selection['baseline_rates'][case][family],
                'runs':len(g),'valid':len(v),'final_hits':sum(r['final_target_hit'] for r in v),
                'median_relative_final_gap':float(np.median([r['relative_final_gap'] for r in v])) if v else None})
    result={'selected_fraction':selected,'selection_not_recomputed_on_confirmation':True,'contrasts':contrasts,'baselines':baseline_summary}
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');(output/'paired.json').write_text(json.dumps(pairs,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');summarize(p.parse_args().output)
