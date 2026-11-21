"""E20 frozen natural-contact mechanism and numerical-sensitivity batches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile
import numpy as np

from .cooling import CoolingPopulation
from .event_replay import (pack, reconstruct_contact, contact_features, run_replay, outcomes)
from .gravity import GravityConfig
from .population import PopulationConfig, PopulationOracle, run_population
from .population_experiment import make_objective, initial_population
from .strategic_experiment import _json, _csv
from .timing import protected_population

ROOT = Path(__file__).resolve().parents[2]
SPEC = 'docs/superpowers/specs/2026-11-09-collision-event-design.md'
PLAN = 'docs/superpowers/plans/2026-11-09-collision-event-plan.md'
CASES = ('quadratic', 'double_well', 'rosenbrock')


class SetupObjective:
    """Count synthetic initial-condition values outside the solver ledger."""
    def __init__(self, objective):
        self.objective = objective; self.value_calls = 0

    def __getattr__(self, name):
        return getattr(self.objective, name)

    def value(self, x):
        self.value_calls += 1
        return self.objective.value(x)


def snapshot(output):
    paths = set(ROOT.glob('*.py'))
    for name in ('src', 'tests'):
        paths.update(p for p in (ROOT/name).rglob('*.py') if '__pycache__' not in p.parts)
    paths.update(ROOT/p for p in ('AGENTS.md', 'pyproject.toml', SPEC, PLAN))
    extension = ROOT/'docs/research/2026-11-09-contact-sensitivity-extension.md'
    if extension.exists():
        paths.add(extension)
    manifest = {}
    with zipfile.ZipFile(output/'source.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            data = path.read_bytes(); name = str(path.relative_to(ROOT))
            z.writestr(name, data); manifest[name] = hashlib.sha256(data).hexdigest()
    _json(output/'source_manifest.json', manifest)
    return manifest


def save_trace(output, name, rows, events, maps=None, uncommitted=None):
    for r in rows:
        r.setdefault('total_oracle_calls', sum(r[k] for k in ('value_calls', 'gradient_calls', 'path_calls')))
    _csv(output/'traces'/(name+'.csv.gz'), rows, True)
    _json(output/'events'/(name+'.json'), events)
    if maps is not None:
        _json(output/'maps'/(name+'.json'), maps)
    if uncommitted is not None:
        _json(output/'uncommitted'/(name+'.json'), uncommitted)


def source_task(output, case, seed, source_budget, source_steps, replay_steps):
    output = Path(output); objective, minimum, boundary = make_objective(case)
    setup = SetupObjective(objective); initial = initial_population(case, setup, seed, rho_max=.24)
    initial_values = [setup.value(p.x) for p in initial.particles]
    first = min(initial_values); target = minimum+.1*(first-minimum)
    config = PopulationConfig(physics=GravityConfig(), rho=.24, max_pair_tests=100000)
    method = CoolingPopulation(config, schedule='ramp', rule='elastic', seed=seed)
    oracle = PopulationOracle(objective, total_budget=source_budget)
    full = run_population(method, oracle, initial, source_steps, basin_boundary=boundary, target_value=target)
    name = f'{case}__s{seed}__source'
    save_trace(output, name, full.rows, full.events, method.map_attempts,
               {'stop_reason': full.stop_reason, 'final_counts': full.counts,
                'last_committed_state': pack(full.final_state),
                'partial_intermediate_states': 'Not returned by frozen solver'})
    _json(output/'initial_states'/(name+'.json'), {**pack(initial), 'values': initial_values,
                                                 'minimum': minimum, 'boundary': boundary, 'target': target})
    pair_indices = [i for i, e in enumerate(full.events) if e['kind'] == 'pair'][:3]
    source = {'run_id': name, 'case': case, 'seed': seed, 'failed': full.stop_reason.startswith(('contact_error:', 'numerical_error:')),
              'stop_reason': full.stop_reason, 'committed_pair_events': sum(e['kind']=='pair' for e in full.events),
              'selected_contacts': len(pair_indices), 'total_oracle_calls': full.counts['total_oracle_calls'],
              'setup_value_calls': setup.value_calls, 'known_reference_value_calls': int(case=='double_well'),
              'physical_time': full.rows[-1]['time'], 'wall_seconds': full.wall_seconds}
    contexts = []; runs = []; unavailable = []; branch_calls = probes = 0
    for ordinal, event_index in enumerate(pair_indices):
        context_id = f'{case}__s{seed}__contact{ordinal}'
        try:
            incoming, context = reconstruct_contact(full.rows, full.events, event_index)
        except ValueError as exc:
            unavailable.append({'context_id': context_id, 'source_run': name, 'reason': str(exc)})
            continue
        probe = PopulationOracle(objective)
        values = [probe.value(p.x) for p in incoming.particles]
        gradients = [probe.gradient(incoming.particles[k].x) for k in (context['i'], context['j'])]
        features = contact_features(incoming, context, values, gradients, first)
        context.update(context_id=context_id, case=case, seed=seed, ordinal=ordinal,
                       incoming=pack(incoming), source_run=name, features=features,
                       values=values, gradients=[g.tolist() for g in gradients],
                       initial_best=first, minimum=minimum, boundary=boundary, target=target,
                       feature_probe_counts=probe.counts(), feature_probe_calls=sum(probe.counts().values()),
                       source_prefix_query_count='Not recorded at substep; full source execution cost retained')
        probes += context['feature_probe_calls']; contexts.append(context)
        _json(output/'contexts'/(context_id+'.json'), context)
        for rule, replica in [('none', 0), ('elastic', 0)]+[('random', k) for k in range(4)]:
            entropy = [20261109, seed, CASES.index(case), ordinal, replica]
            result = run_replay(objective, incoming, context, rule,
                                seed=np.random.SeedSequence(entropy), steps=replay_steps, boundary=boundary)
            key = context_id+f'__{rule}{replica}'
            result['map']['rng_entropy'] = entropy
            save_trace(output, key, result['rows'], result['events'], result['map'], result['uncommitted'])
            views = outcomes(result['rows'], context, first, minimum, target, boundary)
            runs.append({'run_id': key, 'context_id': context_id, 'case': case, 'seed': seed,
                         'rule': rule, 'replica': replica, 'failed': result['failed'],
                         'stop_reason': result['stop_reason'], **result['counts'],
                         'shared_feature_charge': 6, 'views': views,
                         'collision_energy_error': result['map']['energy_error'],
                         'collision_momentum_error': result['map']['momentum_error']})
            branch_calls += result['counts']['actual_branch_calls']
    return {'sources': [source], 'contexts': contexts, 'runs': runs, 'unavailable': unavailable,
            'actual_source_calls': source['total_oracle_calls'], 'actual_probe_calls': probes,
            'actual_replay_calls': branch_calls, 'setup_value_calls': setup.value_calls+int(case=='double_well')}


def sensitivity_task(output, seed, h, tight, horizon=10.):
    output = Path(output); objective, minimum, boundary = make_objective('rosenbrock')
    setup = SetupObjective(objective); initial = protected_population('rosenbrock', setup, minimum, seed)
    values = [setup.value(p.x) for p in initial.particles]
    physics = GravityConfig(h=h, gap_tolerance=1e-11 if tight else 1e-9,
                            time_tolerance=1e-13 if tight else 1e-11,
                            normal_tolerance=1e-13 if tight else 1e-11)
    config = PopulationConfig(physics=physics, rho=.24, collisions=False)
    method = CoolingPopulation(config, schedule='ramp', rule='elastic')
    run = run_population(method, PopulationOracle(objective, total_budget=100000), initial, round(horizon/h))
    name = f'rosenbrock__s{seed}__h{h}__tight{int(tight)}'
    save_trace(output, name, run.rows, run.events, method.map_attempts)
    _json(output/'initial_states'/(name+'.json'), {**pack(initial), 'values': values, 'physics': asdict(physics)})
    rise = [r['values'][0]-values[0] for r in run.rows]
    record = {'run_id': name, 'case': 'rosenbrock', 'seed': seed, 'h': h, 'tight': tight,
              'failed': run.stop_reason.startswith(('contact_error:', 'numerical_error:')),
              'stop_reason': run.stop_reason, 'physical_time': run.rows[-1]['time'],
              'protected_initial_loss': values[0], 'protected_max_rise': max(0., max(rise)),
              'protected_final_rise': rise[-1], 'initial_clearance': initial.particles[0].z-values[0],
              'initial_energy': run.rows[0]['energy'], 'final_energy': run.rows[-1]['energy'],
              'cumulative_damping': run.rows[-1]['cumulative_damping'],
              'wall_contacts': len(run.events), 'total_oracle_calls': run.counts['total_oracle_calls'],
              'setup_value_calls': setup.value_calls, 'wall_seconds': run.wall_seconds}
    return {'sensitivity_runs': [record], 'actual_sensitivity_calls': record['total_oracle_calls'],
            'setup_value_calls': setup.value_calls}


def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    for name in ('traces', 'events', 'maps', 'contexts', 'initial_states', 'uncommitted'):
        (output/name).mkdir()
    seeds = list(range(args.seed_start, args.seed_start+args.seeds))
    manifest = snapshot(output); frozen = None
    if args.stage == 'confirmation':
        if not args.frozen:
            raise ValueError('Confirmation requires a frozen development manifest.')
        path = Path(args.frozen); frozen = json.loads(path.read_text())
        for name, digest in frozen['files'].items():
            if hashlib.sha256((path.parent/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen development inputs changed: '+name)
        _json(output/'frozen_inputs.json', frozen)
    config = {'phase': 'E20', 'stage': args.stage, 'seeds': seeds, 'cases': list(CASES),
              'source_budget': 1800 if args.stage=='preflight' else 6000,
              'source_steps': 120 if args.stage=='preflight' else 500,
              'replay_steps': 20 if args.stage=='preflight' else 100,
              'replay_budget': 2000, 'primary_label_cap': 1000, 'random_replicas': 4,
              'workers': args.workers, 'sensitivity_time': args.sensitivity_time,
              'launch_argv': sys.argv, 'cwd': str(Path.cwd()),
              'launch_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'platform': platform.platform(), 'executable': sys.executable,
              'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
              'source_sha256': manifest, 'source_archive_sha256': hashlib.sha256((output/'source.zip').read_bytes()).hexdigest(),
              'development_frozen_manifest': args.frozen,
              'development_frozen_sha256': hashlib.sha256(Path(args.frozen).read_bytes()).hexdigest() if args.frozen else None,
              'counting_note': 'Sources, six executed replays per contact and derived labels differ. Six feature probes executed once/context; charged to each branch. Setup objective values separately counted.'}
    _json(output/'config.json', config)
    tables = {k: [] for k in ('sources', 'contexts', 'runs', 'unavailable', 'sensitivity_runs')}
    costs = {k: 0 for k in ('actual_source_calls','actual_probe_calls','actual_replay_calls','actual_sensitivity_calls','setup_value_calls')}
    def save():
        for name, rows in tables.items():
            _json(output/(name+'.json'), rows)
            if rows:
                _csv(output/(name+'.csv'), rows)
        _json(output/'execution_counts.json', {**costs,
            'actual_solver_queries': sum(v for k,v in costs.items() if k.startswith('actual_')),
            'actual_queries_including_setup': sum(costs.values()),
            'source_executions': len(tables['sources']), 'replay_executions': len(tables['runs']),
            'sensitivity_executions': len(tables['sensitivity_runs']), 'contact_contexts': len(tables['contexts']),
            'unavailable_contexts': len(tables['unavailable']), 'independent_seed_count': len(seeds)})
    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == 'sensitivity':
            futures = [pool.submit(sensitivity_task, str(output), seed, h, tight, args.sensitivity_time)
                       for seed in seeds for h in (.05, .025, .0125) for tight in (False, True)]
        else:
            futures = [pool.submit(source_task, str(output), case, seed, config['source_budget'], config['source_steps'], config['replay_steps'])
                       for seed in seeds for case in CASES]
        try:
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                for name in tables:
                    tables[name].extend(result.get(name, []))
                for name in costs:
                    costs[name] += result.get(name, 0)
                save()
                if index % 5 == 0 or index == len(futures):
                    print(f'{args.stage}: {index}/{len(futures)} source tasks; {len(tables["runs"])} replays', flush=True)
        except BaseException:
            _json(output/'task_failure.json', {'traceback': traceback.format_exc(),
                   'note': 'Incremental completed tables retained; active task cost may be unavailable.'})
            for future in futures:
                future.cancel()
            raise
    print(json.dumps(json.loads((output/'execution_counts.json').read_text())), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=('preflight','development','confirmation','sensitivity'), required=True)
    p.add_argument('--seed-start', type=int, required=True); p.add_argument('--seeds', type=int, required=True)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--frozen')
    p.add_argument('--sensitivity-time', type=float, default=10.)
    args = p.parse_args()
    if args.seeds < 1 or args.workers < 1:
        p.error('Positive seeds/workers required.')
    if not np.isfinite(args.sensitivity_time) or not 0 < args.sensitivity_time <= 25:
        p.error('Sensitivity time must be in (0,25].')
    run(args)


if __name__ == '__main__':
    main()
