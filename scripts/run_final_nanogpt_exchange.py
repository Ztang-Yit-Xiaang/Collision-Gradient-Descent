"""Run and audit the final, equal-gradient NanoGPT velocity-exchange comparison."""

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import random
import sys
from time import perf_counter
import zipfile

import numpy as np
import torch

from collision_optim.nanogpt_benchmark import (
    LMConfig, TextData, TorchControl, TrainingCost, make_model, make_optimizer,
    model_hash, probe_loss, train_step,
)
from collision_optim.strategic_experiment import _csv, _json


SPEC = Path('docs/superpowers/specs/2026-11-16-final-nanogpt-exchange-design.md')
DATA = Path('data/tinyshakespeare.txt')
METHODS = ('exchange', 'independent', 'adamw')
SEEDS = {'preflight': [5100], 'formal': list(range(5110, 5120))}
SCHEDULE = (20, 40, 60, 80)
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
GRADIENTS = 400
EVAL_BATCHES = 32
MOMENTUM_LR = 0.3
TOL = 1e-12


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def save_source(root):
    files = [Path(__file__), SPEC, Path('AGENTS.md'), Path('pyproject.toml'),
             *Path('src').rglob('*.py'),
             *(p for p in Path('third_party/nanogpt').iterdir() if p.is_file())]
    manifest = {}
    with zipfile.ZipFile(root/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            payload = path.read_bytes()
            archive.writestr(str(path), payload)
            manifest[str(path)] = hashlib.sha256(payload).hexdigest()
    _json(root/'source_manifest.json', manifest)
    return manifest


def exchange_pair(models, optimizers, i, j):
    left = [optimizers[i].state[p] for p in models[i].parameters()]
    right = [optimizers[j].state[p] for p in models[j].parameters()]

    def buffer(states):
        return torch.cat([state['momentum_buffer'].flatten() for state in states])

    hashes_before = [model_hash(models[k]) for k in (i, j)]
    a, b = buffer(left), buffer(right)
    for state_i, state_j in zip(left, right, strict=True):
        buf_i, buf_j = state_i['momentum_buffer'], state_j['momentum_buffer']
        assert buf_i.shape == buf_j.shape and buf_i.data_ptr() != buf_j.data_ptr()
        state_i['momentum_buffer'], state_j['momentum_buffer'] = buf_j, buf_i
    c, d = buffer(left), buffer(right)
    before, after = [tensor_hash(a), tensor_hash(b)], [tensor_hash(c), tensor_hash(d)]
    hashes_after = [model_hash(models[k]) for k in (i, j)]
    vi, vj, wi, wj = [-MOMENTUM_LR*tensor.double() for tensor in (a, b, c, d)]
    energies = [float(0.5*torch.dot(v, v)) for v in (vi, vj, wi, wj)]
    momentum_residual = float(((wi + wj) - (vi + vj)).abs().max())
    kinetic_residual = abs(sum(energies[2:]) - sum(energies[:2]))
    assert after == before[::-1] and hashes_before == hashes_after
    assert before[0] != before[1] and momentum_residual <= TOL and kinetic_residual <= TOL
    return {'pair': [i, j], 'buffer_before': before, 'buffer_after': after,
            'parameter_before': hashes_before, 'parameter_after': hashes_after,
            'nontrivial': before[0] != before[1], 'velocity_difference_norm': float(torch.linalg.vector_norm(vi-vj)),
            'kinetic_before': energies[:2], 'kinetic_after': energies[2:],
            'kinetic_transfer': [energies[2]-energies[0], energies[3]-energies[1]],
            'momentum_inf_residual': momentum_residual, 'kinetic_residual': kinetic_residual,
            'external_impulse': 0., 'external_work': 0., 'dissipated_energy': 0.}


def train_arm(root, data, cfg, seed, method, eval_batches):
    begin = perf_counter()
    count = 1 if method == 'adamw' else 4
    updates = GRADIENTS // count
    stream_seeds = [4*seed+i for i in range(count)]
    models = [make_model(data, cfg, start) for start in stream_seeds]
    kind, lr = ('adamw', 0.001) if method == 'adamw' else ('momentum', MOMENTUM_LR)
    optimizers = [make_optimizer(model, kind, lr, 0., cfg) for model in models]
    controls = [TorchControl(model) for model in models]
    cost = TrainingCost(3*GRADIENTS + EVAL_BATCHES*(count+1))
    initials = [model_hash(model) for model in models]
    history = [[] for _ in models]
    events = []
    training_begin = perf_counter()
    for step in range(1, updates+1):
        for i, (model, optimizer, control) in enumerate(zip(models, optimizers, controls)):
            loss, _, norm = train_step(model, optimizer, control,
                                       data.batch('train', stream_seeds[i], step-1), cfg, cost)
            history[i].append([loss, norm])
        if step == SCHEDULE[0]:
            first_boundary = [model_hash(model) for model in models]
        if method == 'exchange' and step in SCHEDULE:
            for i, j in PAIRINGS[SCHEDULE.index(step) % len(PAIRINGS)]:
                events.append({'step': step, **exchange_pair(models, optimizers, i, j)})
    seconds = perf_counter()-training_begin
    validation = [probe_loss(model, eval_batches['tuning'], cost) for model in models]
    selected = min(range(count), key=validation.__getitem__)
    test_loss = probe_loss(models[selected], eval_batches['test'], cost)
    assert cost.used == cost.limit
    assert cost.counts['real_train_units'] == 3*GRADIENTS
    rows = []
    for i, model in enumerate(models):
        checkpoint = f'checkpoints/{seed}_{method}_{i}.pt'
        torch.save(model.state_dict(), root/checkpoint)
        rows.append({'seed': seed, 'method': method, 'replica': i, 'stream_seed': stream_seeds[i],
                     'initial_sha256': initials[i], 'step20_sha256': first_boundary[i],
                     'final_sha256': model_hash(model), 'checkpoint': checkpoint,
                     'checkpoint_sha256': file_hash(root/checkpoint), 'gradient_evaluations': updates,
                     'training_history': history[i], 'final_training_minibatch_loss': history[i][-1][0],
                     'historical_best_training_minibatch_loss': min(row[0] for row in history[i]),
                     'validation_loss': validation[i], 'test_loss': test_loss if i == selected else None,
                     'selected': i == selected, 'failed': False, 'stop_reason': 'budget_reached'})
    outcome = {'seed': seed, 'method': method, 'parameter_count': sum(p.numel() for p in models[0].parameters()),
               'selected_replica': selected, 'final_test_loss': test_loss, 'selected_validation_loss': validation[selected],
               'gradient_evaluations': GRADIENTS, 'validation_forwards': EVAL_BATCHES*count,
               'test_forwards': EVAL_BATCHES, 'cost': cost.counts, 'events': events,
               'training_seconds': seconds, 'elapsed_seconds': perf_counter()-begin}
    return outcome, rows


def summarize(outcomes):
    losses = {m: np.array([r['final_test_loss'] for r in outcomes if r['method'] == m]) for m in METHODS}
    stats = {m: {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if v.size > 1 else None}
             for m, v in losses.items()}
    diffs = {}
    for reference in METHODS[1:]:
        diff = losses['exchange'] - losses[reference]
        ids = np.random.default_rng(5239).integers(diff.size, size=(20_000, diff.size))
        interval = np.quantile(diff[ids].mean(axis=1), [0.025, 0.975]).tolist() if diff.size > 1 else None
        diffs[reference] = {'mean': float(diff.mean()), 'interval95': interval,
                            'improved': int((diff < 0).sum()), 'worsened': int((diff > 0).sum()),
                            'tied': int((diff == 0).sum())}
    events = [event for row in outcomes for event in row['events']]
    return {'seed_groups': len(losses['exchange']), 'test_loss': stats, 'paired_differences': diffs,
            'total_gradients': sum(row['gradient_evaluations'] for row in outcomes),
            'nontrivial_exchanges': sum(e['nontrivial'] for e in events),
            'max_momentum_inf_residual': max(e['momentum_inf_residual'] for e in events),
            'max_kinetic_residual': max(e['kinetic_residual'] for e in events),
            'median_absolute_pair_energy_transfer': float(np.median([abs(e['kinetic_transfer'][0]) for e in events]))}


def run(root, stage, lock):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = LMConfig()
    data = TextData(DATA, cfg)
    seeds = SEEDS[stage]
    root.mkdir(parents=True, exist_ok=False)
    (root/'checkpoints').mkdir()
    manifest = save_source(root)
    config = {'stage': stage, 'seeds': seeds, 'methods': METHODS, 'schedule': SCHEDULE, 'pairs': PAIRINGS,
              'gradient_budget_per_method': GRADIENTS, 'eval_batches_per_candidate': EVAL_BATCHES,
              'momentum_lr': MOMENTUM_LR, 'adamw_lr': 0.001, 'weight_decay': 0., 'evaluation_seed': 71001,
              'lm': asdict(cfg), 'data': str(DATA), 'data_sha256': data.sha256,
              'split_lengths': {name: len(tokens) for name, tokens in data.splits.items()},
              'source_archive_sha256': file_hash(root/'source.zip'), 'source_sha256': manifest,
              'engineering_lock': str(lock) if lock else None, 'command': [sys.executable, *sys.argv],
              'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'device': 'cpu', 'dtype': str(torch.get_default_dtype()),
                              'torch_threads': torch.get_num_threads(), 'deterministic_algorithms': True}}
    _json(root/'config.json', config)
    if lock:
        previous = json.loads((lock/'config.json').read_text())
        assert (lock/'summary.json').is_file() and previous['stage'] == 'preflight'
        assert manifest == previous['source_sha256'] and data.sha256 == previous['data_sha256']
    batches = {split: [data.batch(split, 71001, i) for i in range(EVAL_BATCHES)] for split in ('tuning', 'test')}
    outcomes, runs = [], []
    for seed in seeds:
        order = list(METHODS)
        random.Random(seed).shuffle(order)
        paired = {}
        for method in order:
            try:
                outcome, rows = train_arm(root, data, cfg, seed, method, batches)
            except Exception as exc:
                _json(root/'failure.json', {'seed': seed, 'method': method, 'reason': repr(exc)})
                raise
            paired[method] = (outcome, rows)
            _json(root/f'{seed}_{method}.json', {'outcome': outcome, 'runs': rows})
        swap, independent = (paired[method][1] for method in METHODS[:2])
        assert paired['adamw'][1][0]['initial_sha256'] == swap[0]['initial_sha256']
        for a, b in zip(swap, independent, strict=True):
            assert a['initial_sha256'] == b['initial_sha256'] and a['step20_sha256'] == b['step20_sha256']
            assert a['training_history'][:20] == b['training_history'][:20]
        assert len(paired['exchange'][0]['events']) == 8
        for method in METHODS:
            outcome, rows = paired[method]
            outcomes.append(outcome)
            runs.extend(rows)
        print(seed, {method: paired[method][0]['final_test_loss'] for method in METHODS}, flush=True)
    _json(root/'runs.json', runs)
    _json(root/'paired_outcomes.json', outcomes)
    _csv(root/'paired_outcomes.csv', [{k: v for k, v in row.items() if k != 'events'} for row in outcomes])
    _json(root/'summary.json', summarize(outcomes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', required=True, choices=SEEDS)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    if (args.stage == 'formal') != (args.lock is not None):
        parser.error('The formal comparison requires a completed preflight source lock.')
    run(args.output, args.stage, args.lock)
