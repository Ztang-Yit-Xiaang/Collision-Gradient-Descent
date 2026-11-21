"""Audit final comparison artifacts without running additional training or evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import torch

from collision_optim.nanogpt_benchmark import LMConfig, TextData, make_model, model_hash
from collision_optim.strategic_experiment import _json
from run_final_nanogpt_exchange import METHODS, PAIRINGS, SCHEDULE, SEEDS, TOL, file_hash, summarize


def read(path):
    return json.loads(path.read_text())


def audit(root):
    config = read(root/'config.json')
    runs = read(root/'runs.json')
    outcomes = read(root/'paired_outcomes.json')
    seeds = config['seeds']
    assert seeds == SEEDS[config['stage']]
    assert [(r['seed'], r['method']) for r in outcomes] == [(s, m) for s in seeds for m in METHODS]
    assert len(runs) == 9*len(seeds)
    assert config['source_archive_sha256'] == file_hash(root/'source.zip')
    with zipfile.ZipFile(root/'source.zip') as archive:
        assert set(archive.namelist()) == set(config['source_sha256'])
        for path, digest in config['source_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    torch.set_num_threads(1)
    cfg = LMConfig(**config['lm'])
    data = TextData(config['data'], cfg)
    assert data.sha256 == config['data_sha256']
    assert {k: len(v) for k, v in data.splits.items()} == config['split_lengths']
    model = make_model(data, cfg, 0)
    by_key = {(r['seed'], r['method'], r['replica']): r for r in runs}
    assert len(by_key) == len(runs)
    counts = Counter()
    exchanges = 0
    for arm in outcomes:
        seed, method = arm['seed'], arm['method']
        n = 1 if method == 'adamw' else 4
        batch_sizes = [32*n, 32]
        rows = [by_key[seed, method, i] for i in range(n)]
        per_arm = read(root/f'{seed}_{method}.json')
        assert arm == per_arm['outcome'] and rows == per_arm['runs']
        assert arm['parameter_count'] == sum(p.numel() for p in model.parameters()) == 108352
        assert sum(r['gradient_evaluations'] for r in rows) == arm['gradient_evaluations'] == 400
        assert [arm['validation_forwards'], arm['test_forwards']] == batch_sizes
        assert arm['cost'] == {'real_train_units': 1200, 'virtual_train_units': 0,
                               'real_probe_units': sum(batch_sizes), 'virtual_probe_units': 0}
        selected = min(range(n), key=lambda i: rows[i]['validation_loss'])
        assert arm['selected_replica'] == selected
        assert rows[selected]['test_loss'] == arm['final_test_loss']
        assert rows[selected]['validation_loss'] == arm['selected_validation_loss']
        counts.update(gradients=arm['gradient_evaluations'], validation_forwards=arm['validation_forwards'],
                      test_forwards=arm['test_forwards'])
        for i, row in enumerate(rows):
            assert not row['failed'] and row['stop_reason'] == 'budget_reached'
            assert row['stream_seed'] == 4*seed+i
            assert row['gradient_evaluations'] == len(row['training_history']) == 400//n
            assert row['final_training_minibatch_loss'] == row['training_history'][-1][0]
            assert row['historical_best_training_minibatch_loss'] == min(p[0] for p in row['training_history'])
            assert row['selected'] == (i == selected) == (row['test_loss'] is not None)
            checkpoint = root/row['checkpoint']
            assert file_hash(checkpoint) == row['checkpoint_sha256']
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            assert model_hash(model) == row['final_sha256']
        if method != 'exchange':
            assert not arm['events']
            continue
        expected = [(step, list(pair)) for k, step in enumerate(SCHEDULE) for pair in PAIRINGS[k % len(PAIRINGS)]]
        assert [(e['step'], e['pair']) for e in arm['events']] == expected
        for e in arm['events']:
            assert e['buffer_before'][::-1] == e['buffer_after']
            assert e['parameter_before'] == e['parameter_after']
            assert e['buffer_before'][0] != e['buffer_before'][1] and e['nontrivial']
            assert e['velocity_difference_norm'] > 0
            assert e['kinetic_before'][::-1] == e['kinetic_after']
            assert sum(e['kinetic_transfer']) == 0
            assert 0 <= max(e['momentum_inf_residual'], e['kinetic_residual']) <= TOL
            assert e['external_impulse'] == e['external_work'] == e['dissipated_energy'] == 0
            if e['step'] == SCHEDULE[0]:
                assert e['parameter_before'] == [rows[i]['step20_sha256'] for i in e['pair']]
            exchanges += 1
        for i, row in enumerate(rows):
            control = by_key[seed, 'independent', i]
            assert row['initial_sha256'] == control['initial_sha256']
            assert row['step20_sha256'] == control['step20_sha256']
            assert row['training_history'][:20] == control['training_history'][:20]
        assert rows[0]['initial_sha256'] == by_key[seed, 'adamw', 0]['initial_sha256']
    assert not (root/'failure.json').exists() and read(root/'summary.json') == summarize(outcomes)
    return {'passed': True, 'failed_runs': [], 'trained_replicas': len(runs), 'method_outcomes': len(outcomes),
            'paired_groups': len(seeds), 'verified_exchanges': exchanges, 'source_members_checked': len(config['source_sha256']),
            'checkpoints_verified': len(runs), 'counts': dict(counts), 'additional_model_queries': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.output)
    _json(arguments.output/'audit.json', result)
    print(json.dumps(result, indent=2))
