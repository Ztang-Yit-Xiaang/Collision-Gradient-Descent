"""Complete the test-loss decomposition on six archived NanoGPT checkpoints."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import platform
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from collision_optim.nanogpt_benchmark import (
    LMConfig, TextData, TrainingCost, make_model, model_hash, probe_loss,
)
from collision_optim.strategic_experiment import _csv, _json

PARENT = Path('output/final_nanogpt_exchange')
SEEDS = (5110, 5111, 5113, 5115, 5117, 5118)
DEPENDENCIES = (
    'run_final_nanogpt_exchange.py',
    'src/collision_optim/nanogpt_benchmark.py',
    'src/collision_optim/strategic_experiment.py',
    'third_party/nanogpt/model.py',
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def evaluate(root):
    cfg = read(PARENT/'config.json')
    assert digest(PARENT/'source.zip') == cfg['source_archive_sha256']
    archived_hashes = {str(Path(path).relative_to(Path.cwd())) if Path(path).is_absolute() else path: value
                       for path, value in cfg['source_sha256'].items()}
    for path in DEPENDENCIES:
        assert digest(path) == archived_hashes[path], path
    outcomes = {(r['seed'], r['method']): r for r in read(PARENT/'paired_outcomes.json')}
    runs = {(r['seed'], r['method'], r['replica']): r for r in read(PARENT/'runs.json')}
    chosen = tuple(s for s in cfg['seeds'] if
                   outcomes[s, 'exchange']['final_test_loss'] > outcomes[s, 'independent']['final_test_loss']
                   and outcomes[s, 'exchange']['selected_replica'] != outcomes[s, 'independent']['selected_replica'])
    assert chosen == SEEDS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.__version__ == cfg['environment']['torch'] and np.__version__ == cfg['environment']['numpy']
    assert str(torch.get_default_dtype()) == cfg['environment']['dtype']
    lm = LMConfig(**cfg['lm'])
    data = TextData(Path(cfg['data']), lm)
    assert data.sha256 == cfg['data_sha256']
    assert {k: len(v) for k, v in data.splits.items()} == cfg['split_lengths']
    count, batch_seed = cfg['eval_batches_per_candidate'], cfg['evaluation_seed']
    assert (count, batch_seed) == (32, 71001)
    batches = [data.batch('test', batch_seed, i) for i in range(count)]
    batch_hashes = [sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() for x, y in batches]

    root.mkdir(parents=True, exist_ok=False)
    files = sorted(set(DEPENDENCIES) | {str(Path(__file__)), *[str(p) for p in Path('src').rglob('*.py')]})
    manifest = {path: digest(path) for path in files}
    with ZipFile(root/'source.zip', 'x', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path)
    _json(root/'source_manifest.json', manifest)
    _json(root/'config.json', {
        'stage': 'checkpoint_evaluation', 'seed_groups': chosen, 'parent': str(PARENT),
        'parent_hashes': {name: digest(PARENT/name) for name in ('config.json', 'runs.json', 'paired_outcomes.json', 'source.zip')},
        'lm': cfg['lm'], 'data': cfg['data'], 'data_sha256': data.sha256, 'split_lengths': cfg['split_lengths'],
        'evaluation_seed': batch_seed, 'test_batch_sha256': batch_hashes, 'test_forwards_per_checkpoint': count,
        'test_forward_budget': len(chosen)*count, 'training_budget': 0, 'source_sha256': manifest,
        'source_archive_sha256': digest(root/'source.zip'), 'command': [sys.executable, *sys.argv],
        'environment': {**cfg['environment'], 'python': sys.version, 'platform': platform.platform(), 'PYTHONPATH': 'src'},
    })
    cost = TrainingCost(len(chosen)*count)
    rows = []
    for seed in chosen:
        try:
            selected, control = (outcomes[seed, method] for method in ('exchange', 'independent'))
            i, j = selected['selected_replica'], control['selected_replica']
            candidate = runs[seed, 'exchange', j]
            checkpoint = PARENT/candidate['checkpoint']
            assert candidate['test_loss'] is None and not candidate['selected']
            assert candidate['initial_sha256'] == runs[seed, 'independent', j]['initial_sha256']
            assert digest(checkpoint) == candidate['checkpoint_sha256']
            model = make_model(data, lm, candidate['stream_seed'])
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            assert model_hash(model) == candidate['final_sha256']
            before = cost.used
            per_batch = [probe_loss(model, [batch], cost) for batch in batches]
            assert model_hash(model) == candidate['final_sha256'] and cost.used-before == count
            fixed = float(np.mean(per_batch))
            s, n = selected['final_test_loss'], control['final_test_loss']
            assert isclose((fixed-n)+(s-fixed), s-n, abs_tol=1e-12)
            rows.append({'seed': seed, 'independent_index': j, 'exchange_index': i,
                         'independent_selected_test_loss': n, 'exchange_fixed_index_test_loss': fixed,
                         'exchange_selected_test_loss': s, 'fixed_index_gap': fixed-n,
                         'reselection_gap': s-fixed, 'selected_gap': s-n,
                         'checkpoint': str(checkpoint), 'checkpoint_sha256': candidate['checkpoint_sha256'],
                         'model_sha256': candidate['final_sha256'], 'test_minibatch_losses': per_batch,
                         'test_forwards': count, 'gradient_evaluations': 0})
            _json(root/'evaluations.json', rows)
        except Exception as exc:
            _json(root/'failure.json', {'seed': seed, 'reason': repr(exc), 'cost': cost.counts})
            raise
    assert cost.used == cost.limit == cost.counts['real_probe_units']
    assert all(v == 0 for k, v in cost.counts.items() if k != 'real_probe_units')
    _csv(root/'evaluations.csv', [{k: v for k, v in row.items() if k != 'test_minibatch_losses'} for row in rows])
    summary = {'checkpoints_evaluated': len(rows), 'gradient_evaluations': 0,
               'test_forwards': cost.counts['real_probe_units'], 'cost': cost.counts, 'failures': [],
               'gaps': {k: {'mean': float(np.mean([r[k] for r in rows])), 'positive': sum(r[k] > 0 for r in rows),
                            'negative': sum(r[k] < 0 for r in rows)}
                        for k in ('fixed_index_gap', 'reselection_gap', 'selected_gap')}}
    _json(root/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args().output)
