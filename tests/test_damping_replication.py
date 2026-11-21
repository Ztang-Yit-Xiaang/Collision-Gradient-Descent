"""Fixed-time control and paired analysis checks before the new seed sweep."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from collision_optim.damping_replication import PeriodicTraining, paired_summary
from collision_optim.momentum_routing import RoutingConfig
from collision_optim.nanogpt_benchmark import (LMConfig, TorchControl, TrainingCost,
                                              UnitLimit, make_optimizer, train_step)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer = torch.nn.Linear(2, 1).double(); self.calls = 0

    def forward(self, x, y):
        self.calls += 1; prediction = self.layer(x)
        return prediction, torch.mean((prediction-y)**2)


def test_periodic_matches_manual_native_sgd_without_extra_forwards():
    torch.manual_seed(9); model = Model(); manual = copy.deepcopy(model); cfg = LMConfig()
    opt = make_optimizer(model, "momentum", .03, 0., cfg)
    manual_opt = make_optimizer(manual, "momentum", .03, 0., cfg)
    training = PeriodicTraining(model, opt, cfg, "none", RoutingConfig(), 0, schedule=(2, 4))
    cost, manual_cost = TrainingCost(18), TrainingCost(18); manual_control = TorchControl(manual)
    batch = (torch.tensor([[1., -1.], [1., 2.]], dtype=torch.double), torch.zeros(2, 1, dtype=torch.double))
    for t in range(6):
        if t in (2, 4):
            # Independent native-buffer implementation of scheduled reduction.
            for p in manual.parameters(): manual_opt.state[p]["momentum_buffer"].mul_(.5)
        training.step(batch, cost, t)
        train_step(manual, manual_opt, manual_control, batch, cfg, manual_cost)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), manual.parameters()))
        if t < 2: assert not training.actions
    assert cost.used == manual_cost.used == 18 and model.calls == manual.calls == 6
    assert cost.counts["real_probe_units"] == cost.counts["virtual_train_units"] == 0
    assert [a["step"] for a in training.actions] == [2, 4]
    assert training.router.dampings == 2 and training.router.shots == 0
    assert training.supplied == training.feature_seconds == 0
    assert training.max_actuator_error < 1e-12 and training.dissipated > 0


def test_periodic_cannot_actuate_an_unpayable_update():
    model = Model(); cfg = LMConfig(); opt = make_optimizer(model, "momentum", .03, 0., cfg)
    training = PeriodicTraining(model, opt, cfg, "none", RoutingConfig(), 0, schedule=(0,))
    for b in training.control.base: b.fill_(1.)
    before = [b.clone() for b in training.control.base]
    with pytest.raises(UnitLimit): training.step((None, None), TrainingCost(0), 0)
    assert not training.actions and model.calls == training.router.dampings == 0
    assert all(torch.equal(a, b) for a, b in zip(before, training.control.base))


def test_paired_summary_preserves_seed_pairing_when_rows_are_reordered():
    rows = []
    for seed, delta in enumerate([-.4, -.2, -.1, .1]):
        rows.extend([{"seed": seed, "method": "base", "final_loss": 2.+seed},
                     {"seed": seed, "method": "damp", "final_loss": 2.+seed+delta}])
    result = paired_summary(rows[::-1], "damp", "base")
    assert result == paired_summary(rows, "damp", "base")
    assert np.isclose(result["mean_difference"], -.15)
    assert result["better"] == 3 and result["worse"] == 1 and result["valid_pairs"] == 4
    assert -.4 <= result["bootstrap_95_low"] <= result["bootstrap_95_high"] <= .1+1e-12
    assert [r["seed"] for r in result["seed_differences"]] == [0, 1, 2, 3]


def test_unpaired_summary_rejected_and_failed_pair_explicitly_excluded():
    rows = [{"seed": 1, "method": "base", "final_loss": 2.}]
    with pytest.raises(ValueError, match="Seed sets"):
        paired_summary(rows, "damp", "base")
    rows.append({"seed": 1, "method": "damp", "final_loss": None})
    assert paired_summary(rows, "damp", "base")["excluded_pairs"] == 1
