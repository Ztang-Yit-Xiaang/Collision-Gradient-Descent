"""Mechanism and accounting checks, independent of favorable benchmark outcomes."""

from dataclasses import replace

import numpy as np
import pytest

from collision_optim.momentum_routing import (Action, MomentumRouter, RoutingConfig,
                                              run_routed, scale_momentum)
from collision_optim.objectives import Quadratic
from collision_optim.optimizer_controls import (OptimizerConfig, OptimizerState,
                                                optimizer_step, run_optimizer)
from collision_optim.population import PopulationOracle


def observe(router, step, *, loss=1., adverse=False, movement=0.):
    return router.decide(step, loss=loss, movement=movement,
                         relative_velocity=.01 if adverse else 0.,
                         cosine=.8 if adverse else 0., speed_ratio=5. if adverse else 0.)


@pytest.mark.parametrize("mode,factor", [("restart", 0.), ("damp", .5), ("adaptive", .5)])
def test_dissipation_requires_persistent_warning(mode, factor):
    router = MomentumRouter(mode)
    assert observe(router, 0, adverse=True).kind == "continue"
    assert observe(router, 1, adverse=False).kind == "continue"
    action = observe(router, 2, adverse=True)
    assert action.factor == factor
    assert action.kind in {"restart", "damp"}
    for t in range(3, 12):
        assert observe(router, t, adverse=True).kind == "continue"
    assert observe(router, 12, adverse=True).kind != "continue"


@pytest.mark.parametrize("mode", ["restart", "damp"])
def test_low_energy_stagnation_does_not_activate_damping_only(mode):
    router = MomentumRouter(mode, RoutingConfig(window=2))
    assert all(observe(router, t).kind == "continue" for t in range(30))


def test_failed_small_pulse_escalates_without_queries_and_protects_uphill():
    router = MomentumRouter("adaptive", RoutingConfig(window=2))
    actions = [observe(router, t) for t in range(6)]
    assert actions[-1] == Action("pulse", radius_index=0)
    for t in range(6, 9):
        assert observe(router, t, adverse=True).kind == "continue"
    observe(router, 9)
    assert router.events[0]["failed"]
    follow = [observe(router, t) for t in range(10, 14)]
    assert follow[-1] == Action("pulse", radius_index=1, reverse=True)
    assert all(observe(router, t).kind == "continue" for t in range(14, 50))


def test_successful_pulse_does_not_escalate():
    router = MomentumRouter("adaptive", RoutingConfig(window=2))
    for t in range(6): observe(router, t)
    for t in range(6, 14): action = observe(router, t, loss=.5)
    assert not router.events[0]["failed"]
    assert action == Action("pulse", radius_index=0, reverse=False)


def test_noisy_flat_loss_uses_uncertainty_tolerance():
    router = MomentumRouter("random", RoutingConfig(window=4, noise_aware=True))
    for t in range(12): action = observe(router, t, loss=[.1, 1.9, .2, 1.8][t % 4])
    assert router.threshold > .1
    assert action.kind == "pulse"


@pytest.mark.parametrize("factor", [0., .5, 1.])
def test_damping_changes_native_and_external_momentum_with_exact_ledger(factor):
    config = OptimizerConfig(lr=.1)
    state = optimizer_step(OptimizerState.initialize([1., 2.]), np.array([2., -3.]), config)
    state = replace(state, external=np.array([.02, -.04]))
    changed = scale_momentum(state, factor)
    assert np.array_equal(changed.x, state.x)
    assert changed.step == state.step
    assert np.allclose(changed.velocity, factor*state.velocity)
    next_state = optimizer_step(changed, np.array([.3, .4]), config)
    assert np.allclose(next_state.velocity, .9*changed.velocity-.1*np.array([.3, .4]))
    assert np.isclose(.5*(changed.velocity @ changed.velocity-state.velocity @ state.velocity),
                      .5*(factor**2-1)*(state.velocity @ state.velocity))


def test_no_action_matches_preexisting_optimizer_bit_for_bit():
    config = OptimizerConfig(lr=.01)
    objective = Quadratic(np.diag([1., 30.]))
    original = run_optimizer(PopulationOracle(objective, total_budget=120), np.array([.5, -.2]), config)
    summary, trace, actions, _, state = run_routed(PopulationOracle(objective, total_budget=120),
        np.array([.5, -.2]), config, mode="none")
    assert np.array_equal(state.x, original.state.x)
    assert summary["total_oracle_calls"] == 120
    assert summary["planning_calls"] == 0 and not actions
    assert trace[-1]["best_value"] == original.best_value


@pytest.mark.parametrize("mode", ["none", "restart", "damp", "random", "adaptive"])
def test_exact_query_budget_energy_and_momentum_balance(mode):
    objective = Quadratic(np.diag([1., 100.]))
    summary, rows, actions, _, _ = run_routed(PopulationOracle(objective, total_budget=1500),
        np.array([.6, .8]), OptimizerConfig(lr=.015), mode=mode, seed=83)
    assert summary["total_oracle_calls"] == 1500
    assert summary["steps"] == 749 and summary["planning_calls"] == 0
    assert summary["max_energy_balance_error"] < 1e-12
    assert summary["max_momentum_balance_error"] < 1e-12
    assert summary["supplied"] <= summary["supply_cap"]+1e-12
    assert summary["shots"] <= 2 and summary["dampings"] <= 20
    assert not summary["failure"]
    assert len(rows) == 750
    if mode in {"restart", "damp", "adaptive"}:
        assert any(a["kind"] in {"restart", "damp"} for a in actions)
    for a in actions:
        assert a["delta_kinetic"] > 0 if a["kind"] == "pulse" else a["delta_kinetic"] <= 0


def test_good_minimum_is_not_misrepresented_as_certified_safe():
    summary, _, actions, _, _ = run_routed(PopulationOracle(Quadratic(np.eye(2)), total_budget=1500),
        np.zeros(2), OptimizerConfig(lr=.01), seed=5)
    assert summary["shots"] == 2
    assert [a["radius_index"] for a in actions if a["kind"] == "pulse"] == [0, 1]
    assert summary["best_f"] == 0 and summary["supplied"] > 0


def test_reject_routing_on_adam():
    with pytest.raises(ValueError, match="Momentum"):
        run_routed(PopulationOracle(Quadratic(np.eye(2))), [0., 0.], OptimizerConfig(kind="adam"))


def test_torch_none_is_canonical_and_no_extra_forwards():
    torch = pytest.importorskip("torch")
    from collision_optim.nanogpt_benchmark import LMConfig, TorchControl, TrainingCost, make_optimizer, train_step
    from collision_optim.nanogpt_routing import RoutedTraining
    import copy

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(2, 1).double(); self.calls = 0
        def forward(self, x, y):
            self.calls += 1; z = self.linear(x)
            return z, torch.mean((z-y)**2)

    torch.manual_seed(7); model = Model(); canonical = copy.deepcopy(model)
    cfg = LMConfig(); opt = make_optimizer(model, "momentum", .03, 0., cfg)
    old_opt = make_optimizer(canonical, "momentum", .03, 0., cfg)
    training = RoutedTraining(model, opt, cfg, "none", RoutingConfig(), 0)
    control = TorchControl(canonical); cost = TrainingCost(18); old_cost = TrainingCost(18)
    batch = (torch.tensor([[1., -1.], [1., 2.]], dtype=torch.double), torch.zeros(2, 1, dtype=torch.double))
    for t in range(6):
        training.step(batch, cost, t); train_step(canonical, old_opt, control, batch, cfg, old_cost)
    assert model.calls == canonical.calls == 6
    assert cost.counts == old_cost.counts and cost.used == 18
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), canonical.parameters()))


def test_torch_active_path_has_no_extra_forwards_and_correct_actuators():
    torch = pytest.importorskip("torch")
    from collision_optim.nanogpt_benchmark import LMConfig, TrainingCost, make_optimizer
    from collision_optim.nanogpt_routing import RoutedTraining

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.double)); self.calls = 0
        def forward(self, x, y):
            self.calls += 1
            return self.p, torch.sum((self.p-y)**2)

    model = Model(); cfg = LMConfig(); opt = make_optimizer(model, "momentum", .1, 0., cfg)
    training = RoutedTraining(model, opt, cfg, "adaptive", RoutingConfig(), 0)
    cost = TrainingCost(15)
    schedule = {1: Action("pulse"), 3: Action("damp", factor=.5)}
    training.router.decide = lambda step, **kw: schedule.get(step, Action())
    for t in range(5): training.step((None, torch.zeros(2, dtype=torch.double)), cost, t)
    assert model.calls == 5 and cost.used == 15
    assert cost.counts["real_probe_units"] == cost.counts["virtual_train_units"] == 0
    assert training.max_actuator_error < 1e-12 and training.max_recurrence_error < 1e-12
    assert training.supplied > 0 and training.dissipated > 0


def test_torch_scale_edits_native_buffer_and_preserves_position():
    torch = pytest.importorskip("torch")
    from collision_optim.nanogpt_benchmark import TorchControl
    from collision_optim.nanogpt_routing import scale_torch_momentum
    model = torch.nn.Linear(2, 1); opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    model(torch.ones(1, 2)).sum().backward(); opt.step()
    control = TorchControl(model)
    for b, q in zip(control.base, control.external): b.fill_(2.); q.fill_(3.)
    before = [p.detach().clone() for p in model.parameters()]
    buffers = [opt.state[p]["momentum_buffer"].clone() for p in model.parameters()]
    scale_torch_momentum(opt, control, .5)
    for p, old, buffer, b, q in zip(model.parameters(), before, buffers, control.base, control.external):
        assert torch.equal(p, old) and torch.equal(opt.state[p]["momentum_buffer"], .5*buffer)
        assert torch.all(b == 1.) and torch.all(q == 1.5)
