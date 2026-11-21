import numpy as np
import pytest

from collision_optim import (
    ActiveConfig, ActiveReflection, Budget, CountedObjective, EvaluationBudgetExceeded,
    HybridConfig, HybridContact, MomentumRestart, MomentumSGD, Quadratic, State, run,
)


def test_velocity_form_matches_classical_momentum_sgd():
    cfg = ActiveConfig(h=0.08, beta=0.9, gradient_scale=0.7)
    method, oracle = MomentumSGD(cfg), CountedObjective(Quadratic())
    state, theta, buffer = State([1.0, -2.0], [0.0, 0.0]), np.array([1.0, -2.0]), np.zeros(2)
    for _ in range(25):
        buffer = cfg.beta * buffer + theta
        theta = theta - cfg.gradient_scale * cfg.h**2 * buffer
        state = method.step(oracle, state).state
        np.testing.assert_allclose(state.x, theta, atol=1e-14)


def test_patience_permits_brief_uphill_motion_then_reflects():
    method = ActiveReflection(ActiveConfig(h=0.01, beta=1, epsilon_f=1e-4, patience=3))
    oracle, state = CountedObjective(Quadratic()), State([1.0], [10.0])
    for index in range(3):
        outcome = method.step(oracle, state)
        assert (outcome.diagnostics["intervention"] == "reflect") == (index == 2)
        if index == 2:
            assert outcome.state.v[0] < 0
            assert outcome.diagnostics["intervention_energy"] == pytest.approx(0.0, abs=1e-12)
        else:
            assert outcome.state.v[0] > 0
        state = outcome.state


def test_patience_resets_after_a_nonviolating_step():
    method = ActiveReflection(ActiveConfig(h=0.01, beta=1, epsilon_f=0, patience=2))
    oracle = CountedObjective(Quadratic())
    assert method.step(oracle, State([1.0], [2.0])).diagnostics["streak"] == 1
    assert method.step(oracle, State([-1.0], [2.0])).diagnostics["streak"] == 0
    assert method.step(oracle, State([1.0], [2.0])).diagnostics["intervention"] == "none"


def test_tolerance_equality_does_not_trigger():
    cfg = ActiveConfig(h=0.1, beta=1, patience=1)
    state = State([1.0], [1.0])
    y = state.x + cfg.h * (state.v - cfg.h * state.x)
    change = (Quadratic().value(y) - 0.5) / 1.5
    method = ActiveReflection(ActiveConfig(h=0.1, beta=1, patience=1, epsilon_f=change))
    assert method.step(CountedObjective(Quadratic()), state).diagnostics["intervention"] == "none"


def test_zero_gradient_skips_direction_test():
    result = ActiveReflection(ActiveConfig(h=0.1, beta=1, patience=1, epsilon_f=0)).step(
        CountedObjective(Quadratic()), State([0.0], [1.0]))
    assert result.diagnostics["direction_cosine"] is None
    assert result.diagnostics["intervention"] == "none"


def test_restart_and_reflect_share_trigger_but_differ_in_response():
    cfg = ActiveConfig(h=0.1, beta=1, patience=1, epsilon_f=0)
    state = State([1.0], [2.0])
    reflected = ActiveReflection(cfg).step(CountedObjective(Quadratic()), state)
    restarted = MomentumRestart(cfg).step(CountedObjective(Quadratic()), state)
    assert reflected.diagnostics["intervention"] == "reflect"
    assert restarted.diagnostics["intervention"] == "restart"
    np.testing.assert_allclose(restarted.state.v, [-0.1])
    assert not np.allclose(reflected.state.v, restarted.state.v)


def test_adaptive_uncertainty_threshold_and_diagnostic_cost():
    changes = iter([0.01, 0.01, 0.09, 0.09])
    def paired(x, y, rng):
        base = 0.5
        return base, base + next(changes) * (1 + abs(base))
    cfg = ActiveConfig(h=0.1, beta=1, patience=1, epsilon_f=0.001,
                       adaptive=True, kappa=3, diagnostic_pairs=4)
    oracle = CountedObjective(Quadratic())
    outcome = ActiveReflection(cfg, paired_sampler=paired).step(oracle, State([1.0], [2.0]))
    assert outcome.diagnostics["paired_change"] == pytest.approx(0.05)
    assert outcome.diagnostics["diagnostic_se"] > 0
    assert outcome.diagnostics["loss_threshold"] > outcome.diagnostics["paired_change"]
    assert outcome.diagnostics["intervention"] == "none"
    assert oracle.values == 10  # Two true values plus four two-point diagnostics.


def test_deterministic_adaptive_threshold_has_zero_sampling_uncertainty():
    cfg = ActiveConfig(h=0.1, beta=1, patience=1, epsilon_f=0.001, adaptive=True)
    outcome = ActiveReflection(cfg).step(CountedObjective(Quadratic()), State([1.0], [2.0]))
    assert outcome.diagnostics["diagnostic_se"] == 0
    assert outcome.diagnostics["loss_threshold"] == cfg.epsilon_f
    assert outcome.diagnostics["intervention"] == "reflect"


def test_failed_budget_step_does_not_commit_streak_or_change_input():
    method = ActiveReflection(ActiveConfig(h=0.1, beta=1, patience=3, epsilon_f=0))
    method.streak = 2
    state = State([1.0], [2.0])
    with pytest.raises(EvaluationBudgetExceeded):
        method.step(CountedObjective(Quadratic(), Budget(values=2)), state)
    assert method.streak == 2
    np.testing.assert_array_equal(state.x, [1.0])
    np.testing.assert_array_equal(state.v, [2.0])


def test_budget_stop_keeps_only_completed_states_and_counts_spent_calls():
    oracle = CountedObjective(Quadratic(), Budget(values=6, gradients=100))
    result = run(MomentumSGD(), oracle, State([1.0], [0.0]), 100)
    assert result.stop_reason.startswith("budget:")
    assert result.rows[-1]["step"] == 2
    assert oracle.values == 6  # Includes one value spent on the unfinished third step.
    assert len(result.rows) == 3


@pytest.mark.parametrize("hybrid", [False, True])
def test_energy_ledger_closes_without_hiding_integration_residual(hybrid):
    oracle = CountedObjective(Quadratic())
    if hybrid:
        method = HybridContact(HybridConfig(h=0.02, beta=0.99))
        initial = State([0.7], [0.3], z=0.9, vz=-0.3)
    else:
        method = ActiveReflection(ActiveConfig(h=0.02, beta=0.99, patience=1, epsilon_f=0))
        initial = State([0.7], [0.3])
    result = run(method, oracle, initial, 80)
    assert result.stop_reason == "completed"
    e0 = result.rows[0]["energy"]
    for row in result.rows:
        assert row["energy"] - e0 == pytest.approx(
            row["cumulative_damping"] + row["cumulative_intervention"] + row["cumulative_residual"],
            abs=1e-12,
        )


def test_conservative_energy_error_decreases_with_step_size():
    errors = []
    for h in [0.1, 0.05, 0.025]:
        result = run(HybridContact(HybridConfig(h=h, beta=1)), CountedObjective(Quadratic()),
                     State([1.0], [0.0], z=2.0, vz=0.0), round(1 / h))
        assert result.stop_reason == "completed"
        errors.append(abs(result.rows[-1]["energy"] - result.rows[0]["energy"]))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


@pytest.mark.parametrize("kwargs", [{"h": 0}, {"beta": 1.1}, {"patience": 0}, {"epsilon_c": 1}])
def test_invalid_active_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ActiveConfig(**kwargs)

