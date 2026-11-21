"""Check that removing the intervention recovers the intended base dynamics."""

import numpy as np
import pytest

from collision_optim import (
    ActiveConfig, ActiveReflection, CountedObjective, HybridConfig, HybridContact,
    MomentumSGD, Quadratic, State, run,
)
from collision_optim.highdim_objectives import CoupledDoubleWell


@pytest.mark.parametrize("objective", [Quadratic(), CoupledDoubleWell(17)])
def test_disabling_both_reflection_mechanisms_recovers_momentum_dynamics(objective):
    rng = np.random.default_rng(77)
    x, v = rng.normal(0, 0.1, 17), rng.normal(0, 0.03, 17)
    cfg = ActiveConfig(h=0.02, beta=0.995, gradient_scale=0.5, epsilon_f=1e9)
    plain = run(MomentumSGD(cfg), CountedObjective(objective), State(x, v), 100)
    inactive = run(ActiveReflection(cfg), CountedObjective(objective), State(x, v), 100)
    # A distant surface cannot collide during this trajectory. The vertical
    # state is a spectator, so the horizontal dynamics must match ordinary SGD.
    hybrid = run(HybridContact(HybridConfig(h=0.02, beta=0.995, mixing=0.5)),
                 CountedObjective(objective), State(x, v, objective.value(x)+10000, 0), 100)
    assert hybrid.stop_reason == inactive.stop_reason == plain.stop_reason == "completed"
    assert not hybrid.contacts
    np.testing.assert_allclose(inactive.final_state.x, plain.final_state.x, atol=1e-14)
    np.testing.assert_allclose(hybrid.final_state.x, plain.final_state.x, atol=1e-14)
    np.testing.assert_allclose(hybrid.final_state.v, plain.final_state.v, atol=1e-14)
    assert hybrid.best_value == pytest.approx(plain.best_value, abs=1e-14)

