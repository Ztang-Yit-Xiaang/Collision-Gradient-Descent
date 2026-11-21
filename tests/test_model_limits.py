"""Diagnose model limitations without disguising them as solver failures."""

import numpy as np
import pytest

from collision_optim import (
    ActiveReflection, CountedObjective, HybridContact, State, TiltedDoubleWell,
    run, specular_reflection,
)


@pytest.mark.parametrize("dimension", [1, 10, 1000])
def test_incoming_graph_contact_adds_a_downhill_horizontal_impulse(dimension):
    rng = np.random.default_rng(99)
    g = rng.normal(size=dimension)
    g /= np.linalg.norm(g)
    vx = rng.normal(size=dimension)
    vx *= 0.3 / np.linalg.norm(vx)
    vz = float(g @ vx - 0.4)  # Incoming: vz - grad(f).vx < 0.
    normal = np.r_[-g, 1.0]
    before = np.r_[vx, vz]
    after = specular_reflection(before, normal)
    expected = 2 * (vz - g @ vx) * g / (1 + g @ g)
    np.testing.assert_allclose(after[:-1] - vx, expected, atol=1e-13)
    assert (after[:-1] - vx) @ g < 0


def test_landscape_mechanisms_cannot_create_horizontal_motion_at_exact_saddle_from_rest():
    # f(x,y)=(x^2-1)^2/4+y^2/2 has an exact saddle at (0,0).
    objective = TiltedDoubleWell(tilt=0)
    x = np.zeros(2)
    plain_state = State(x, x)
    active = run(ActiveReflection(), CountedObjective(objective), plain_state, 100)
    hybrid = run(HybridContact(), CountedObjective(objective), State(x, x, 0.45, 0), 100)
    assert active.stop_reason == hybrid.stop_reason == "completed"
    np.testing.assert_array_equal(active.final_state.x, x)
    np.testing.assert_array_equal(hybrid.final_state.x, x)
    assert active.best_value == hybrid.best_value == 0.25

