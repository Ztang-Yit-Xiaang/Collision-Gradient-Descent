import numpy as np
import pytest

from collision_optim import (
    Affine, CountedObjective, HybridConfig, HybridContact, Quadratic,
    Rosenbrock, State, TiltedDoubleWell, elastic_pair_collision,
    first_contact_time, specular_reflection,
)
from collision_optim.hybrid import ContactResolutionError


@pytest.mark.parametrize("slope", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
def test_oblique_plane_redirects_vertical_momentum(slope):
    normal = np.array([-slope, 1.0])
    out = specular_reflection([0.0, -1.0], normal)
    expected = np.array([-2 * slope, 1 - slope**2]) / (1 + slope**2)
    np.testing.assert_allclose(out, expected, atol=1e-14)
    assert np.dot(out, normal) > 0
    assert np.dot(out, out) == pytest.approx(1.0, abs=1e-14)


def test_random_wall_and_pair_invariants():
    rng = np.random.default_rng(12)
    for _ in range(250):
        normal = rng.normal(size=8)
        normal /= np.linalg.norm(normal)
        vi, vj = rng.normal(size=(2, 8))
        reflected = specular_reflection(vi, normal)
        np.testing.assert_allclose(reflected @ normal, -(vi @ normal), atol=1e-13)
        np.testing.assert_allclose(reflected - (reflected @ normal) * normal,
                                   vi - (vi @ normal) * normal, atol=1e-13)
        assert reflected @ reflected == pytest.approx(vi @ vi, abs=1e-12)
        left, right = elastic_pair_collision(vi, vj, normal)
        np.testing.assert_allclose(left + right, vi + vj, atol=1e-13)
        assert left @ left + right @ right == pytest.approx(vi @ vi + vj @ vj, abs=1e-12)


@pytest.mark.parametrize("objective,x,dx", [
    (Affine(np.array([0.3, -1.0])), np.array([0.7, 1.2]), np.array([2.0, -0.2])),
    (Quadratic(), np.array([0.7, 1.2]), np.array([2.0, -0.2])),
    (TiltedDoubleWell(), np.array([0.7, 1.2]), np.array([2.0, -0.2])),
    (Rosenbrock(), np.array([0.7, 1.2]), np.array([2.0, -0.2])),
])
def test_objectives_gradients_and_drift_geometry(objective, x, dx):
    eps = 1e-6
    numeric = np.array([(objective.value(x + eps * axis) - objective.value(x - eps * axis)) / (2 * eps)
                        for axis in np.eye(x.size)])
    np.testing.assert_allclose(objective.gradient(x), numeric, rtol=1e-6, atol=1e-7)
    coefficients = objective.path_coefficients(x, dx)
    for fraction in np.linspace(0, 1, 21):
        assert np.polynomial.polynomial.polyval(fraction, coefficients) == pytest.approx(
            objective.value(x + fraction * dx), abs=1e-10
        )


def test_contact_detector_finds_crossing_when_both_endpoints_are_feasible():
    # z=-0.5 crosses the arch f(x)=-x^2 twice between x=-1 and x=1.
    oracle = CountedObjective(Quadratic(np.array([[-2.0]])))
    hit = first_contact_time(oracle, np.array([-1.0]), -0.5, np.array([2.0]), 0.0,
                             1.0, HybridConfig())
    assert hit == pytest.approx((1 - np.sqrt(0.5)) / 2, abs=1e-12)


def test_outgoing_contact_is_not_reflected_again():
    oracle = CountedObjective(Affine(np.array([0.0])))
    assert first_contact_time(oracle, np.array([0.0]), 0.0, np.array([0.0]), 1.0,
                              0.1, HybridConfig()) is None


def test_initial_incoming_contact_is_handled_at_zero_time():
    oracle = CountedObjective(Affine(np.array([0.0])))
    assert first_contact_time(oracle, np.array([0.0]), 0.0, np.array([0.0]), -1.0,
                              0.1, HybridConfig()) == 0.0


def test_hybrid_uses_contact_normal_not_starting_gradient():
    oracle = CountedObjective(Quadratic())
    method = HybridContact(HybridConfig(h=0.5, beta=1, mixing=0.5))
    outcome = method.step(oracle, State([0.0], [1.0], z=0.1, vz=-0.1))
    assert outcome.events
    event = outcome.events[0]
    assert event.point[0] > 0.0  # Starting gradient was zero.
    expected = np.array([-event.point[0], 1.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(event.normal, expected, atol=1e-12)
    assert event.energy_error == pytest.approx(0.0, abs=1e-12)
    assert outcome.state.z >= outcome.value - method.config.gap_tolerance


def test_zero_loss_gradient_does_not_disable_real_wall_contacts():
    method = HybridContact(HybridConfig(h=0.1, beta=1))
    outcome = method.step(CountedObjective(Affine(np.array([0.0]))),
                          State([0.0], [0.0], z=0.02, vz=-1.0))
    assert len(outcome.events) == 1
    assert outcome.state.vz > 0
    assert outcome.events[0].energy_error == pytest.approx(0.0, abs=1e-14)


def test_grazing_contact_is_reported_instead_of_tunneling():
    with pytest.raises(ContactResolutionError, match="sustained-contact"):
        first_contact_time(CountedObjective(Quadratic()), np.array([0.0]), 0.0,
                           np.array([1.0]), 0.0, 1.0, HybridConfig())


def test_penetrating_initial_state_is_rejected():
    with pytest.raises(ContactResolutionError, match="penetrates"):
        HybridContact().step(CountedObjective(Quadratic()), State([1.0], [0.0], z=0.0, vz=0.0))


def test_generic_black_box_is_not_given_fake_exact_contact_detection():
    class BlackBox:
        def value(self, x): return float(x @ x)
        def gradient(self, x): return 2 * x
    with pytest.raises(NotImplementedError, match="path_coefficients"):
        HybridContact().step(CountedObjective(BlackBox()), State([0.0], [1.0], z=1.0, vz=0.0))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0])
def test_invalid_mixing_is_rejected(bad):
    with pytest.raises(ValueError):
        HybridConfig(mixing=bad)

