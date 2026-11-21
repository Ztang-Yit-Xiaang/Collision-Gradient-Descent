import numpy as np
import pytest

from collision_optim import (
    Affine, Budget, CountedObjective, GravityConfig, GravityContact, Quadratic,
    State, ballistic_contact_time, run,
)
from collision_optim.hybrid import ContactResolutionError


def test_free_flight_has_no_horizontal_force_and_exact_gravity_work():
    cfg = GravityConfig(h=0.3, beta=1, gravity=0.7)
    initial = State([1, -2], [0.4, 0.2], z=10, vz=0.3)
    outcome = GravityContact(cfg).step(CountedObjective(Quadratic()), initial)
    np.testing.assert_allclose(outcome.state.x, initial.x+cfg.h*initial.v, atol=1e-14)
    np.testing.assert_array_equal(outcome.state.v, initial.v)
    assert outcome.state.z == pytest.approx(10+0.3*0.3-0.5*0.7*0.3**2)
    assert outcome.state.vz == pytest.approx(0.3-0.7*0.3)
    assert outcome.diagnostics["energy_residual"] == pytest.approx(0, abs=1e-13)
    assert outcome.state.kinetic-initial.kinetic == pytest.approx(outcome.diagnostics["gravity_work"])


def test_flat_ground_bounce_uses_ballistic_impact_time_and_preserves_energy():
    cfg = GravityConfig(h=4.5, beta=1, gravity=0.5)
    outcome = GravityContact(cfg).step(CountedObjective(Affine([0.])),
                                       State([0.], [0.3], z=1., vz=0.))
    assert len(outcome.events) == 1
    event = outcome.events[0]
    assert event.time == pytest.approx(2.0, abs=1e-12)
    np.testing.assert_allclose(event.before, [0.3, -1.0], atol=1e-12)
    np.testing.assert_allclose(event.after, [0.3, 1.0], atol=1e-12)
    assert outcome.state.z == pytest.approx(0.9375, abs=1e-12)
    assert outcome.diagnostics["energy_residual"] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("slope", [-2., -0.4, 0., 0.6, 3.])
def test_sustained_plane_contact_matches_exact_sliding_solution(slope):
    cfg = GravityConfig(h=0.4, beta=1, gravity=0.8, height_scale=1.3)
    method = GravityContact(cfg)
    oracle = CountedObjective(Affine([slope]))
    initial = method.initialize(oracle, [0.7], [0.2], gap=0, on_surface=True)
    outcome = method.step(oracle, initial)
    s = cfg.height_scale*slope
    acceleration = -cfg.gravity*s/(1+s*s)
    expected_v = 0.2+cfg.h*acceleration
    expected_x = 0.7+cfg.h*0.2+0.5*cfg.h**2*acceleration
    np.testing.assert_allclose(outcome.state.x, [expected_x], atol=1e-12)
    np.testing.assert_allclose(outcome.state.v, [expected_v], atol=1e-12)
    assert outcome.state.z == pytest.approx(s*expected_x, abs=1e-12)
    assert outcome.state.vz == pytest.approx(s*expected_v, abs=1e-12)
    assert outcome.diagnostics["sliding_time"] == pytest.approx(cfg.h)
    assert not outcome.events
    assert outcome.diagnostics["energy_residual"] == pytest.approx(0., abs=1e-12)


def test_sustained_flat_rest_never_generates_micro_bounces():
    method = GravityContact(GravityConfig(h=0.1, beta=0.99))
    oracle = CountedObjective(Affine([0.]))
    initial = method.initialize(oracle, [0.], [0.], gap=0, on_surface=True)
    result = run(method, oracle, initial, 30)
    assert result.stop_reason == "completed"
    assert not result.contacts
    assert all(row["z"] == 0 for row in result.rows)
    assert all(row["energy"] == 0 for row in result.rows)


def test_curved_sustained_contact_is_conservative_with_no_damping():
    cfg = GravityConfig(h=0.05, beta=1, energy_tolerance=1e-10)
    method = GravityContact(cfg)
    oracle = CountedObjective(Quadratic())
    initial = method.initialize(oracle, [0.8, 0.3], [0.1, -0.1], gap=0, on_surface=True)
    result = run(method, oracle, initial, 120)
    assert result.stop_reason == "completed"
    assert not result.contacts
    energy = result.rows[0]["energy"]
    assert max(abs(row["energy"]-energy) for row in result.rows) < 2e-8
    assert max(abs(row["gap"]) for row in result.rows[1:]) < 1e-13


def test_negative_support_causes_immediate_liftoff_instead_of_attraction():
    cfg = GravityConfig(h=0.2, beta=1)
    method = GravityContact(cfg)
    oracle = CountedObjective(Quadratic(np.array([[-1.]])))
    initial = method.initialize(oracle, [0.], [1.], gap=0, on_surface=True)
    outcome = method.step(oracle, initial)
    np.testing.assert_allclose(outcome.state.x, [0.2])
    np.testing.assert_allclose(outcome.state.v, [1.])
    assert outcome.state.z == pytest.approx(-0.01)
    assert outcome.diagnostics["gap"] > 0
    assert outcome.diagnostics["sliding_time"] == 0


def test_sliding_detaches_when_normal_support_reaches_zero():
    from numpy.polynomial import Polynomial

    class QuarticHill:
        def value(self, x): return -0.25*x[0]**4
        def gradient(self, x): return -x**3
        def path_coefficients(self, x, displacement):
            return (-0.25*Polynomial([x[0], displacement[0]])**4).coef

    method = GravityContact(GravityConfig(h=0.1, beta=1))
    oracle = CountedObjective(QuarticHill())
    initial = method.initialize(oracle, [0.4], [0.3], gap=0, on_surface=True)
    result = run(method, oracle, initial, 60)
    assert result.stop_reason == "completed", result.stop_reason
    assert sum(row.get("detachments", 0) for row in result.rows) == 1
    assert result.rows[-1]["gap"] > 0.01
    assert max(abs(row["cumulative_residual"]) for row in result.rows) < 1e-7


def test_ballistic_detector_checks_interior_even_when_endpoints_are_above():
    cfg = GravityConfig(gravity=0.1)
    oracle = CountedObjective(Quadratic(np.array([[-2.]])))
    initial = State([-1.], [2.], z=-0.5, vz=0.)
    hit = ballistic_contact_time(oracle, initial, 1., cfg)
    assert hit == pytest.approx((4-np.sqrt(16-4*3.95*0.5))/(2*3.95), abs=1e-12)


def test_initial_energy_uses_gravity_times_height_not_old_mixing_potential():
    cfg = GravityConfig(gravity=0.7, height_scale=1.3)
    initial = State([0.5], [0.2], z=3., vz=-0.1)
    result = run(GravityContact(cfg), CountedObjective(Quadratic()), initial, 0)
    assert result.rows[0]["potential"] == pytest.approx(2.1)
    assert result.rows[0]["energy"] == pytest.approx(2.125)


def test_damping_and_energy_residual_are_separate():
    cfg = GravityConfig(h=0.05, beta=0.9)
    initial = State([0.], [0.4], z=2., vz=-0.1)
    outcome = GravityContact(cfg).step(CountedObjective(Affine([0.])), initial)
    assert outcome.diagnostics["damping_energy"] == pytest.approx((0.9**2-1)*initial.kinetic)
    assert outcome.diagnostics["energy_residual"] == pytest.approx(0., abs=1e-13)


def test_penetration_and_invalid_scales_are_rejected():
    with pytest.raises(ContactResolutionError, match="penetrates"):
        GravityContact().step(CountedObjective(Quadratic()), State([1.], [0.], z=0., vz=0.))
    for kwargs in ({"gravity": 0}, {"height_scale": -1}, {"beta": 1.1}, {"max_events": 0}):
        with pytest.raises(ValueError):
            GravityConfig(**kwargs)


def test_budget_failure_preserves_last_state_and_accounts_for_attempts():
    initial = State([0.], [0.2], z=0., vz=0.)
    oracle = CountedObjective(Affine([0.]), Budget(values=20, gradients=0, paths=20))
    result = run(GravityContact(), oracle, initial, 4)
    assert result.stop_reason.startswith("budget:")
    assert len(result.rows) == 1
    np.testing.assert_array_equal(result.final_state.x, initial.x)
    assert result.counts["value_calls"] > 1


def test_nearly_stationary_quartic_does_not_lose_ballistic_roots():
    from collision_optim import TiltedDoubleWell
    objective=TiltedDoubleWell()
    x=float(max(np.roots([1.,0.,-1.,objective.tilt]).real))
    cfg=GravityConfig(h=.05,beta=1)
    initial=State([x,0.],[1e-17,0.],objective.value(np.array([x,0.]))+.001,-.02)
    hit=ballistic_contact_time(CountedObjective(objective),initial,cfg.h,cfg)
    expected=(-.02+np.sqrt(.02**2+2*cfg.gravity*.001))/cfg.gravity
    assert hit==pytest.approx(expected,abs=1e-12)


def test_quiet_local_minimum_can_bounce_for_many_observation_steps():
    from collision_optim import TiltedDoubleWell
    objective=TiltedDoubleWell()
    x=float(max(np.roots([1.,0.,-1.,objective.tilt]).real))
    initial=State([x,0.],[0.,0.],objective.value(np.array([x,0.]))+.14,-.03)
    result=run(GravityContact(GravityConfig(h=.05)),CountedObjective(objective),initial,400)
    assert result.stop_reason=="completed",result.stop_reason
    assert max(abs(r["cumulative_residual"]) for r in result.rows)<1e-10
