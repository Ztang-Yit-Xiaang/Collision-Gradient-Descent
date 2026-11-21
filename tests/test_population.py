from dataclasses import replace
import numpy as np
import pytest

from collision_optim import Affine, Budget, CountedObjective, GravityConfig, GravityContact, State, run
from collision_optim.hybrid import ContactResolutionError
from collision_optim.population import (
    PopulationConfig, PopulationGravity, PopulationOracle, PopulationState,
    pair_contact_time, run_population,
)


def pair_config(**kwargs):
    return PopulationConfig(physics=GravityConfig(h=0.05, beta=1), **kwargs)


def energy_focusing_initial():
    return PopulationState((State([0.5, -0.37], [0., 1.], 3., 0.),
                            State([-0.37, 0.], [1., 0.], 3., 0.)))


def test_binary_root_and_actual_collision_normal_focus_energy():
    cfg = pair_config(rho=0.5)
    initial = energy_focusing_initial()
    assert pair_contact_time(*initial.particles, 1., cfg) == pytest.approx(.37, abs=1e-13)
    result = run_population(PopulationGravity(cfg), PopulationOracle(Affine([0.,0.], -10.)), initial, 16)
    assert result.stop_reason == "completed"
    pair = [e for e in result.events if e["kind"] == "pair"]
    assert len(pair) == 1
    event = pair[0]
    assert event["time"] == pytest.approx(.37, abs=1e-12)
    np.testing.assert_allclose(event["normal"], [1,0,0], atol=1e-12)
    np.testing.assert_allclose(event["after_i"][:2], [1,1], atol=1e-12)
    np.testing.assert_allclose(event["after_j"][:2], [0,0], atol=1e-12)
    assert event["horizontal_kinetic_gain_i"] == pytest.approx(.5, abs=1e-12)
    assert abs(event["energy_error"]) < 1e-12
    assert event["momentum_error"] < 1e-12
    assert max(abs(r["cumulative_residual"]) for r in result.rows) < 1e-12
    assert max(r["momentum_residual"] for r in result.rows) < 1e-12


def test_head_on_equal_mass_collision_exchanges_normal_velocities():
    initial = PopulationState((State([-1.], [1.], 10., 0.), State([1.], [-1.], 10., 0.)))
    cfg = PopulationConfig(physics=GravityConfig(h=1, beta=1), rho=.4)
    result = run_population(PopulationGravity(cfg), PopulationOracle(Affine([0.])), initial, 1)
    assert result.stop_reason == "completed"
    event = next(e for e in result.events if e["kind"] == "pair")
    assert event["time"] == pytest.approx(.8)
    np.testing.assert_allclose(result.final_state.particles[0].v, [-1.])
    np.testing.assert_allclose(result.final_state.particles[1].v, [1.])


def test_root_finds_collision_with_separated_flight_endpoints():
    cfg = pair_config(rho=.2)
    a, b = State([-1.], [2.], 3., 0.), State([1.], [-2.], 3., 0.)
    assert pair_contact_time(a,b,1.,cfg) == pytest.approx(.45)


def test_outgoing_stationary_and_exact_tangent_pairs_do_not_collide():
    cfg = pair_config(rho=.5)
    assert pair_contact_time(State([-.25],[-1.],3.,0.),State([.25],[1.],3.,0.),1.,cfg) is None
    assert pair_contact_time(State([-1.],[0.],3.,0.),State([1.],[0.],3.,0.),1.,cfg) is None
    assert pair_contact_time(State([-1.,.5],[1.,0.],3.,0.),State([0.,0.],[0.,0.],3.,0.),2.,cfg) is None


def test_initial_incoming_contact_and_overlap_are_distinguished():
    cfg = pair_config(rho=.5)
    assert pair_contact_time(State([-.25],[1.],3.,0.),State([.25],[-1.],3.,0.),1.,cfg) == 0
    with pytest.raises(ContactResolutionError, match="overlap"):
        pair_contact_time(State([0.],[1.],3.,0.),State([.1],[-1.],3.,0.),1.,cfg)


def test_common_gravity_cancels_in_relative_collision_time():
    initial = energy_focusing_initial()
    times=[]
    for gravity in [.1, .5, 9.8]:
        cfg = PopulationConfig(physics=GravityConfig(h=.5,beta=1,gravity=gravity),rho=.5)
        result=run_population(PopulationGravity(cfg),PopulationOracle(Affine([0.,0.],-100)),initial,1)
        times.append(result.events[0]["time"])
    np.testing.assert_allclose(times,.37,atol=1e-12)


def test_noninteracting_population_matches_single_particle_gravity_trajectories():
    cfg=GravityConfig(h=.05,beta=.99)
    initial=PopulationState((State([-.5],[.2],.2,-.1),State([.7],[-.1],.4,.2)))
    population=run_population(PopulationGravity(PopulationConfig(physics=cfg,rho=.1,collisions=False)),
                              PopulationOracle(Affine([0.])),initial,80)
    assert population.stop_reason=="completed"
    for p,actual in zip(initial.particles,population.final_state.particles):
        single=run(GravityContact(cfg),CountedObjective(Affine([0.])),p,80)
        np.testing.assert_allclose(actual.x,single.final_state.x,atol=1e-12)
        np.testing.assert_allclose(actual.velocity,single.final_state.velocity,atol=1e-12)
        assert actual.z==pytest.approx(single.final_state.z,abs=1e-12)


def test_disjoint_simultaneous_wall_events_are_supported_and_wall_impulse_is_external():
    cfg=PopulationConfig(physics=GravityConfig(h=1.,beta=1),rho=.1)
    initial=PopulationState((State([-1.],[0.],.1,-.3),State([1.],[0.],.1,-.3)))
    result=run_population(PopulationGravity(cfg),PopulationOracle(Affine([0.])),initial,1)
    assert result.stop_reason=="completed"
    assert len(result.events)==2
    assert all(e["kind"]=="wall" for e in result.events)
    assert np.linalg.norm(result.rows[-1]["wall_impulse"])>0
    assert result.rows[-1]["momentum_residual"]<1e-12


def test_shared_simultaneous_pair_contacts_stop_without_arbitrary_order():
    initial=PopulationState((State([-1.],[1.],10.,0.),State([0.],[0.],10.,0.),State([1.],[-1.],10.,0.)))
    cfg=PopulationConfig(physics=GravityConfig(h=1.,beta=1),rho=.2)
    result=run_population(PopulationGravity(cfg),PopulationOracle(Affine([0.])),initial,1)
    assert result.stop_reason.startswith("contact_error:")
    assert "share a particle" in result.stop_reason
    assert len(result.rows)==1
    np.testing.assert_array_equal(result.final_state.particles[0].x,initial.particles[0].x)
    assert not result.events


def test_unsupported_sustained_contact_is_reported_instead_of_micro_bouncing():
    initial=PopulationState((State([0.],[0.],0.,0.),))
    result=run_population(PopulationGravity(pair_config()),PopulationOracle(Affine([0.])),initial,1)
    assert result.stop_reason.startswith("contact_error:")
    assert len(result.rows)==1


def test_budget_is_shared_and_attempted_work_survives_rollback():
    initial=energy_focusing_initial()
    oracle=PopulationOracle(Affine([0.,0.],-10.),Budget(),total_budget=5)
    result=run_population(PopulationGravity(pair_config(rho=.5)),oracle,initial,4)
    assert result.stop_reason.startswith("budget:")
    assert len(result.rows)==1
    assert result.counts["total_oracle_calls"]==5
    np.testing.assert_array_equal(result.final_state.particles[0].x,initial.particles[0].x)


def test_pair_prediction_budget_stops_cleanly():
    cfg=replace(pair_config(rho=.5),max_pair_tests=0)
    result=run_population(PopulationGravity(cfg),PopulationOracle(Affine([0.,0.],-10.)),energy_focusing_initial(),1)
    assert result.stop_reason.startswith("budget:")
    assert result.counts["pair_tests"]==0
    assert len(result.rows)==1


def test_crossing_time_is_geometric_and_best_value_is_scored_on_common_grid():
    cfg=PopulationConfig(physics=GravityConfig(h=.5,beta=1),rho=.1,collisions=False)
    initial=PopulationState((State([.8],[-1.],100.,0.),))
    result=run_population(PopulationGravity(cfg),PopulationOracle(Affine([0.],-10.)),initial,3,basin_boundary=.05)
    assert result.first_crossing_times[0]==pytest.approx(.75)
    assert [r["time"] for r in result.rows]==[0.,.5,1.,1.5]


def test_invalid_population_and_configs_are_rejected():
    with pytest.raises(ValueError): PopulationState(())
    with pytest.raises(ValueError): PopulationState((State([0.],[0.]),))
    with pytest.raises(ValueError): PopulationConfig(rho=0)
    with pytest.raises(ValueError): PopulationConfig(max_pair_tests=-1)
