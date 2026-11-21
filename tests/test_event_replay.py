from copy import deepcopy
import numpy as np
import pytest

from collision_optim.core import State
from collision_optim.cooling import CoolingPopulation, drag_rate
from collision_optim.event_replay import (MAGNITUDE_KEYS, DIRECTION_KEYS, pack,
    reconstruct_contact, mapped_state, contact_features, ReplayPopulation, run_replay, outcomes)
from collision_optim.gravity import GravityConfig
from collision_optim.objectives import Affine
from collision_optim.population import PopulationConfig, PopulationOracle, PopulationState, run_population


@pytest.fixture(scope='module')
def contact():
    objective = Affine(np.zeros(2))
    state = PopulationState(tuple(State([x, 0.], [v, 0.], 2., 0.)
                                  for x, v in [(-.2,.2),(.2,-.2),(-3.,0.),(3.,0.)]))
    method = CoolingPopulation(PopulationConfig(physics=GravityConfig(), rho=.24), schedule='ramp')
    run = run_population(method, PopulationOracle(objective), state, 20)
    index = next(i for i, e in enumerate(run.events) if e['kind']=='pair')
    incoming, context = reconstruct_contact(run.rows, run.events, index)
    return objective, run, index, incoming, context


def test_contact_reconstruction_matches_saved_event_and_entire_step(contact):
    _, run, index, incoming, context = contact
    event = run.events[index]
    assert context['reconstruction_error'] < 1e-12
    for name in ('i', 'j'):
        particle = incoming.particles[event[name]]
        np.testing.assert_array_equal(particle.velocity, event['before_'+name])
        np.testing.assert_array_equal(np.r_[particle.x,particle.z], event['point_'+name])


def test_reconstruction_rejects_wrong_drag_history(contact):
    _, run, index, _, context = contact
    rows = deepcopy(run.rows); rows[context['source_step']]['drag_beta'] = .5
    with pytest.raises(ValueError, match='reconstruction failed'):
        reconstruct_contact(rows, run.events, index)


@pytest.mark.parametrize('rule', ['elastic','random','none'])
def test_maps_preserve_pair_energy_linear_momentum_and_other_particles(contact, rule):
    _, _, _, incoming, context = contact
    state, record = mapped_state(incoming, context, rule, seed=72)
    assert abs(state.kinetic-incoming.kinetic) < 1e-14
    np.testing.assert_allclose(state.momentum, incoming.momentum, atol=1e-14)
    for k in (2,3):
        np.testing.assert_array_equal(state.particles[k].velocity, incoming.particles[k].velocity)
    if rule == 'none':
        assert pack(state) == pack(incoming)
    elif rule == 'elastic':
        assert record['angular_impulse_norm'] < 1e-14


def test_partial_resume_has_no_double_drag_and_keeps_original_clock(contact):
    objective, _, _, incoming, context = contact
    method = ReplayPopulation(context); oracle = PopulationOracle(objective)
    result = method.step(oracle, incoming)
    assert result.diagnostics['partial_source_step']
    assert result.diagnostics['drag_beta'] == 1.
    assert result.diagnostics['damping_energy'] == 0.
    np.testing.assert_allclose(result.state.particles[0].v, incoming.particles[0].v)
    assert abs(result.diagnostics['duration']-(.05-context['local_time'])) < 1e-14
    second = method.step(oracle, result.state)
    clock = context['source_step']*.05
    assert abs(second.diagnostics['absolute_start_time']-clock) < 1e-14
    assert second.diagnostics['drag_beta'] == pytest.approx(np.exp(-drag_rate('ramp',clock,.05)*.05))


def test_angular_diagnostic_has_position_times_impulse_units(contact):
    _, _, _, incoming, context = contact
    state, record = mapped_state(incoming, context, 'random', seed=73)
    def angular(pop):
        return sum((np.outer(np.r_[p.x,p.z],p.velocity)-np.outer(p.velocity,np.r_[p.x,p.z])
                    for p in pop.particles),np.zeros((3,3)))
    assert record['angular_impulse_norm']==pytest.approx(np.linalg.norm(angular(state)-angular(incoming)),abs=1e-14)


def test_ghost_control_can_pass_through_and_query_charge_is_explicit(contact):
    objective, _, _, incoming, context = contact
    run = run_replay(objective, incoming, context, 'none', steps=20)
    assert run['stop_reason']=='completed'
    assert run['rows'][0]['total_oracle_calls']==10
    assert run['rows'][0]['actual_branch_calls']==4
    assert run['counts']['actual_branch_calls']==run['counts']['total_oracle_calls']-6
    assert not any(e['kind']=='pair' for e in run['events'])
    assert run['rows'][-1]['positions'][0][0] > run['rows'][-1]['positions'][1][0]


def test_features_use_only_incoming_state_and_paid_current_probes(contact):
    _, _, _, incoming, context = contact
    values = [1.,2.,3.,4.]; gradients = [np.zeros(2),np.array([1.,0.])]
    features = contact_features(incoming,context,values,gradients,1.)
    assert set(features)==set(MAGNITUDE_KEYS+DIRECTION_KEYS)
    assert features['gradient_zero_i']==1.
    assert features['gradient_impulse_cos_i']==0.
    assert all(np.isfinite(list(features.values())))
    altered = {**context,'future_loss':-1e9,'minimum':-1e9}
    assert contact_features(incoming,altered,values,gradients,1.)==features


def test_outcomes_separate_current_best_and_partial_budget_views(contact):
    objective, _, _, incoming, context = contact
    run = run_replay(objective,incoming,context,'elastic',steps=20)
    views = outcomes(run['rows'],context,1.,0.,.1,None)
    assert views['boundary10']['step']==10
    assert views['boundary50'] is None
    assert views['budget1000']['charged_calls'] <= 1000
    assert views['end']['pair_best_loss'] <= views['end']['pair_final_loss']


def test_drag_integral_matches_when_observation_step_changes():
    exposures = [sum(drag_rate('ramp',k*h,h)*h for k in range(round(10/h)))
                 for h in (.05,.025,.0125)]
    np.testing.assert_allclose(exposures,exposures[0],atol=1e-13)
