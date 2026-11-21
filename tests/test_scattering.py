"""Physical invariants, map distribution and frozen-engine equivalence."""

from dataclasses import replace
import json

import numpy as np
import pytest

from collision_optim.core import State, elastic_pair_collision
from collision_optim.gravity import GravityConfig
from collision_optim.objectives import Affine
from collision_optim.population import PopulationConfig, PopulationGravity, PopulationOracle, PopulationState, run_population
from collision_optim.population_experiment import initial_population, make_objective
from collision_optim.scattering_population import ScatteringPopulation, pair_diagnostics, random_pair_collision


@pytest.mark.parametrize("dimension", [1, 3, 10, 100])
def test_conserving_random_map_has_outgoing_velocity_and_unchanged_inputs(dimension):
    rng = np.random.default_rng(5239+dimension)
    for _ in range(30):
        n = rng.normal(size=dimension); n /= np.linalg.norm(n)
        center = rng.normal(size=dimension); u = rng.normal(size=dimension)
        if u @ n > 0: u = -u
        vi, vj = center+u, center-u; saved = [a.copy() for a in (vi, vj, n)]
        ai, aj = random_pair_collision(vi, vj, n, rng)
        assert np.allclose(ai+aj, vi+vj, atol=1e-12, rtol=1e-12)
        assert np.isclose(ai @ ai+aj @ aj, vi @ vi+vj @ vj, atol=1e-11, rtol=1e-12)
        assert (ai-aj) @ n >= 0
        assert all(np.array_equal(a,b) for a,b in zip((vi,vj,n),saved))


def test_zero_center_preserves_each_speed_and_transfer_formula():
    rng = np.random.default_rng(9239); n = np.array([1.,0.,0.]); u = np.array([-.4,.2,.1])
    for center in (np.zeros(3), np.array([.3,-.2,.1])):
        for _ in range(50):
            vi, vj = center+u, center-u
            ai, aj = random_pair_collision(vi,vj,n,rng)
            diagnostic = pair_diagnostics(vi,vj,ai,aj,n)
            assert diagnostic["transfer_formula_error"] < 1e-15
            if np.linalg.norm(center) == 0:
                assert np.isclose(vi @ vi, ai @ ai, atol=1e-15)
                assert np.isclose(vj @ vj, aj @ aj, atol=1e-15)


def test_uniform_hemisphere_mean_and_reproducible_seed():
    vi,vj,n = np.array([-1.,0.,0.]),np.array([1.,0.,0.]),np.array([1.,0.,0.])
    rng = np.random.default_rng(17)
    samples = np.array([random_pair_collision(vi,vj,n,rng)[0] for _ in range(5000)])
    assert np.allclose(samples.mean(axis=0), [.5,0.,0.], atol=.025)
    left = random_pair_collision(vi,vj,n,np.random.default_rng(5))
    right = random_pair_collision(vi,vj,n,np.random.default_rng(5))
    assert all(np.array_equal(a,b) for a,b in zip(left,right))


def test_1d_random_outgoing_rule_is_the_elastic_rule():
    vi,vj,n = np.array([-.2]),np.array([.7]),np.array([1.])
    physical = elastic_pair_collision(vi,vj,n)
    for seed in range(10):
        random = random_pair_collision(vi,vj,n,np.random.default_rng(seed))
        assert all(np.allclose(a,b,atol=1e-15) for a,b in zip(physical,random))


def test_angular_momentum_distinguishes_physical_and_ablation():
    vi,vj,n = np.array([-.4,.4,0.]),np.array([.4,.4,0.]),np.array([1.,0.,0.])
    a,b = elastic_pair_collision(vi,vj,n)
    assert pair_diagnostics(vi,vj,a,b,n)["angular_impulse_frobenius"] < 1e-15
    a,b = random_pair_collision(vi,vj,n,np.random.default_rng(12))
    assert pair_diagnostics(vi,vj,a,b,n)["angular_impulse_frobenius"] > 1e-3


@pytest.mark.parametrize("enabled", [False, True])
def test_mirrored_scheduler_equals_frozen_original(enabled):
    objective, minimum, boundary = make_objective("quadratic")
    initial = initial_population("quadratic",objective,2)
    cfg = PopulationConfig(collisions=enabled,rho=.24)
    old = run_population(PopulationGravity(cfg),PopulationOracle(objective),initial,80)
    new = run_population(ScatteringPopulation(cfg,rule="elastic"),PopulationOracle(objective),initial,80)
    assert old.stop_reason == new.stop_reason and old.counts == new.counts
    assert old.best_value == new.best_value
    for p,q in zip(old.final_state.particles,new.final_state.particles):
        assert np.array_equal(p.position,q.position) if hasattr(p,"position") else np.array_equal(p.x,q.x) and p.z == q.z
        assert np.array_equal(p.velocity,q.velocity)
    for a,b in zip(old.rows,new.rows):
        assert all(a[k] == b[k] for k in a)


def test_event_kernel_diagnostics_and_full_motion_ledger():
    objective = Affine([0.,0.],-10.)
    initial=PopulationState((State([.5,-.37],[0.,1.],3.,0.),State([-.37,0.],[1.,0.],3.,0.)))
    cfg=PopulationConfig(physics=GravityConfig(h=.05,beta=1),rho=.5)
    for mode in ("elastic","random"):
        method = ScatteringPopulation(cfg,rule=mode,seed=8)
        before_rng = json.dumps(method.rng.bit_generator.state)
        result = run_population(method,PopulationOracle(objective),initial,16)
        assert result.stop_reason == "completed"
        pair = [e for e in result.events if e["kind"]=="pair"]
        assert len(pair) == 1 and len(method.map_attempts)==1
        assert pair[0]["outgoing_normal_speed"] > 0
        assert pair[0]["transfer_formula_error"] < 1e-15
        assert max(abs(r["cumulative_residual"]) for r in result.rows) < 1e-12
        assert max(r["momentum_residual"] for r in result.rows) < 1e-12
        if mode == "elastic": assert json.dumps(method.rng.bit_generator.state) == before_rng


def test_invalid_or_nonincoming_map_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError): random_pair_collision([0.,0.],[1.],[1.],rng)
    with pytest.raises(ValueError): random_pair_collision([0.],[1.],[0.],rng)
    with pytest.raises(ValueError): random_pair_collision([1.],[0.],[1.],rng)
