from copy import deepcopy
import numpy as np
import pytest
from collision_optim.timing import (CheckpointPopulation,resume_off,protected_population,multistart,
    fit_ridge,predict_ridge,BASE_FEATURES,PHYSICS_FEATURES)
from collision_optim.cooling import CoolingPopulation
from collision_optim.population import PopulationConfig,PopulationOracle,run_population
from collision_optim.population_experiment import make_objective,initial_population


@pytest.mark.parametrize('rule',['elastic','random'])
def test_checkpoint_observation_leaves_dynamics_and_cost_unchanged(rule):
    obj,_,_=make_objective('quadratic');initial=initial_population('quadratic',obj,790,rho_max=.24)
    cfg=PopulationConfig(rho=.24);f0=min(obj.value(p.x) for p in initial.particles)
    method=CheckpointPopulation(cfg,rule=rule,seed=9,thresholds=[100,200,300],budget=1000,initial_value=f0)
    new=run_population(method,PopulationOracle(obj,total_budget=1000),initial,80)
    old=run_population(CoolingPopulation(cfg,schedule='ramp',rule=rule,seed=9),PopulationOracle(obj,total_budget=1000),initial,80)
    assert new.counts==old.counts and new.stop_reason==old.stop_reason
    for a,b in zip(new.rows,old.rows):assert a==b
    assert len(method.snapshots)==3
    for threshold,s in method.snapshots.items():
        assert s['calls']>=threshold and s['feature_last_observation_calls']==s['calls']
        assert s['history_step_range'][1]==s['step'] and s['history_step_range'][0]<=s['step']
        assert set(s['features'])==set(BASE_FEATURES+PHYSICS_FEATURES)
        assert [p.velocity.tolist() for p in s['state'].particles]==new.rows[s['step']]['velocities']
        child,oracle=resume_off(s)
        assert not child.config.collisions and child.resuming and child.committed_steps==s['step']
        assert child.rng.bit_generator.state==s['rng']


@pytest.mark.parametrize('case',['quadratic','double_well','rosenbrock'])
def test_protection_state_good_parameter_controlled_collision_and_budget(case):
    obj,minimum,_=make_objective(case);state=protected_population(case,obj,minimum,790)
    assert abs(state.kinetic-.6)<1e-12 and np.linalg.norm(state.momentum)<1e-12
    assert np.linalg.norm(obj.gradient(state.particles[0].x))<1e-12
    a,b=state.particles[:2];delta=np.r_[a.x-b.x,a.z-b.z];relative=a.velocity-b.velocity
    assert np.isclose(np.linalg.norm(delta),.26) and delta@relative<0
    assert a.z==b.z and a.vz==b.vz==0
    assert min(p.z-obj.value(p.x) for p in state.particles)>0


@pytest.mark.parametrize('family',['gd','momentum'])
def test_local_baseline_updates_and_exact_paid_commits(family):
    obj,_,_=make_objective('quadratic');initial=initial_population('quadratic',obj,790,rho_max=.24)
    rows,reason,counts,attempts=multistart(PopulationOracle(obj,total_budget=23),initial,family=family,eta=.02)
    assert counts['total_oracle_calls']==23 and len(rows)-1==9 and len(attempts)==1
    for row in rows[1:]:
        g=np.array(row['gradient']);old=np.array(row['buffer_before'])
        direction=g if family=='gd' else .9*old+g
        assert np.array_equal(direction,row['direction'])
        assert np.allclose(np.array(row['positions'])[row['particle']],np.array(row['before'])-.02*direction)


def test_ridge_preprocessing_is_frozen_to_training():
    rows=[{'features':{'x':float(i)},'value':2.*i} for i in range(4)]
    model=fit_ridge(rows,['x']);before=deepcopy(model)
    predicted=predict_ridge(model,[{'features':{'x':1e6}}])
    assert model==before and model['mean']==[1.5]
    assert np.isfinite(predicted).all() and predicted[0]>1e5


def test_development_selection_prefers_prespecified_tie_and_penalizes_failures():
    from analyze_timing import select
    runs=[{'case':'quadratic','rule':'independent' if f==0 else 'elastic','fraction':f,
           'failed':False,'relative_final_gap':1.} for f in (0.,.25,.5,.75,1.)]
    baselines=[{'case':'quadratic','family':family,'eta':eta,'failed':False,'relative_final_gap':1.}
               for family in ('gd','momentum') for eta in (.02,.05)]
    assert select(runs,baselines)['fraction']==.5
    runs[2]['failed']=True
    selected=select(runs,baselines)
    assert selected['fraction']==.25 and selected['baseline_rates']['quadratic']['gd']==.02
