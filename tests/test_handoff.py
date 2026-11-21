from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from collision_optim.cooling import CoolingPopulation
from collision_optim.handoff import HandoffPopulation, partition_state, gd_refine
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.population_experiment import make_objective, initial_population
from collision_optim.objectives import EvaluationBudgetExceeded, Quadratic
from collision_optim.core import State
from collision_optim.population import PopulationState
from collision_optim.hybrid import ContactResolutionError


@pytest.mark.parametrize("rule",["elastic","random"])
def test_committed_stop_and_fork_matches_continuous_states(rule):
    objective,_,_=make_objective("quadratic");initial=initial_population("quadratic",objective,690,rho_max=.24)
    cfg=PopulationConfig(rho=.24)
    method=HandoffPopulation(cfg,schedule="ramp",rule=rule,seed=7,handoff_budget=300)
    oracle=PopulationOracle(objective,total_budget=3000)
    prefix=run_population(method,oracle,initial,80)
    assert prefix.stop_reason=="budget: committed handoff threshold reached"
    assert sum(oracle.counts().values())>=300
    old_counts=oracle.counts();old_step=method.committed_steps
    with pytest.raises(EvaluationBudgetExceeded):method.step(oracle,prefix.final_state)
    assert old_counts==oracle.counts() and old_step==method.committed_steps
    branch=deepcopy(method);branch.handoff_budget=None
    after=run_population(branch,deepcopy(oracle),prefix.final_state,80-old_step)
    reference=run_population(CoolingPopulation(cfg,schedule="ramp",rule=rule,seed=7),
                             PopulationOracle(objective,total_budget=3000),initial,80)
    for a,b in zip(prefix.rows+after.rows[1:],reference.rows):
        assert a["positions"]==b["positions"] and a["velocities"]==b["velocities"]
    assert after.counts['total_oracle_calls']==reference.counts['total_oracle_calls']+4
    assert branch.committed_steps==80 and method.committed_steps==old_step
    assert branch.pair_tests==reference.counts['pair_tests']


def test_fork_disable_contacts_preserves_state_rng_and_independent_copy():
    obj,_,_=make_objective("quadratic");initial=initial_population("quadratic",obj,691,rho_max=.24)
    method=HandoffPopulation(PopulationConfig(rho=.24),schedule="ramp",rule="random",seed=17)
    branch=deepcopy(method);branch.config=replace(branch.config,collisions=False)
    assert method.config.collisions and not branch.config.collisions
    assert branch.rng.bit_generator.state==method.rng.bit_generator.state
    branch.rng.normal();assert branch.rng.bit_generator.state!=method.rng.bit_generator.state
    assert all(p.kinetic>=0 for p in initial.particles)


def test_current_selection_and_export_energy_momentum_no_reset():
    obj,_,_=make_objective("double_well");initial=initial_population("double_well",obj,690,rho_max=.24)
    values=[obj.value(p.x) for p in initial.particles];interface=partition_state(initial,values)
    i=interface['selected_index'];assert i==int(np.argmin(values))
    assert interface['selected_state']['position']==np.r_[initial.particles[i].x,initial.particles[i].z].tolist()
    assert interface['selected_state']['velocity']==initial.particles[i].velocity.tolist()
    assert abs(interface['energy_partition_residual'])<1e-12
    assert interface['momentum_partition_residual']<1e-12


@pytest.mark.parametrize('collisions',[False,True])
def test_resume_independent_overlap_allowed_but_true_contact_still_enforced(collisions):
    objective=Quadratic(np.eye(2))
    state=PopulationState(tuple(State([0.,0.],[.01*i,0.],1.,0.) for i in range(4)))
    method=HandoffPopulation(PopulationConfig(rho=.24,collisions=collisions),schedule='ramp')
    method.resuming=True
    oracle=PopulationOracle(objective,total_budget=100)
    if collisions:
        with pytest.raises(ContactResolutionError):run_population(method,oracle,state,1)
    else:
        result=run_population(method,oracle,state,1)
        assert result.stop_reason=='completed'
        assert result.rows[0]['positions']==[np.r_[p.x,p.z].tolist() for p in state.particles]


@pytest.mark.parametrize("budget",[0,1,2,5])
def test_gd_pays_gradient_and_value_before_commit(budget):
    oracle=PopulationOracle(Quadratic(np.eye(2)),total_budget=budget)
    run=gd_refine(oracle,[1.,2.],2.5,eta=.02)
    assert run.counts['total_oracle_calls']==budget
    assert len(run.rows)-1==budget//2
    assert len(run.uncommitted)==budget%2
    expected=np.array([1.,2.])*(.98**(budget//2))
    assert np.allclose(run.rows[-1]['x'],expected)
    assert run.rows[-1]['total_oracle_calls']==2*(len(run.rows)-1)
