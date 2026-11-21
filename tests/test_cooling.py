from dataclasses import replace
import numpy as np
import pytest

from collision_optim.cooling import CoolingPopulation, drag_rate, trajectory_metrics, GAMMA0, GAMMA1
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.population_experiment import make_objective, initial_population
from collision_optim.scattering_population import ScatteringPopulation


def test_drag_timing_and_nominal_equal_integrals():
    ramp=np.array([drag_rate("ramp",i*.05,.05) for i in range(500)])
    assert np.all(np.diff(ramp)>0) and GAMMA0 < ramp[0] < ramp[-1] < GAMMA1
    assert abs(.05*ramp.sum()-25*drag_rate("constant",0,.05))<1e-12
    assert np.exp(-drag_rate("weak",0,.05)*.05)==.995


@pytest.mark.parametrize("schedule,time,h",[("x",0,.05),("ramp",-1,.05),("weak",0,0)])
def test_invalid_drag(schedule,time,h):
    with pytest.raises(ValueError):drag_rate(schedule,time,h)


@pytest.mark.parametrize("collisions,rule",[(False,"elastic"),(True,"elastic"),(True,"random")])
def test_weak_exact_equivalence_and_committed_counter(collisions,rule):
    objective,_,_=make_objective("quadratic"); initial=initial_population("quadratic",objective,590,rho_max=.24)
    cfg=PopulationConfig(rho=.24,collisions=collisions)
    old=run_population(ScatteringPopulation(cfg,rule=rule,seed=7),PopulationOracle(objective,total_budget=500),initial,80)
    method=CoolingPopulation(cfg,schedule="weak",rule=rule,seed=7)
    new=run_population(method,PopulationOracle(objective,total_budget=500),initial,80)
    assert old.stop_reason==new.stop_reason and old.counts==new.counts
    assert method.committed_steps==len(new.rows)-1
    for a,b in zip(old.rows,new.rows): assert all(a[k]==b[k] for k in a)
    assert old.events==new.events


@pytest.mark.parametrize("schedule",["constant","ramp"])
def test_actual_heat_and_momentum_ledger(schedule):
    objective,_,_=make_objective("quadratic");initial=initial_population("quadratic",objective,591,rho_max=.24)
    method=CoolingPopulation(PopulationConfig(rho=.24),schedule=schedule)
    result=run_population(method,PopulationOracle(objective),initial,60)
    assert result.stop_reason=="completed"
    for previous,current in zip(result.rows,result.rows[1:]):
        beta=current["drag_beta"]
        assert abs(current["damping_energy"]-(beta**2-1)*previous["kinetic"])<1e-12
        expected=(beta-1)*np.array(previous["momentum"])
        assert np.linalg.norm(expected-np.array(current["damping_impulse"]))<1e-12
        assert abs(current["energy_residual"])<1e-12


def test_hit_metrics_include_exits_and_weight_tail_intervals():
    def point(time,hit):return {"time":time,"values":[0. if hit else 2.],"positions":[[-1.,0.,3.]],"kinetic":.5,"cumulative_damping":-time}
    result=trajectory_metrics([point(0,False),point(1,True),point(2,False),point(3,True)],1.,0.)
    assert result["observed_ever_hit"] and result["observed_final_hit"]
    assert result["first_observed_hit_time"]==1 and result["hit_exit_transitions"]==1
    assert result["tail_target_occupancy"]==0 and result["removed_energy"]==3
    assert trajectory_metrics([point(0,False)],1.,0.)["tail_target_occupancy"] is None
