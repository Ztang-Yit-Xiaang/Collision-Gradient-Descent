from dataclasses import replace
import numpy as np
import pytest

from collision_optim import Affine,GravityConfig,State,TiltedDoubleWell
from collision_optim.population import PopulationConfig,PopulationGravity,PopulationOracle,PopulationState
from collision_optim import strategic
from collision_optim.strategic import StrategicConfig,StagnationMonitor,kinetic_kick,plan_launch,run_strategic


def flat_initial():
    return PopulationState((State([0.,0.],[0.,0.],.14,-.03),))


def history_until(end,*,moving=False,kinetic=0.,improving=False):
    rows=[]
    for t in np.arange(0,end+.01,.5):
        rows.append({"time":float(t),"best_value":1-.1*t if improving else 1.,
                     "horizontal_kinetic":kinetic,"x":np.array([np.sin(np.pi*t) if moving else 0.])})
    return rows


def test_stagnation_requires_two_quiet_windows_and_respects_cooldown():
    cfg=StrategicConfig();monitor=StagnationMonitor(cfg)
    assert monitor.evaluate(history_until(1),1)["eligible"] is False
    assert monitor.evaluate(history_until(2),2)["eligible"] is True
    monitor.action(2)
    assert monitor.evaluate(history_until(3),3) is None
    assert monitor.evaluate(history_until(6),6)["eligible"] is False
    assert monitor.evaluate(history_until(7),7)["eligible"] is True


@pytest.mark.parametrize("kwargs",[{"moving":True},{"kinetic":.1},{"improving":True}])
def test_motion_progress_and_energy_prevent_false_low_energy_stagnation(kwargs):
    monitor=StagnationMonitor(StrategicConfig())
    row=monitor.evaluate(history_until(1,**kwargs),1)
    assert row["quiet"] is False


def test_launch_budget_includes_potential_and_aim_accounts_for_immediate_damping():
    physics=GravityConfig(h=.1,beta=.9)
    cfg=StrategicConfig(population=PopulationConfig(physics=physics,rho=.12))
    method=PopulationGravity(cfg.population)
    oracle=PopulationOracle(Affine([0.,0.]))
    initial=PopulationState((State([0.,0.],[.01,-.01],.10,.05),))
    proposal=plan_launch(method,oracle,initial,cfg,np.random.default_rng(7))
    assert proposal is not None
    assert proposal.helper.kinetic+physics.gravity*proposal.helper.z==pytest.approx(cfg.shot_energy,abs=1e-13)
    assert proposal.helper.kinetic<cfg.shot_energy
    assert np.linalg.norm(proposal.helper.x-initial.particles[0].x)>cfg.population.rho
    launched=PopulationState(initial.particles+(proposal.helper,))
    result=method.step(oracle,launched)
    pair=next(e for e in result.events if e["kind"]=="pair")
    assert pair["time"]==pytest.approx(cfg.launch_time,abs=1e-12)
    assert abs(pair["energy_error"])<1e-12 and pair["momentum_error"]<1e-12


def test_unaffordable_launch_is_rejected_without_inventing_energy():
    cfg=StrategicConfig()
    initial=PopulationState((State([0.,0.],[0.,0.],1.,0.),))
    method=PopulationGravity(cfg.population)
    assert plan_launch(method,PopulationOracle(Affine([0.,0.])),initial,cfg,np.random.default_rng(0)) is None


def test_direct_kick_has_exact_supplied_energy_and_retains_vertical_motion():
    initial=PopulationState((State([0.,0.],[.2,-.3],.14,.4),))
    changed=kinetic_kick(initial,.2,[-1.,1.])
    assert changed.kinetic-initial.kinetic==pytest.approx(.2,abs=1e-13)
    assert changed.particles[0].vz==initial.particles[0].vz
    np.testing.assert_array_equal(changed.particles[0].x,initial.particles[0].x)


@pytest.mark.parametrize("mode",["random_kick","geometry_kick","ball"])
def test_open_system_energy_and_momentum_include_injection(mode):
    cfg=StrategicConfig()
    result=run_strategic(PopulationOracle(Affine([0.,0.])),flat_initial(),120,mode=mode,config=cfg,seed=0)
    assert result.stop_reason=="completed",result.stop_reason
    assert result.counts["shots"]>=1
    assert result.rows[-1]["cumulative_source_energy"]<=cfg.energy_budget+1e-12
    assert max(abs(r["energy_balance_error"]) for r in result.rows)<1e-10
    assert max(abs(r["target_energy_balance_error"]) for r in result.rows)<1e-10
    assert max(r["momentum_balance_error"] for r in result.rows)<1e-10
    shots=[r for r in result.interventions if r["kind"]==mode]
    assert all(r["energy_input"]==pytest.approx(cfg.shot_energy) for r in shots)
    if mode=="ball":
        assert all(r["planned_contact_observed"] for r in shots)
        assert len(result.final_state.particles)==1+len(shots)
    else:
        assert len(result.final_state.particles)==1


def test_first_feasible_direction_and_time_are_matched_before_trajectories_diverge():
    results=[]
    for mode in ["geometry_kick","ball"]:
        results.append(run_strategic(PopulationOracle(Affine([0.,0.])),flat_initial(),70,mode=mode,seed=9))
    first=[next(s for s in r.interventions if s["kind"] in {"ball","geometry_kick"}) for r in results]
    assert first[0]["time"]==first[1]["time"]
    np.testing.assert_allclose(first[0]["direction"],first[1]["direction"],atol=1e-14)
    assert first[0]["energy_input"]==pytest.approx(first[1]["energy_input"])


def test_committed_pulse_survives_a_later_drift_budget_failure():
    initial=flat_initial()
    base=run_strategic(PopulationOracle(Affine([0.,0.])),initial,40,mode="continue")
    cap=base.counts["total_oracle_calls"]
    result=run_strategic(PopulationOracle(Affine([0.,0.]),total_budget=cap),initial,41,mode="random_kick")
    assert result.stop_reason.startswith("budget:")
    assert result.counts["shots"]==1
    assert result.rows[-1]["time"]==pytest.approx(2.)
    assert result.rows[-1]["cumulative_source_energy"]==pytest.approx(.2)
    assert np.linalg.norm(result.final_state.particles[0].v)>0
    assert abs(result.rows[-1]["energy_balance_error"])<1e-10
    assert result.counts["total_oracle_calls"]==cap


def test_only_original_target_is_scored_even_when_helper_has_better_loss(monkeypatch):
    objective=TiltedDoubleWell()
    roots=sorted(np.roots([1.,0.,-1.,objective.tilt]).real)
    x=np.array([roots[-1],0.])
    initial=PopulationState((State(x,[0.,0.],objective.value(x)+.14,-.03),))
    helper=State([roots[0],0.],[0.,0.],.4,0.)
    proposal=strategic.LaunchProposal(helper,np.array([1.,0.]),.025,.2,1)
    monkeypatch.setattr(strategic,"plan_launch",lambda *args:proposal)
    cfg=replace(StrategicConfig(),max_shots=1)
    result=run_strategic(PopulationOracle(objective),initial,120,mode="ball",config=cfg)
    assert result.stop_reason=="completed"
    assert len(result.final_state.particles)==2
    assert objective.value(result.final_state.particles[1].x)<0
    assert result.best_value>0


def test_shot_limit_and_insufficient_total_supply_are_enforced():
    cfg=replace(StrategicConfig(),energy_budget=.1)
    result=run_strategic(PopulationOracle(Affine([0.,0.])),flat_initial(),100,mode="ball",config=cfg)
    assert result.counts["shots"]==0
    assert result.rows[-1]["cumulative_source_energy"]==0


def test_invalid_launch_configuration_is_rejected():
    for kwargs in [{"window":0},{"max_shots":0},{"launch_time":.1},{"epsilon_k":-1}]:
        with pytest.raises(ValueError): StrategicConfig(**kwargs)
