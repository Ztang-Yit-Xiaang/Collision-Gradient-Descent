from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from collision_optim import Affine,State
from collision_optim.objectives import EvaluationBudgetExceeded
from collision_optim.population import PopulationGravity,PopulationOracle,PopulationState
from collision_optim.predictive import PredictiveConfig,QueryAccount,choose_launch,forecast
from collision_optim.predictive_runner import PhysicalLedger,apply_scheduled_impulse,run_predictive
from collision_optim.strategic import StrategicConfig


def initial(z=.14):
    return PopulationState((State([0.,0.],[0.,0.],z,-.03),))


def quick_config(**kwargs):
    strategy=replace(StrategicConfig(),window=.05,patience=1,max_shots=1)
    return PredictiveConfig(strategy=strategy,horizon=.2,max_decisions=1,**kwargs)


def test_all_virtual_queries_are_paid_even_for_rejected_or_interrupted_plans():
    oracle=PopulationOracle(Affine([0.,0.]),total_budget=1000)
    account=QueryAccount(oracle);state=initial();cfg=quick_config(session_queries=2)
    before=state.particles[0]
    decision=choose_launch(account,state,0.,cfg,np.random.default_rng(0),.4)
    assert decision.log["reason"]=="planning_limit" and decision.proposal is None
    assert account.planning_used==decision.log["planning_calls"]==2
    assert sum(oracle.counts().values())==2
    np.testing.assert_array_equal(state.particles[0].x,before.x)
    np.testing.assert_array_equal(state.particles[0].velocity,before.velocity)


def test_query_attribution_counts_failed_objective_calls_and_parent_budget():
    oracle=PopulationOracle(Affine([0.,0.]),total_budget=1)
    account=QueryAccount(oracle);view=account.view("planning",10)
    view.value(np.zeros(2))
    with pytest.raises(EvaluationBudgetExceeded):view.value(np.zeros(2))
    assert account.planning_used==1 and account.counts()["dynamics_calls"]==0


def test_selects_lowest_beneficial_energy_and_both_directions_in_that_tier():
    cfg=quick_config();account=QueryAccount(PopulationOracle(Affine([.4,0.])))
    decision=choose_launch(account,initial(.10),0.,cfg,np.random.default_rng(2),.4)
    assert decision.log["reason"]=="accepted"
    assert decision.proposal.energy==pytest.approx(.1)
    assert len(decision.log["candidates"])==2
    assert decision.log["predicted_gain"]>cfg.epsilon_gain
    assert decision.log["candidates"][decision.log["selected"]]["first_contact"]["time"]==pytest.approx(.025)


def test_forecast_allows_initial_uphill_and_scores_only_original_target():
    values=iter([[1.2,-100.],[1.4,-100.],[.7,-100.]])
    class Method:
        config=SimpleNamespace(physics=SimpleNamespace(h=.05))
        def step(self,oracle,state):return SimpleNamespace(state=state,values=next(values),events=[])
    result=forecast(Method(),None,initial(),3,1.)
    assert result["best_value"]==.7
    assert [r["best_value"] for r in result["trace"]]==[1.,1.,1.,.7]


def test_historical_best_is_not_replaced_by_a_worse_virtual_branch():
    cfg=quick_config();account=QueryAccount(PopulationOracle(Affine([.4,0.])))
    decision=choose_launch(account,initial(),-10.,cfg,np.random.default_rng(2),.4)
    assert decision.proposal is None and decision.log["reason"]=="no_benefit"
    assert all(c["score"] is None or c["score"]==-10. for c in decision.log["candidates"])


def test_rejected_planning_changes_cost_but_not_physical_state_or_supply():
    cfg=quick_config(session_queries=2)
    runs=[run_predictive(PopulationOracle(Affine([0.,0.])),initial(),10,mode=m,config=cfg) for m in ["continue","predictive_ball"]]
    assert all(r.stop_reason=="completed" for r in runs)
    np.testing.assert_allclose(runs[0].final_state.particles[0].velocity,runs[1].final_state.particles[0].velocity,atol=1e-14)
    np.testing.assert_array_equal(runs[0].final_state.particles[0].x,runs[1].final_state.particles[0].x)
    assert runs[1].counts["shots"]==0 and runs[1].rows[-1]["cumulative_source_energy"]==0
    assert runs[1].counts["total_oracle_calls"]==runs[0].counts["total_oracle_calls"]+2


def test_first_impulse_matches_contact_time_and_target_motion_without_later_contacts():
    cfg=quick_config();runs=[]
    for mode in ["predictive_ball","predictive_impulse"]:
        runs.append(run_predictive(PopulationOracle(Affine([.4,0.])),initial(),6,mode=mode,config=cfg,seed=2))
    assert all(r.stop_reason=="completed" and r.counts["shots"]==1 for r in runs)
    ball,kick=runs;shot=ball.interventions[0];pulse=kick.interventions[0]
    assert shot["time"]==pulse["time"]
    assert shot["planned_contact_observed"]
    assert pulse["actual_impulse_time"]==pytest.approx(shot["planned_contact_time"])
    contact=next(e for e in ball.events if e["kind"]=="pair")
    np.testing.assert_allclose(np.array(contact["after_i"])-contact["before_i"],pulse["actual_impulse"],atol=1e-12)
    left=[r for r in ball.rows if r["record"]=="observation"]
    right=[r for r in kick.rows if r["record"]=="observation"]
    np.testing.assert_allclose([r["target_x"] for r in left],[r["target_x"] for r in right],atol=1e-12)
    np.testing.assert_allclose(ball.final_state.particles[0].velocity,kick.final_state.particles[0].velocity,atol=1e-12)
    for run in runs:
        assert max(abs(r["energy_balance_error"]) for r in run.rows)<1e-11
        assert max(abs(r["target_energy_balance_error"]) for r in run.rows)<1e-11
        assert max(r["momentum_balance_error"] for r in run.rows)<1e-11
        assert run.counts["planning_calls"]+run.counts["dynamics_calls"]==run.counts["total_oracle_calls"]


def test_cost_checkpoint_during_planning_cannot_score_or_commit_a_virtual_result():
    cfg=quick_config()
    run=run_predictive(PopulationOracle(Affine([.4,0.])),initial(),6,config=cfg,seed=2,cost_checkpoints=(20,))
    row=next(r for r in run.rows if r["record"]=="cost_checkpoint")
    assert row["total_oracle_calls"]==20 and row["planning_calls"]>0
    assert row["time"]==pytest.approx(.05)
    assert row["best_value"]==0 and row["shots"]==0
    assert len(row["positions"])==1
    assert run.best_value<0


def test_committed_midstep_impulse_is_retained_if_remaining_drift_cannot_be_paid():
    cfg=quick_config();oracle=PopulationOracle(Affine([0.,0.]),total_budget=4)
    account=QueryAccount(oracle);ledger=PhysicalLedger(account,initial(),cfg,None,None)
    proposal=SimpleNamespace(contact_delay=.025)
    event,late,_=apply_scheduled_impulse(ledger,proposal,np.array([.2,0.,0.]))
    assert event["time"]==pytest.approx(.025)
    with pytest.raises(EvaluationBudgetExceeded):ledger.advance(late)
    assert ledger.time==pytest.approx(.025) and ledger.shots==1
    assert ledger.source==pytest.approx(.02)
    assert ledger.state.particles[0].v[0]==pytest.approx(.2)
    assert abs(ledger.record("stopped")["energy_balance_error"])<1e-12


def test_invalid_prediction_configuration():
    for kwargs in [{"horizon":.023},{"energies":(.2,.1)},{"epsilon_gain":-1},{"session_queries":0}]:
        with pytest.raises(ValueError):PredictiveConfig(**kwargs)
