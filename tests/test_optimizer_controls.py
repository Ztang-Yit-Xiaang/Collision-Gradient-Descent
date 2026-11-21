from dataclasses import replace
import numpy as np
import pytest

from collision_optim import Quadratic
from collision_optim.population import PopulationOracle
from collision_optim.predictive import QueryAccount
from collision_optim.optimizer_controls import (
    OptimizerConfig,OptimizerState,ControlConfig,optimizer_step,inject_optimizer,
    preview_optimizer,run_optimizer,
)


@pytest.mark.parametrize("kind",["momentum","adam","adamw"])
@pytest.mark.parametrize("decay",[0.,.07])
def test_canonical_updates_match_pytorch_with_and_without_decay(kind,decay):
    torch=pytest.importorskip("torch")
    initial=np.array([.2,-.7,1.1]);p=torch.tensor(initial,dtype=torch.float64,requires_grad=True)
    cfg=OptimizerConfig(kind=kind,lr=.03,weight_decay=decay)
    options={"lr":cfg.lr,"weight_decay":decay,"foreach":False}
    if kind=="momentum":opt=torch.optim.SGD([p],momentum=cfg.beta1,**options)
    else:opt=getattr(torch.optim,"AdamW" if kind=="adamw" else "Adam")([p],betas=(cfg.beta1,cfg.beta2),eps=cfg.eps,**options)
    state=OptimizerState.initialize(initial)
    for step in range(30):
        gradient=np.sin(np.arange(3)+step*.3)+.1
        p.grad=torch.tensor(gradient,dtype=torch.float64);opt.step()
        state=optimizer_step(state,gradient,cfg)
        np.testing.assert_allclose(state.x,p.detach().numpy(),rtol=1e-12,atol=1e-13)


def test_zero_decay_adam_and_adamw_have_identical_updates():
    a=b=OptimizerState.initialize([.3,-.2])
    for _ in range(40):
        a=optimizer_step(a,a.x,OptimizerConfig(kind="adam"))
        b=optimizer_step(b,b.x,OptimizerConfig(kind="adamw"))
        np.testing.assert_array_equal(a.x,b.x)


def test_external_velocity_equals_native_momentum_impulse_when_decays_match():
    cfg=OptimizerConfig();state=OptimizerState.initialize([.3,-.2])
    for _ in range(8):state=optimizer_step(state,state.x,cfg)
    changed,impulse=inject_optimizer(state,.02,[1.,-1.])
    native=OptimizerState(state.x,state.first-impulse/cfg.lr,state.second,np.zeros(2),state.base_step+impulse,state.step)
    for _ in range(20):
        changed=optimizer_step(changed,changed.x,cfg);native=optimizer_step(native,native.x,cfg)
        np.testing.assert_allclose(changed.x,native.x,atol=1e-13)
        np.testing.assert_allclose(changed.velocity,native.velocity,atol=1e-13)


def test_injection_preserves_adam_moments_and_has_declared_kinetic_increment():
    cfg=OptimizerConfig(kind="adam");state=OptimizerState.initialize([.3,-.2])
    state=optimizer_step(state,state.x,cfg)
    changed,_=inject_optimizer(state,.02,[1.,1.])
    assert changed.step==state.step
    np.testing.assert_array_equal(changed.first,state.first)
    np.testing.assert_array_equal(changed.second,state.second)
    assert .5*(changed.velocity@changed.velocity-state.velocity@state.velocity)==pytest.approx(.02)


def test_cancelled_preview_charges_queries_but_does_not_change_state():
    cfg=OptimizerConfig();state=OptimizerState.initialize([.3,-.2])
    account=QueryAccount(PopulationOracle(Quadratic(),total_budget=1000))
    changed,impulse,log=preview_optimizer(account,state,cfg,ControlConfig(planning_budget=2),np.random.default_rng(1),.065,[.01,.02],.04)
    assert changed is impulse is None and log["reason"]=="planning_limit"
    assert log["calls"]==account.planning_used==2
    assert state.step==0 and np.all(state.first==0)


def test_rejected_predictive_control_only_changes_cost_at_convex_optimum():
    cfg=OptimizerConfig();control=ControlConfig(window=2,patience=1,horizon=3,reserve=2)
    result=run_optimizer(PopulationOracle(Quadratic(),total_budget=100),[0.,0.],cfg,mode="predictive",control=control,max_steps=20)
    assert result.interventions==[] and result.best_value==0
    assert result.counts["planning_calls"]>0
    assert result.counts["planning_calls"]+result.counts["dynamics_calls"]==result.counts["total_oracle_calls"]
    assert all(r["best_value"]==0 for r in result.rows)
