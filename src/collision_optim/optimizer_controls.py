"""Canonical first-order optimizers plus explicitly separate inertial control."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .core import vector
from .objectives import EvaluationBudgetExceeded
from .predictive import PlanningLimit,QueryAccount


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str="momentum"
    lr: float=.01
    beta1: float=.9
    beta2: float=.999
    eps: float=1e-8
    weight_decay: float=0.
    control_decay: float=.9

    def __post_init__(self):
        if self.kind not in {"momentum","adam","adamw"}:raise ValueError("Unknown optimizer.")
        if not np.all(np.isfinite([self.lr,self.beta1,self.beta2,self.eps,self.weight_decay,self.control_decay])):
            raise ValueError("Optimizer scales must be finite.")
        if self.lr<=0 or self.eps<=0 or self.weight_decay<0 or not all(0<=v<1 for v in [self.beta1,self.beta2,self.control_decay]):
            raise ValueError("Invalid optimizer scales.")


@dataclass(frozen=True)
class OptimizerState:
    x: np.ndarray
    first: np.ndarray
    second: np.ndarray
    external: np.ndarray
    base_step: np.ndarray
    step: int=0

    def __post_init__(self):
        arrays=[vector(getattr(self,k)) for k in ["x","first","second","external","base_step"]]
        if any(a.shape!=arrays[0].shape for a in arrays) or not isinstance(self.step,int) or self.step<0:
            raise ValueError("Invalid optimizer state.")
        for k,a in zip(["x","first","second","external","base_step"],arrays):
            a.setflags(write=False);object.__setattr__(self,k,a)

    @classmethod
    def initialize(cls,x):
        x=vector(x);z=np.zeros_like(x)
        return cls(x,z,z,z,z)

    @property
    def velocity(self):return self.base_step+self.external


def optimizer_step(state,gradient,config):
    """SGD momentum buffer / bias-corrected Adam, then external displacement."""
    g=vector(gradient)
    if g.shape!=state.x.shape:raise ValueError("Gradient shape mismatch.")
    if config.kind!="adamw":g=g+config.weight_decay*state.x
    t=state.step+1
    if config.kind=="momentum":
        first=config.beta1*state.first+g
        second=state.second
        base=-config.lr*first
    else:
        first=config.beta1*state.first+(1-config.beta1)*g
        second=config.beta2*state.second+(1-config.beta2)*g*g
        corrected=first/(1-config.beta1**t)
        denominator=np.sqrt(second/(1-config.beta2**t))+config.eps
        base=-config.lr*corrected/denominator
        if config.kind=="adamw":base=base-config.lr*config.weight_decay*state.x
    external=config.control_decay*state.external
    return OptimizerState(state.x+base+external,first,second,external,base,t)


def inject_optimizer(state,energy,direction):
    """Add external velocity without editing Adam moments or step counters.

    The model-velocity kinetic increment is an intervention diagnostic. It is
    not a claim of Hamiltonian dynamics or energy conservation for Adam.
    """
    d=vector(direction);d/=np.linalg.norm(d)
    projection=float(state.velocity @ d)
    amplitude=-projection+np.sqrt(projection**2+2*energy)
    impulse=amplitude*d
    changed=OptimizerState(state.x,state.first,state.second,state.external+impulse,state.base_step,state.step)
    return changed,impulse


@dataclass(frozen=True)
class ControlConfig:
    window: int=10
    patience: int=2
    epsilon_progress: float=1e-4
    epsilon_move: float=1e-3
    horizon: int=20
    radii: tuple[float,...]=(.05,.15)
    cooldown: int=40
    max_shots: int=2
    max_decisions: int=2
    planning_budget: int=1000
    reserve: int=100


def _rollout(oracle,state,config,steps,best):
    trace=[]
    for _ in range(steps):
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            changed=optimizer_step(state,oracle.gradient(state.x),config)
            value=oracle.value(changed.x)
        state=changed;best=min(best,value);trace.append(float(best))
    return best,trace


def preview_optimizer(account,state,config,control,rng,best,energies,remaining):
    available=account.oracle.total_budget
    limit=control.planning_budget-account.planning_used
    if available is not None:limit=min(limit,max(0,available-sum(account.oracle.counts().values())-control.reserve))
    log={"reason":"no_benefit","candidates":[],"selected":None,"baseline":None,"calls":0}
    if limit<=0:log["reason"]="planning_limit";return None,None,log
    oracle=account.view("planning",limit)
    try:
        baseline,trace=_rollout(oracle,state,config,control.horizon,best)
        log["baseline"]={"best":baseline,"trace":trace}
        d=rng.normal(size=state.x.size);d/=np.linalg.norm(d)
        for energy in energies:
            if energy>remaining+1e-12:continue
            valid=[]
            for direction in [d,-d]:
                candidate={"energy":energy,"direction":direction.tolist(),"status":"incomplete"}
                log["candidates"].append(candidate)
                try:
                    changed,impulse=inject_optimizer(state,energy,direction)
                    score,trace=_rollout(oracle,changed,config,control.horizon,best)
                    gain=(baseline-score)/(1+abs(best))
                    candidate.update({"score":score,"gain":gain,"trace":trace,"status":"completed"})
                    if gain>control.epsilon_progress:valid.append((gain,len(log["candidates"])-1,changed,impulse))
                except (FloatingPointError,ValueError) as exc:
                    candidate.update({"status":"numerical_error","error":str(exc)})
            if valid:
                gain,index,changed,impulse=max(valid,key=lambda z:(z[0],-z[1]))
                log.update({"reason":"accepted","selected":index,"predicted_gain":gain,"energy":energy})
                return changed,impulse,log
        return None,None,log
    except PlanningLimit:
        log["reason"]="planning_limit";return None,None,log
    except (FloatingPointError,ValueError) as exc:
        log.update({"reason":"baseline_error","error":str(exc)});return None,None,log
    finally:log["calls"]=oracle.used


@dataclass
class OptimizerRun:
    rows: list[dict]
    decisions: list[dict]
    interventions: list[dict]
    state: OptimizerState
    best_value: float
    stop_reason: str
    wall_seconds: float
    counts: dict


def run_optimizer(oracle,x0,config,*,mode="none",control=None,seed=0,max_steps=10000,checkpoints=()):
    if mode not in {"none","random","predictive"}:raise ValueError("Unknown control.")
    control=control or ControlConfig();started=perf_counter()
    account=QueryAccount(oracle,checkpoints=checkpoints)
    state=OptimizerState.initialize(x0)
    value=account.view("dynamics").value(state.x);best=value
    rng=np.random.default_rng(523906+seed)
    energies=[.5*(r*(1+np.linalg.norm(state.x)))**2 for r in control.radii]
    budget=control.max_shots*energies[-1];supplied=0.
    rows=[];decisions=[];interventions=[];history=[]
    shots=attempts=streak=0;allowed=0;failures=0;feedback=None;reason="steps"

    def record(kind):
        row={"step":state.step,"record":kind,"value":float(value),"best_value":float(best),
             "x":state.x.tolist(),"step_norm":float(np.linalg.norm(state.velocity)),
             "external_norm":float(np.linalg.norm(state.external)),"shots":shots,"decisions":attempts,
             "supplied_model_kinetic":supplied,**account.counts()}
        rows.append(row);return row

    account.on_checkpoint=lambda total:record("cost_checkpoint")
    record("observation");history.append((0,best,state.x.copy(),0.))
    for _ in range(max_steps):
        t=state.step
        if feedback is not None and t>=feedback["due"]:
            gain=(feedback["reference"]-best)/(1+abs(feedback["best"]))
            feedback["action"].update({"realized_gain":gain,"realized_best":best})
            failures=failures+1 if gain<=control.epsilon_progress else 0
            if failures:allowed=max(allowed,t+control.cooldown*2**failures)
            feedback=None
        try:
            if t and t%control.window==0:
                window=[r for r in history if r[0]>=t-control.window]
                progress=(window[0][1]-best)/(1+abs(window[0][1]))
                movement=float(np.mean([r[3] for r in window[1:]]))
                quiet=progress<=control.epsilon_progress and movement<=control.epsilon_move
                streak=streak+1 if quiet else 0
                if mode!="none" and streak>=control.patience and t>=allowed and feedback is None and shots<control.max_shots and max_steps-t>=control.horizon:
                    changed=impulse=log=None;energy=None
                    if mode=="random" and supplied+energies[-1]<=budget+1e-12:
                        energy=energies[-1];changed,impulse=inject_optimizer(state,energy,rng.normal(size=state.x.size))
                    elif mode=="predictive" and attempts<control.max_decisions:
                        attempts+=1
                        changed,impulse,log=preview_optimizer(account,state,config,control,rng,best,energies,budget-supplied)
                        log["step"]=t;decisions.append(log);record("planning")
                        if changed is not None:energy=log["energy"]
                        else:
                            failures+=1;allowed=t+control.cooldown*2**min(failures,4)
                    if changed is not None:
                        before=state.velocity;after=changed.velocity
                        actual=float(.5*(after @ after-before @ before))
                        if abs(actual-energy)>1e-10:raise FloatingPointError("External kinetic increment mismatch.")
                        action={"step":t,"energy":actual,"impulse":impulse.tolist(),"kind":mode,
                                "base_kind":config.kind,"predicted_gain":None if log is None else log["predicted_gain"]}
                        state=changed;shots+=1;supplied+=actual;interventions.append(action);record("intervention")
                        feedback={"due":t+control.horizon,"reference":best if log is None else log["baseline"]["best"],"best":best,"action":action}
                        allowed=t+control.cooldown;streak=0
            old=state.x
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                candidate=optimizer_step(state,account.view("dynamics").gradient(state.x),config)
                candidate_value=account.view("dynamics").value(candidate.x)
            state=candidate;value=candidate_value;best=min(best,value)
            relative=float(np.linalg.norm(state.x-old)/(1+np.linalg.norm(old)))
            history.append((state.step,best,state.x.copy(),relative));record("observation")
        except EvaluationBudgetExceeded as exc:
            reason=f"budget: {exc}";record("stopped");break
        except (FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
            reason=f"numerical_error: {exc}";record("stopped");break
    return OptimizerRun(rows,decisions,interventions,state,best,reason,perf_counter()-started,account.counts())
