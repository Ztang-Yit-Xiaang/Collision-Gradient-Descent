"""Finite-horizon launch selection; virtual branches share the paid oracle."""

from __future__ import annotations

from dataclasses import dataclass,field,replace
import numpy as np

from .core import State
from .hybrid import ContactResolutionError
from .objectives import EvaluationBudgetExceeded
from .population import PopulationGravity,PopulationState
from .strategic import StrategicConfig,plan_launch


class PlanningLimit(EvaluationBudgetExceeded):
    """A local planning limit, not exhaustion of the physical run's oracle."""


@dataclass(frozen=True)
class PredictiveConfig:
    strategy: StrategicConfig=field(default_factory=StrategicConfig)
    horizon: float=8.
    energies: tuple[float,...]=(.10,.20,.30)
    epsilon_gain: float=1e-4
    session_queries: int=10000
    planning_queries: int=18000
    max_decisions: int=2

    def __post_init__(self):
        h=self.strategy.population.physics.h
        if not np.isfinite(self.horizon) or self.horizon<=0 or abs(self.horizon/h-round(self.horizon/h))>1e-8:
            raise ValueError("The forecast horizon must be a positive multiple of h.")
        if not self.energies or any(not np.isfinite(e) or e<=0 for e in self.energies) or tuple(sorted(set(self.energies)))!=tuple(self.energies):
            raise ValueError("Supply levels must be finite, positive and strictly increasing.")
        if not np.isfinite(self.epsilon_gain) or self.epsilon_gain<0:
            raise ValueError("The gain threshold must be finite and nonnegative.")
        if any(not isinstance(n,int) or n<1 for n in [self.session_queries,self.planning_queries,self.max_decisions]):
            raise ValueError("Planning counts must be positive integers.")
        if not self.strategy.population.collisions:
            raise ValueError("Predictive launch selection requires particle collisions.")


class QueryAccount:
    """Attribute each real oracle call, including failed speculative branches."""
    def __init__(self,oracle,*,checkpoints=(),on_checkpoint=None):
        self.oracle=oracle
        self.phases={p:{k:0 for k in oracle.counts()} for p in ["dynamics","planning"]}
        self.checkpoints=set(checkpoints)
        self.on_checkpoint=on_checkpoint

    def view(self,phase,limit=None):
        if phase not in self.phases:raise ValueError("Unknown query phase.")
        return QueryView(self,phase,limit)

    @property
    def planning_used(self):
        return sum(self.phases["planning"].values())

    def counts(self):
        counts=self.oracle.counts()
        return {**counts,"total_oracle_calls":sum(counts.values()),
                "planning_calls":self.planning_used,
                "dynamics_calls":sum(self.phases["dynamics"].values()),
                **{f"{phase}_{k}":v for phase,group in self.phases.items() for k,v in group.items()}}


class QueryView:
    def __init__(self,account,phase,limit):
        self.account=account;self.phase=phase;self.limit=limit;self.used=0

    def _call(self,name,*args):
        if self.limit is not None and self.used>=self.limit:
            raise PlanningLimit(f"planning allowance exhausted ({self.used}/{self.limit})")
        parent=self.account.oracle;before=parent.counts()
        try:
            return getattr(parent,name)(*args)
        finally:
            after=parent.counts()
            spent=sum(after.values())-sum(before.values());self.used+=spent
            for k in before:self.account.phases[self.phase][k]+=after[k]-before[k]
            total=sum(after.values())
            if spent and total in self.account.checkpoints and self.account.on_checkpoint is not None:
                self.account.on_checkpoint(total)

    def value(self,x):return self._call("value",x)
    def gradient(self,x):return self._call("gradient",x)
    def path_coefficients(self,x,displacement):return self._call("path_coefficients",x,displacement)


class FixedDirection:
    """Give the existing physical feasibility planner exactly one direction."""
    def __init__(self,direction):self.direction=np.asarray(direction)
    def normal(self,*,size):
        if size!=self.direction.size:raise ValueError("Direction dimension mismatch.")
        return self.direction.copy()


def forecast(method,oracle,initial,steps,best_value,*,first_helper=None,delay=None):
    """Only completed observation steps affect the virtual score."""
    state=initial;best=float(best_value)
    trace=[{"time":0.,"best_value":best,"target_x":state.particles[0].x.tolist()}]
    contact=None
    for i in range(steps):
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            outcome=method.step(oracle,state)
        state=outcome.state;best=min(best,outcome.values[0])
        for event in outcome.events:
            if first_helper is not None and event["kind"]=="pair" and {event["i"],event["j"]}=={0,first_helper}:
                when=i*method.config.physics.h+event["time"]
                if abs(when-delay)<100*method.config.physics.time_tolerance:
                    suffix="i" if event["i"]==0 else "j"
                    before=np.array(event[f"before_{suffix}"]);after=np.array(event[f"after_{suffix}"])
                    contact={"time":when,"impulse":(after-before).tolist(),
                             "work":float(.5*(after @ after-before @ before)),
                             "before":before.tolist(),"after":after.tolist(),
                             "point":event[f"point_{suffix}"]}
        trace.append({"time":(i+1)*method.config.physics.h,"best_value":best,"target_x":state.particles[0].x.tolist()})
    return {"best_value":best,"trace":trace,"first_contact":contact}


@dataclass
class LaunchDecision:
    log: dict
    proposal: object | None=None
    impulse: np.ndarray | None=None


def choose_launch(account,state,best_value,config,rng,remaining_energy,*,query_limit=None,on_branch=None):
    """Choose the lowest fully evaluated beneficial supply tier, or do nothing.

    This function never commits physical motion, supply, or a virtual best value.
    Its only lasting effects are query charges and advances of the planning RNG.
    """
    cfg=config.strategy;physics=cfg.population.physics
    limit=min(config.session_queries,config.planning_queries-account.planning_used)
    if query_limit is not None:limit=min(limit,query_limit)
    log={"reason":"no_benefit","baseline":None,"candidates":[],"selected":None,
         "planning_calls":0,"planning_pair_tests":0,"horizon":config.horizon}
    if limit<=0:
        log["reason"]="planning_limit";return LaunchDecision(log)
    oracle=account.view("planning",limit)
    method=PopulationGravity(cfg.population)
    steps=round(config.horizon/physics.h)
    scale=1+abs(best_value)
    try:
        start=oracle.used
        try:
            baseline=forecast(method,oracle,state,steps,best_value)
        except (ContactResolutionError,FloatingPointError,np.linalg.LinAlgError) as exc:
            log["reason"]="baseline_error";log["error"]=str(exc);return LaunchDecision(log)
        log["baseline"]={**baseline,"calls":oracle.used-start}
        if on_branch is not None:on_branch()
        direction=rng.normal(size=state.particles[0].x.size)
        direction/=np.linalg.norm(direction)
        directions=[direction,-direction]
        for energy in config.energies:
            if energy>remaining_energy+1e-12:continue
            eligible=[]
            for index,d in enumerate(directions):
                candidate={"energy":float(energy),"direction":d.tolist(),"direction_index":index,
                           "status":"infeasible","score":None,"gain":None,"trace":[]}
                log["candidates"].append(candidate);start=oracle.used
                try:
                    proposal=plan_launch(method,oracle,state,replace(cfg,shot_energy=energy,directions=1),FixedDirection(d))
                    if proposal is not None:
                        helper=len(state.particles)
                        branch=PopulationState(state.particles+(proposal.helper,))
                        result=forecast(method,oracle,branch,steps,best_value,first_helper=helper,delay=proposal.contact_delay)
                        contact=result["first_contact"]
                        if contact is None:raise ContactResolutionError("Forecast did not observe its planned first contact.")
                        gain=(baseline["best_value"]-result["best_value"])/scale
                        candidate.update({"status":"completed","score":result["best_value"],"gain":float(gain),
                                          "trace":result["trace"],"first_contact":contact})
                        if gain>config.epsilon_gain and max(0.,contact["work"])<=remaining_energy+1e-12:
                            eligible.append((gain,len(log["candidates"])-1,proposal,np.array(contact["impulse"])))
                except PlanningLimit:
                    candidate["status"]="planning_limit"
                    raise
                except (ContactResolutionError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    candidate.update({"status":"physical_error","error":str(exc)})
                finally:
                    candidate["calls"]=oracle.used-start
                    if on_branch is not None:on_branch()
            if eligible:
                gain,index,proposal,impulse=max(eligible,key=lambda entry:(entry[0],-entry[1]))
                log.update({"reason":"accepted","selected":index,"predicted_gain":float(gain),"selected_energy":energy})
                return LaunchDecision(log,proposal,impulse)
        return LaunchDecision(log)
    except PlanningLimit as exc:
        log.update({"reason":"planning_limit","error":str(exc)})
        return LaunchDecision(log)
    finally:
        log["planning_calls"]=oracle.used
        log["planning_pair_tests"]=method.pair_tests
