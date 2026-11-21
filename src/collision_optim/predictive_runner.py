"""Execute selected interventions with independent physical and query ledgers."""

from __future__ import annotations

from dataclasses import dataclass,replace
from time import perf_counter
import numpy as np

from .core import State
from .hybrid import ContactResolutionError
from .objectives import EvaluationBudgetExceeded
from .population import PopulationGravity,PopulationState
from .predictive import PredictiveConfig,QueryAccount,choose_launch
from .strategic import StagnationMonitor,StrategicRun,kinetic_kick


MODES=("continue","random_kick","predictive_impulse","predictive_ball")


class PhysicalLedger:
    def __init__(self,account,initial,config,boundary,target_value):
        self.account=account;self.config=config;self.physics=config.strategy.population.physics
        self.method=PopulationGravity(config.strategy.population)
        self.state=initial;self.time=0.;self.boundary=boundary;self.target_value=target_value
        values,_=self.method._validate(account.view("dynamics"),initial.particles,initial=True)
        self.value=self.best_value=values[0];self.best_x=initial.particles[0].x.copy()
        self.energy0=self.method.energy(initial);self.momentum0=initial.momentum
        self.target_energy0=initial.particles[0].kinetic+self.physics.gravity*initial.particles[0].z
        self.source=self.positive_supply=self.extracted=self.damping=self.impulse_error=0.
        self.source_momentum=np.zeros_like(self.momentum0);self.dynamics_momentum=np.zeros_like(self.momentum0)
        self.target_source=self.target_damping=self.target_transfer=self.target_wall=self.gravity_work=0.
        self.rows=[];self.events=[];self.shots=self.decisions=self.real_pairs=self.planning_pairs=0
        self.first_crossing=0. if boundary is not None and initial.particles[0].x[0]<boundary else None
        self.first_target=0. if self.target_hit() else None
        self.record("observation")

    def target_hit(self):
        return self.target_value is not None and self.value<=self.target_value and (self.boundary is None or self.state.particles[0].x[0]<self.boundary)

    def record(self,kind,diagnostics=None):
        p=self.state.particles[0];energy=self.method.energy(self.state)
        row={"time":float(self.time),"record":kind,"value":self.value,"best_value":self.best_value,
             "target_x":p.x.tolist(),"target_v":p.v.tolist(),"target_z":float(p.z),"target_vz":float(p.vz),
             "target_horizontal_kinetic":float(.5*(p.v @ p.v)),"particle_count":len(self.state.particles),
             "energy":energy,"cumulative_source_energy":self.source,"positive_supply":self.positive_supply,
             "extracted_energy":self.extracted,"cumulative_source_momentum":self.source_momentum.tolist(),
             "cumulative_damping":self.damping,"cumulative_impulse_error":self.impulse_error,
             "energy_balance_error":energy-self.energy0-self.source-self.damping-self.impulse_error,
             "momentum_balance_error":float(np.linalg.norm(self.state.momentum-self.momentum0-self.source_momentum-self.dynamics_momentum)),
             "target_energy":p.kinetic+self.physics.gravity*p.z,"target_source_energy":self.target_source,
             "target_pair_energy_transfer":self.target_transfer,"target_damping_energy":self.target_damping,
             "target_wall_energy_error":self.target_wall,"target_gravity_work":self.gravity_work,
             "target_energy_balance_error":p.kinetic+self.physics.gravity*p.z-self.target_energy0-self.target_source-self.target_damping-self.target_transfer-self.target_wall,
             "shots":self.shots,"launch_attempts":self.decisions,"first_crossing":self.first_crossing,
             "target_hit":self.first_target is not None,"first_target_time":self.first_target,
             "positions":[np.r_[q.x,q.z].tolist() for q in self.state.particles],
             "velocities":[q.velocity.tolist() for q in self.state.particles],
             "pair_tests":self.real_pairs,"planning_pair_tests":self.planning_pairs,**self.account.counts()}
        if diagnostics is not None:
            row.update({k:diagnostics[k] for k in ["minimum_surface_gap","minimum_pair_gap","pair_collisions","wall_collisions"]})
        self.rows.append(row)
        return row

    def supply(self,state,*,target_direct):
        energy=self.method.energy(state)-self.method.energy(self.state)
        momentum=state.momentum-self.state.momentum
        if self.positive_supply+max(energy,0.)>self.config.strategy.energy_budget+1e-10:
            raise FloatingPointError("Physical positive supply budget exceeded.")
        self.source+=energy;self.positive_supply+=max(energy,0.);self.extracted+=max(-energy,0.)
        self.source_momentum+=momentum
        if target_direct:self.target_source+=energy
        self.state=state;self.shots+=1
        self.record("intervention")
        return float(energy),momentum.tolist()

    def advance(self,method,*,score=True):
        before=self.state;offset=self.time;cfg=method.config.physics
        pairs_before=method.pair_tests
        try:
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                outcome=method.step(self.account.view("dynamics"),before,basin_boundary=self.boundary)
            numeric=[v for v in outcome.diagnostics.values() if isinstance(v,(int,float,np.number))]
            if not np.all(np.isfinite(numeric)):raise FloatingPointError("Nonfinite physical diagnostic.")
        finally:
            self.real_pairs+=method.pair_tests-pairs_before
        self.state=outcome.state;self.value=outcome.values[0];self.time=offset+cfg.h
        if score and self.value<self.best_value:self.best_value=self.value;self.best_x=self.state.particles[0].x.copy()
        d=outcome.diagnostics
        self.damping+=d["damping_energy"];self.impulse_error+=d["impulse_energy"]
        self.dynamics_momentum+=np.array(d["damping_impulse"])+np.array(d["gravity_impulse"])+np.array(d["wall_impulse"])
        self.target_damping+=(cfg.beta**2-1)*before.particles[0].kinetic
        self.gravity_work+=cfg.gravity*(before.particles[0].z-self.state.particles[0].z)
        for crossing in outcome.crossings:
            if crossing["particle"]==0 and self.first_crossing is None:self.first_crossing=offset+crossing["time"]
        for e in outcome.events:
            self.events.append({**e,"time":offset+e["time"]})
            if e["i"]==0 or e.get("j")==0:
                suffix="i" if e["i"]==0 else "j"
                v=np.array(e[f"before_{suffix}"]);w=np.array(e[f"after_{suffix}"])
                gain=float(.5*(w @ w-v @ v))
                if e["kind"]=="pair":self.target_transfer+=gain
                else:self.target_wall+=gain
        if score and self.first_target is None and self.target_hit():self.first_target=self.time
        return self.record("observation" if score else "partial_step",d)


def apply_scheduled_impulse(ledger,proposal,impulse):
    """Commit pre-impact flight, then the same first pair impulse, then drift.

    The second segment has no extra damping. Each segment is committed so a
    later budget failure cannot erase an already applied external pulse.
    """
    population=ledger.config.strategy.population;cfg=population.physics;t0=ledger.time
    early=PopulationGravity(replace(population,physics=replace(cfg,h=proposal.contact_delay)))
    ledger.advance(early,score=False)
    target=ledger.state.particles[0];before=target.velocity
    after=before+impulse
    changed=State(target.x,after[:-1],target.z,after[-1])
    supplied,momentum=ledger.supply(PopulationState((changed,)+ledger.state.particles[1:]),target_direct=True)
    event={"time":ledger.time,"kind":"external_impulse","i":0,"j":None,
           "point_i":np.r_[target.x,target.z].tolist(),"before_i":before.tolist(),"after_i":after.tolist(),
           "energy_input":supplied,"momentum_input":momentum}
    ledger.events.append(event)
    remaining=cfg.h-proposal.contact_delay
    late=PopulationGravity(replace(population,physics=replace(cfg,h=remaining,beta=1.)))
    return event,late,t0


@dataclass
class PredictiveRun(StrategicRun):
    decisions: list[dict]


def run_predictive(oracle,initial,steps,*,mode="predictive_ball",config=None,seed=0,
                   basin_boundary=None,target_value=None,cost_checkpoints=(1500,)):
    if mode not in MODES or not isinstance(steps,int) or steps<0 or len(initial.particles)!=1:
        raise ValueError("Require a known mode, nonnegative steps, and one original target.")
    if sum(oracle.counts().values()):raise ValueError("Use a fresh oracle per physical run.")
    config=config or PredictiveConfig();strategy=config.strategy;physics=strategy.population.physics
    started=perf_counter();account=QueryAccount(oracle,checkpoints=cost_checkpoints)
    ledger=PhysicalLedger(account,initial,config,basin_boundary,target_value)
    account.on_checkpoint=lambda total:ledger.record("cost_checkpoint")
    monitor=StagnationMonitor(strategy);rng=np.random.default_rng(811106+seed)
    decisions=[];interventions=[];detector=[];history=[]
    next_allowed=0.;rejections=failed_actions=0;feedback=None;reason="completed"

    def save_history():
        p=ledger.state.particles[0]
        history.append({"time":ledger.time,"best_value":ledger.best_value,"x":p.x.copy(),"horizontal_kinetic":float(.5*(p.v @ p.v))})

    save_history()
    for index in range(steps):
        time=index*physics.h
        if abs(ledger.time-time)>1e-8:raise FloatingPointError("Physical observation clock drift.")
        if feedback is not None and time>=feedback["due"]-1e-9:
            gain=(feedback["baseline"]-ledger.best_value)/feedback["scale"]
            feedback["action"]["realized_gain"]=gain
            feedback["action"]["realized_best_at_horizon"]=ledger.best_value
            feedback["action"]["feedback_time"]=time
            failed_actions=failed_actions+1 if gain<=config.epsilon_gain else 0
            next_allowed=max(next_allowed,time+strategy.cooldown*2**failed_actions if failed_actions else time)
            feedback=None
        try:
            diagnostic=monitor.evaluate(history,time)
            if diagnostic is not None:detector.append(diagnostic)
            eligible=diagnostic is not None and diagnostic["eligible"] and time>=next_allowed-1e-9 and feedback is None
            enough=(ledger.shots<strategy.max_shots and ledger.positive_supply<strategy.energy_budget-1e-12
                    and (steps-index)*physics.h>=config.horizon-1e-9)
            action=None;proposal=None;decision=None;pending_impulse=None
            if eligible and enough and mode!="continue":
                clearance=ledger.state.particles[0].z-physics.height_scale*ledger.value
                if clearance>=strategy.clearance:
                    if mode=="random_kick" and ledger.positive_supply+strategy.shot_energy<=strategy.energy_budget+1e-12:
                        direction=rng.normal(size=initial.particles[0].x.size)
                        changed=kinetic_kick(ledger.state,strategy.shot_energy,direction)
                        supplied,momentum=ledger.supply(changed,target_direct=True)
                        action={"time":time,"kind":mode,"energy_input":supplied,"momentum_input":momentum,
                                "direction":(direction/np.linalg.norm(direction)).tolist(),"planned_contact_observed":False}
                    elif mode!="random_kick" and ledger.decisions<config.max_decisions:
                        reserve=6*len(ledger.state.particles)*(steps-index)+100
                        available=config.session_queries if oracle.total_budget is None else max(0,oracle.total_budget-sum(oracle.counts().values())-reserve)
                        ledger.decisions+=1
                        decision=choose_launch(account,ledger.state,ledger.best_value,config,rng,strategy.energy_budget-ledger.positive_supply,
                                               query_limit=available,on_branch=lambda:ledger.record("planning"))
                        decision.log.update({"time":time,"number":ledger.decisions})
                        decisions.append(decision.log);ledger.planning_pairs+=decision.log["planning_pair_tests"]
                        ledger.record("planning")
                        proposal=decision.proposal
                        if proposal is None:
                            rejections+=1;next_allowed=time+strategy.cooldown*2**min(rejections,4)
                        else:
                            rejections=0
                            action={"time":time,"kind":mode,"decision":ledger.decisions,
                                    "direction":(-proposal.outward_direction).tolist(),"planned_contact_time":time+proposal.contact_delay,
                                    "planned_contact_observed":False,"virtual_supply":proposal.energy,
                                    "predicted_gain":decision.log["predicted_gain"],"predicted_best":decision.log["candidates"][decision.log["selected"]]["score"]}
                            if mode=="predictive_ball":
                                helper=len(ledger.state.particles)
                                supplied,momentum=ledger.supply(PopulationState(ledger.state.particles+(proposal.helper,)),target_direct=False)
                                action.update({"energy_input":supplied,"momentum_input":momentum,"helper_index":helper})
                            else:pending_impulse=decision.impulse
                    if action is not None:
                        interventions.append(action)
                        monitor.action(time);next_allowed=time+config.horizon
                        feedback={"due":time+config.horizon,"baseline":decision.log["baseline"]["best_value"] if decision is not None else history[-1]["best_value"],
                                  "scale":1+abs(history[-1]["best_value"]),"action":action}
            if pending_impulse is not None:
                event,late,_=apply_scheduled_impulse(ledger,proposal,pending_impulse)
                action.update({"energy_input":event["energy_input"],"momentum_input":event["momentum_input"],
                               "actual_impulse_time":event["time"],"actual_impulse":pending_impulse.tolist()})
                ledger.advance(late)
            else:
                ledger.advance(ledger.method)
            for launch in interventions:
                if launch["kind"]!="predictive_ball" or launch["planned_contact_observed"]:continue
                launch["planned_contact_observed"]=any(e["kind"]=="pair" and {e["i"],e["j"]}=={0,launch["helper_index"]}
                    and abs(e["time"]-launch["planned_contact_time"])<100*physics.time_tolerance for e in ledger.events)
            save_history()
        except EvaluationBudgetExceeded as exc:
            reason=f"budget: {exc}";ledger.record("stopped");break
        except ContactResolutionError as exc:
            reason=f"contact_error: {exc}";ledger.record("stopped");break
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            reason=f"numerical_error: {exc}";ledger.record("stopped");break
    if feedback is not None and ledger.time>=feedback["due"]-1e-9:
        feedback["action"].update({"realized_gain":(feedback["baseline"]-ledger.best_value)/feedback["scale"],
                                   "realized_best_at_horizon":ledger.best_value,"feedback_time":ledger.time})
    counts={**account.counts(),"pair_tests":ledger.real_pairs,"planning_pair_tests":ledger.planning_pairs,
            "launch_attempts":ledger.decisions,"shots":ledger.shots,
            "accepted_decisions":sum(d["reason"]=="accepted" for d in decisions),"rejected_decisions":sum(d["reason"]!="accepted" for d in decisions)}
    return PredictiveRun(ledger.rows,ledger.events,interventions,detector,ledger.state,ledger.best_value,ledger.best_x,
                         reason,counts,perf_counter()-started,ledger.first_crossing,ledger.first_target,decisions)
