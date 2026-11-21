"""Low-energy stagnation interventions with explicit external supply ledgers."""

from __future__ import annotations

from dataclasses import dataclass,field
from time import perf_counter
import numpy as np

from .core import State
from .gravity import GravityConfig,ballistic_contact_time
from .hybrid import ContactResolutionError
from .objectives import EvaluationBudgetExceeded
from .population import PopulationConfig,PopulationGravity,PopulationState


MODES=("continue","random_kick","geometry_kick","ball")


@dataclass(frozen=True)
class StrategicConfig:
    population: PopulationConfig=field(default_factory=lambda:PopulationConfig(
        physics=GravityConfig(h=.05,beta=.995),rho=.12,max_pair_tests=100000))
    window: float=1.
    patience: int=2
    epsilon_f: float=1e-4
    epsilon_k: float=.002
    epsilon_move: float=.05
    cooldown: float=4.
    shot_energy: float=.20
    max_shots: int=2
    energy_budget: float=.40
    launch_time: float=.025
    clearance: float=1e-4
    directions: int=8
    max_launch_attempts: int=20

    def __post_init__(self):
        positive=(self.window,self.cooldown,self.shot_energy,self.energy_budget,
                  self.launch_time,self.clearance)
        tolerances=(self.epsilon_f,self.epsilon_k,self.epsilon_move)
        if not np.all(np.isfinite((*positive,*tolerances))) or min(positive)<=0 or min(tolerances)<0:
            raise ValueError("Intervention scales must be finite and positive; tolerances nonnegative.")
        if any(not isinstance(n,int) or n<1 for n in (self.patience,self.max_shots,self.directions,self.max_launch_attempts)):
            raise ValueError("Intervention counts must be positive integers.")
        if self.launch_time>=self.population.physics.h or self.population.physics.beta<=0:
            raise ValueError("Launches need beta>0 and contact before the next damping tick.")


class StagnationMonitor:
    def __init__(self,config):
        self.config=config
        self.next_check=config.window
        self.blocked_until=0.
        self.streak=0

    def action(self,time):
        self.streak=0
        self.blocked_until=time+self.config.cooldown

    def evaluate(self,history,time):
        cfg=self.config
        if time+1e-10<self.next_check:
            return None
        self.next_check=time+cfg.window
        if time+1e-10<self.blocked_until:
            self.streak=0
            return None
        samples=[r for r in history if r["time"]>=time-cfg.window-1e-10]
        if not samples or time-samples[0]["time"]<cfg.window-1e-8:
            return None
        improvement=(samples[0]["best_value"]-samples[-1]["best_value"])/(1+abs(samples[0]["best_value"]))
        kinetic=float(np.mean([r["horizontal_kinetic"] for r in samples]))
        path=sum(np.linalg.norm(b["x"]-a["x"]) for a,b in zip(samples,samples[1:]))
        activity=float(path/cfg.window/(1+np.linalg.norm(samples[0]["x"])))
        quiet=improvement<=cfg.epsilon_f and kinetic<=cfg.epsilon_k and activity<=cfg.epsilon_move
        self.streak=self.streak+1 if quiet else 0
        return {"time":float(time),"relative_progress":float(improvement),
                "mean_horizontal_kinetic":kinetic,"activity":activity,
                "quiet":bool(quiet),"streak":self.streak,"eligible":self.streak>=cfg.patience}


@dataclass(frozen=True)
class LaunchProposal:
    helper: State
    outward_direction: np.ndarray
    contact_delay: float
    energy: float
    candidate: int


def plan_launch(method,oracle,state,config,rng):
    """Plan a real flight with fixed K+g*z supply and no intervening contacts.

    A common damping tick immediately follows insertion. Position/velocity
    construction explicitly includes beta so contact occurs at launch_time.
    The potential reference is fixed at z=0; infeasible energy is never topped up.
    """
    cfg=config.population.physics
    target=state.particles[0]
    direction=np.zeros(target.x.size+1)
    w=target.velocity
    available=config.shot_energy-cfg.gravity*target.z
    if available<=0:
        return None
    damped=[State(p.x,cfg.beta*p.v,p.z,cfg.beta*p.vz) for p in state.particles]
    # Existing motion must stay predictable throughout the short launch flight.
    for p in damped:
        if ballistic_contact_time(oracle,p,config.launch_time,cfg) is not None:
            return None
    for i,p in enumerate(damped):
        for q in damped[i+1:]:
            if method._pair_test(p,q,config.launch_time) is not None:
                return None
    for candidate in range(1,config.directions+1):
        horizontal=rng.normal(size=target.x.size)
        horizontal/=np.linalg.norm(horizontal)
        direction[:-1]=horizontal
        projection=float(w @ direction)
        discriminant=projection**2+2*available-float(w @ w)
        if discriminant<=0:
            continue
        relative_before=projection+np.sqrt(discriminant)
        if relative_before<=cfg.normal_tolerance:
            continue
        relative_after=cfg.beta*relative_before
        helper_w=w-relative_before*direction
        separation=config.population.rho+relative_after*config.launch_time
        helper=State(target.x+separation*horizontal,helper_w[:-1],target.z,helper_w[-1])
        if helper.z-cfg.height_scale*oracle.value(helper.x)<config.clearance:
            continue
        if any(np.linalg.norm(np.r_[helper.x-p.x,helper.z-p.z])<=config.population.rho+config.population.pair_tolerance for p in state.particles):
            continue
        damped_helper=State(helper.x,cfg.beta*helper.v,helper.z,cfg.beta*helper.vz)
        if ballistic_contact_time(oracle,damped_helper,config.launch_time,cfg) is not None:
            continue
        contact_target=method.gravity._flight(damped[0],config.launch_time)
        contact_helper=method.gravity._flight(damped_helper,config.launch_time)
        if min(p.z-cfg.height_scale*oracle.value(p.x) for p in [contact_target,contact_helper])<config.clearance:
            continue
        isolated=True
        for i,p in enumerate(damped):
            hit=method._pair_test(damped_helper,p,config.launch_time)
            if i==0:
                isolated=hit is not None and abs(hit-config.launch_time)<100*cfg.time_tolerance
            elif hit is not None:
                isolated=False
            if not isolated:
                break
        if not isolated:
            continue
        energy=float(helper.kinetic+cfg.gravity*helper.z)
        if abs(energy-config.shot_energy)>1e-10:
            raise FloatingPointError("Launch energy does not match its supply budget.")
        return LaunchProposal(helper,horizontal.copy(),config.launch_time,energy,candidate)
    return None


def kinetic_kick(state,energy,direction):
    """Add exactly energy to horizontal kinetic energy; retain vertical motion."""
    target=state.particles[0]
    d=np.asarray(direction,dtype=float)
    d=d/np.linalg.norm(d)
    projection=float(target.v @ d)
    amplitude=-projection+np.sqrt(projection**2+2*energy)
    changed=State(target.x,target.v+amplitude*d,target.z,target.vz)
    return PopulationState((changed,)+state.particles[1:])


@dataclass
class StrategicRun:
    rows: list[dict]
    events: list[dict]
    interventions: list[dict]
    detector: list[dict]
    final_state: PopulationState
    best_value: float
    best_x: np.ndarray
    stop_reason: str
    counts: dict
    wall_seconds: float
    first_crossing: float | None
    first_target: float | None


def run_strategic(oracle,initial,steps,*,mode="ball",config=None,seed=0,
                  basin_boundary=None,target_value=None):
    if mode not in MODES or not isinstance(steps,int) or steps<0:
        raise ValueError("Invalid strategy mode or step count.")
    if len(initial.particles)!=1:
        raise ValueError("This experiment starts with exactly one scored target.")
    config=config or StrategicConfig()
    method=PopulationGravity(config.population)
    monitor=StagnationMonitor(config)
    rng=np.random.default_rng(801105+seed)
    cfg=config.population.physics
    started=perf_counter()
    state=initial
    values,gap=method._validate(oracle,state.particles,initial=True)
    value=values[0];best_value=value;best_x=state.particles[0].x.copy()
    energy0=method.energy(state);momentum0=state.momentum
    source_energy=damping=impulse=0.
    target_energy0=state.particles[0].kinetic+cfg.gravity*state.particles[0].z
    target_source=target_damping=target_transfer=target_wall_error=target_gravity_work=0.
    source_momentum=np.zeros_like(momentum0)
    dynamics_momentum=np.zeros_like(momentum0)
    rows,events,interventions,detector,history=[],[],[],[],[]
    first_crossing=0. if basin_boundary is not None and state.particles[0].x[0]<basin_boundary else None
    first_target=None
    shots=attempts=0
    reason="completed"

    def target_hit():
        return target_value is not None and value<=target_value and (basin_boundary is None or state.particles[0].x[0]<basin_boundary)

    def record(time,kind,diag=None):
        target=state.particles[0]
        counts=oracle.counts()
        row={"time":float(time),"record":kind,"value":value,"best_value":best_value,
             "target_x":target.x.tolist(),"target_v":target.v.tolist(),"target_z":float(target.z),
             "target_vz":float(target.vz),"target_horizontal_kinetic":float(.5*(target.v @ target.v)),
             "particle_count":len(state.particles),"energy":method.energy(state),
             "cumulative_source_energy":source_energy,"cumulative_source_momentum":source_momentum.tolist(),
             "cumulative_damping":damping,"cumulative_impulse_error":impulse,
             "energy_balance_error":method.energy(state)-energy0-source_energy-damping-impulse,
             "momentum_balance_error":float(np.linalg.norm(state.momentum-momentum0-source_momentum-dynamics_momentum)),
             "target_energy":target.kinetic+cfg.gravity*target.z,
             "target_source_energy":target_source,"target_pair_energy_transfer":target_transfer,
             "target_damping_energy":target_damping,"target_wall_energy_error":target_wall_error,
             "target_gravity_work":target_gravity_work,
             "target_energy_balance_error":target.kinetic+cfg.gravity*target.z-target_energy0-target_source-target_damping-target_transfer-target_wall_error,
             "shots":shots,"launch_attempts":attempts,"first_crossing":first_crossing,
             "target_hit":first_target is not None,"first_target_time":first_target,
             "positions":[np.r_[p.x,p.z].tolist() for p in state.particles],
             "velocities":[p.velocity.tolist() for p in state.particles],
             "pair_tests":method.pair_tests,**counts,"total_oracle_calls":sum(counts.values())}
        if diag is not None:
            row.update({"pair_collisions":diag["pair_collisions"],"wall_collisions":diag["wall_collisions"],
                        "minimum_surface_gap":diag["minimum_surface_gap"],"minimum_pair_gap":diag["minimum_pair_gap"]})
        rows.append(row)
        return row

    if target_hit():
        first_target=0.
    record(0.,"observation")
    history.append({"time":0.,"best_value":best_value,"x":best_x.copy(),"horizontal_kinetic":rows[-1]["target_horizontal_kinetic"]})
    for index in range(steps):
        time=index*cfg.h
        try:
            diagnostic=monitor.evaluate(history,time)
            if diagnostic is not None:
                detector.append(diagnostic)
            eligible=diagnostic is not None and diagnostic["eligible"]
            enough=shots<config.max_shots and source_energy+config.shot_energy<=config.energy_budget+1e-10
            if eligible and enough and mode!="continue":
                target=state.particles[0]
                clearance=target.z-cfg.height_scale*value
                if clearance>=config.clearance:
                    proposal=None
                    if mode in {"ball","geometry_kick"}:
                        if attempts>=config.max_launch_attempts:
                            continue_action=False
                        else:
                            attempts+=1
                            proposal=plan_launch(method,oracle,state,config,rng)
                            continue_action=proposal is not None
                            if proposal is None:
                                interventions.append({"time":time,"kind":"infeasible_launch","attempt":attempts,"energy_input":0.})
                                record(time,"planning")
                    else:
                        continue_action=True
                    if continue_action:
                        before_energy=method.energy(state);before_momentum=state.momentum
                        if mode=="ball":
                            helper_index=len(state.particles)
                            state=PopulationState(state.particles+(proposal.helper,))
                            direction=-proposal.outward_direction
                        else:
                            helper_index=None
                            direction=-proposal.outward_direction if proposal is not None else rng.normal(size=target.x.size)
                            state=kinetic_kick(state,config.shot_energy,direction)
                        supplied=method.energy(state)-before_energy
                        supplied_momentum=state.momentum-before_momentum
                        if abs(supplied-config.shot_energy)>1e-10:
                            raise FloatingPointError("Intervention energy mismatch.")
                        source_energy+=supplied;source_momentum+=supplied_momentum
                        if mode!="ball":
                            target_source+=supplied
                        shots+=1
                        monitor.action(time)
                        intervention={"time":time,"kind":mode,"shot":shots,"helper_index":helper_index,
                                      "energy_input":float(supplied),"momentum_input":supplied_momentum.tolist(),
                                      "direction":(direction/np.linalg.norm(direction)).tolist(),
                                      "planned_contact_time":time+proposal.contact_delay if mode=="ball" else None,
                                      "planned_contact_observed":False,
                                      "helper_kinetic":proposal.helper.kinetic if mode=="ball" else None,
                                      "helper_potential":cfg.gravity*proposal.helper.z if mode=="ball" else None,
                                      "candidate":proposal.candidate if proposal is not None else None}
                        interventions.append(intervention)
                        # An insertion/pulse is already a complete event, even if
                        # the following drift exhausts its budget or is unsupported.
                        record(time,"intervention")
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                outcome=method.step(oracle,state,basin_boundary=basin_boundary)
            numeric=[v for v in outcome.diagnostics.values() if isinstance(v,(int,float,np.number))]
            if not np.all(np.isfinite(numeric)):
                raise FloatingPointError("Nonfinite strategic dynamics diagnostics.")
        except EvaluationBudgetExceeded as exc:
            reason=f"budget: {exc}";record(time,"stopped");break
        except ContactResolutionError as exc:
            reason=f"contact_error: {exc}";record(time,"stopped");break
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            reason=f"numerical_error: {exc}";record(time,"stopped");break
        previous_target=state.particles[0]
        target_damping+=(cfg.beta**2-1)*previous_target.kinetic
        target_gravity_work+=cfg.gravity*(previous_target.z-outcome.state.particles[0].z)
        state=outcome.state;value=outcome.values[0]
        if value<best_value:
            best_value=value;best_x=state.particles[0].x.copy()
        damping+=outcome.diagnostics["damping_energy"]
        impulse+=outcome.diagnostics["impulse_energy"]
        dynamics_momentum+=np.array(outcome.diagnostics["damping_impulse"])+np.array(outcome.diagnostics["gravity_impulse"])+np.array(outcome.diagnostics["wall_impulse"])
        for crossing in outcome.crossings:
            if crossing["particle"]==0 and first_crossing is None:
                first_crossing=time+crossing["time"]
        for event in outcome.events:
            full={**event,"time":time+event["time"]}
            events.append(full)
            if event["i"]==0 or event.get("j")==0:
                suffix="i" if event["i"]==0 else "j"
                before=np.asarray(event[f"before_{suffix}"])
                after=np.asarray(event[f"after_{suffix}"])
                transfer=float(.5*(after @ after-before @ before))
                if event["kind"]=="pair":
                    target_transfer+=transfer
                else:
                    target_wall_error+=transfer
            for launch in interventions:
                if launch["kind"]=="ball" and event["kind"]=="pair" and {event["i"],event["j"]}=={0,launch["helper_index"]}:
                    if abs(full["time"]-launch["planned_contact_time"])<100*cfg.time_tolerance:
                        launch["planned_contact_observed"]=True
        if first_target is None and target_hit():
            first_target=(index+1)*cfg.h
        row=record((index+1)*cfg.h,"observation",outcome.diagnostics)
        history.append({"time":row["time"],"best_value":best_value,"x":state.particles[0].x.copy(),
                        "horizontal_kinetic":row["target_horizontal_kinetic"]})
    counts={**oracle.counts(),"pair_tests":method.pair_tests,"launch_attempts":attempts,"shots":shots}
    counts["total_oracle_calls"]=sum(oracle.counts().values())
    return StrategicRun(rows,events,interventions,detector,state,best_value,best_x,reason,counts,
                        perf_counter()-started,first_crossing,first_target)
