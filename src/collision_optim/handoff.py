"""Committed checkpoints and explicit physical-to-GD interfaces."""

from dataclasses import dataclass
import numpy as np

from .cooling import CoolingPopulation
from .objectives import EvaluationBudgetExceeded


class HandoffPopulation(CoolingPopulation):
    def __init__(self, *args, handoff_budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoff_budget=handoff_budget
        self.resuming=False

    def _validate(self, oracle, particles, *, initial=False):
        # Initial separation is a setup constraint, not a constraint on
        # independent particles resuming after legitimately passing each other.
        return super()._validate(oracle,particles,initial=initial and not self.resuming)

    def step(self, oracle, state, *, basin_boundary=None):
        if self.handoff_budget is not None and sum(oracle.counts().values())>=self.handoff_budget:
            raise EvaluationBudgetExceeded("committed handoff threshold reached")
        return super().step(oracle,state,basin_boundary=basin_boundary)


def particle_record(p):
    return {"position":np.r_[p.x,p.z].tolist(),"velocity":p.velocity.tolist()}


def partition_state(state, values, gravity=.5):
    index=int(np.argmin(values));p=state.particles[index]
    energy=lambda q:q.kinetic+gravity*q.z
    exported=[q for i,q in enumerate(state.particles) if i!=index]
    kept_energy=energy(p);removed_energy=sum(energy(q) for q in exported)
    kept_momentum=p.velocity;removed_momentum=np.sum([q.velocity for q in exported],axis=0)
    return {"selected_index":index,"selected_value":float(values[index]),
        "selected_state":particle_record(p),"exported_states":[particle_record(q) for q in exported],
        "retained_energy":float(kept_energy),"exported_energy":float(removed_energy),
        "retained_momentum":kept_momentum.tolist(),"exported_momentum":removed_momentum.tolist(),
        "energy_partition_residual":float(state.kinetic+gravity*sum(q.z for q in state.particles)-kept_energy-removed_energy),
        "momentum_partition_residual":float(np.linalg.norm(state.momentum-kept_momentum-removed_momentum)),
        "physical_ledger_ends_at_interface":True,"selected_state_reset":False,
        "gd_mapping":"Use current selected x; velocity/height archived unchanged, not dynamical variables of GD"}


@dataclass
class GDRun:
    rows:list
    stop_reason:str
    counts:dict
    uncommitted:list


def gd_refine(oracle, x, value, *, eta=.02):
    if not np.isfinite(eta) or eta<=0:raise ValueError("eta must be positive.")
    x=np.array(x,dtype=float,copy=True);value=float(value)
    def counts():return {**oracle.counts(),"total_oracle_calls":sum(oracle.counts().values())}
    rows=[{"step":0,"x":x.tolist(),"value":value,"before_x":None,"gradient":None,"eta":None,**counts()}]
    reason="completed";attempts=[]
    while True:
        attempt={"before_x":x.tolist(),"gradient":None,"candidate_x":None}
        try:
            gradient=oracle.gradient(x);attempt["gradient"]=gradient.tolist()
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                candidate=x-eta*gradient
            attempt["candidate_x"]=candidate.tolist()
            candidate_value=oracle.value(candidate)
        except EvaluationBudgetExceeded as exc:
            reason=f"budget: {exc}"
            if attempt["gradient"] is not None:attempts.append({**attempt,**counts()})
            break
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            reason=f"numerical_error: {exc}";attempts.append({**attempt,**counts()});break
        rows.append({"step":len(rows),"x":candidate.tolist(),"value":candidate_value,
                     "before_x":x.tolist(),"gradient":gradient.tolist(),"eta":eta,**counts()})
        x,value=candidate,candidate_value
    return GDRun(rows,reason,counts(),attempts)
