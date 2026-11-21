"""Predetermined bulk-drag schedules; contact laws stay unchanged."""

from dataclasses import replace
import json
import numpy as np

from .scattering_population import ScatteringPopulation

HORIZON = 25.
GAMMA0 = -np.log(.995)/.05
GAMMA1 = .5
SCHEDULES = ("weak", "constant", "ramp")


def drag_rate(schedule, time, h):
    if schedule not in SCHEDULES or not np.isfinite(time+h) or time < 0 or h <= 0:
        raise ValueError("Invalid schedule, time or step.")
    if schedule == "weak": return float(GAMMA0)
    if schedule == "constant": return float((GAMMA0+GAMMA1)/2)
    return float(GAMMA0+(GAMMA1-GAMMA0)*np.clip((time+h/2)/HORIZON, 0, 1))


class CoolingPopulation(ScatteringPopulation):
    def __init__(self, config, *, schedule, rule="elastic", seed=0):
        super().__init__(config, rule=rule, seed=seed)
        drag_rate(schedule, 0., config.physics.h)
        self.schedule = schedule
        self.committed_steps = 0

    def step(self, oracle, state, *, basin_boundary=None):
        h = self.config.physics.h
        rate = drag_rate(self.schedule, self.committed_steps*h, h)
        beta = .995 if self.schedule == "weak" else float(np.exp(-rate*h))
        self.config = replace(self.config, physics=replace(self.config.physics, beta=beta))
        result = super().step(oracle, state, basin_boundary=basin_boundary)
        result.diagnostics.update({"drag_rate": rate, "drag_beta": beta,
                                   "drag_exposure": rate*h})
        self.committed_steps += 1
        return result


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def trajectory_metrics(rows, target, boundary, gravity=.5):
    """Discrete population hits; tail occupancy is left-held on observed intervals."""
    hits = [any(v <= target and (boundary is None or p[0] < boundary)
                for v,p in zip(decoded(r["values"]), decoded(r["positions"]))) for r in rows]
    times = np.array([float(r["time"]) for r in rows])
    first = next((i for i,h in enumerate(hits) if h), None)
    tail_start = .8*times[-1]
    weights = np.maximum(0., times[1:]-np.maximum(times[:-1], tail_start))
    occupancy = float(weights @ np.array(hits[:-1],float)/weights.sum()) if weights.sum() else None
    last = rows[-1]; positions = np.array(decoded(last["positions"]))
    values = np.array(decoded(last["values"]))
    return {"observed_ever_hit": any(hits), "observed_final_hit": hits[-1],
            "first_observed_hit_time": None if first is None else float(times[first]),
            "hit_exit_transitions": sum(a and not b for a,b in zip(hits,hits[1:])),
            "lost_after_any_hit": bool(any(hits) and not hits[-1]),
            "tail_target_occupancy": occupancy,
            "final_kinetic": float(last["kinetic"]),
            "final_clearance_potential": float(gravity*np.sum(positions[:,-1]-values)),
            "removed_energy": -float(last["cumulative_damping"]),
            "committed_drag_exposure": sum(float(r.get("drag_exposure") or 0.) for r in rows[1:])}
