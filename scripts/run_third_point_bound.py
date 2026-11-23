"""E52: screen the three saved points using contact-safe displacement bounds."""

import argparse
from pathlib import Path

import numpy as np

from collision_optim.cooling import drag_rate
from run_second_contact_reflection import BASE as E50, run
from run_third_saved_point import BASE as E51, FORECAST_CAP, PREFLIGHT as E51_PREFLIGHT

BASE = Path("output/collision_third_point_bound")
SPEC = Path("docs/superpowers/specs/2026-11-15-third-point-bound-design.md")
PREFLIGHT = E51_PREFLIGHT | {"mixed__s3213__late_low","mixed__s3216__late_low"}


def contact_safe_bounds(objective, state, cfg, start, first_step, last_step):
    """Bound the tilted well and the restoring transverse direction separately."""
    tilt = objective.base.tilt
    stationary = np.roots([1.,0.,-1.,tilt]).real
    def well(x):
        return .25*(x*x-1.)**2 + tilt*x
    center = min(stationary,key=well)
    minimum = well(center)
    local_radius = .01
    curvature = max(1.,3.*(abs(center)+local_radius)**2-1.)
    gravity, height, stiffness = cfg.gravity, cfg.height_scale, objective.stiffness
    x,y = objective.coordinates(state.x)
    vx,vy = objective.coordinates(state.v)
    vy *= np.copysign(1.,y)
    y = abs(y)
    speed = np.sqrt(max(0.,float(state.velocity@state.velocity)+2*gravity*(state.z-height*(minimum-1e-12))))
    drift, scale, drag_sum, previous = 0., 1., 0., start
    bounds = []
    for index in range(first_step,last_step+1):
        end = index*cfg.h
        if index!=first_step:
            beta = np.exp(-cfg.h*drag_rate("ramp",previous,cfg.h))
            scale *= beta
            drag_sum += 1.-beta
        drift += scale*(end-previous)
        previous = end
        remaining = end-start
        impulse = max(0.,speed-state.vz+gravity*remaining+speed*drag_sum)
        radius = 2.*(abs(x-center)+abs(vx)*drift)+1e-12
        transverse = y+vy*drift
        stays_positive = vy>=0. and y>height*stiffness*transverse*impulse*remaining
        stays_local = radius<=local_radius and height*curvature*impulse*remaining<=.5
        upper = minimum+.5*(curvature*radius**2+stiffness*transverse**2)
        bounds.append(float(upper) if stays_positive and stays_local else None)
    return bounds


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight","formal"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if (args.stage=="formal") != bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args, phase="E52", spec=SPEC, preflight=PREFLIGHT, forecast_cap=FORECAST_CAP,
        comparison=E51, third_saved_point=True, earlier_archives=(E50,), reflection_loss_bounds=contact_safe_bounds)
