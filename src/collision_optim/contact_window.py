"""Use one paid line query to forecast the next ballistic terrain contact."""

import numpy as np
from numpy.polynomial import Polynomial

from .cooling import drag_rate
from .core import State
from .gravity import GravityContact, ballistic_contact_time
from .postbounce_warning import _RestrictedPath


def forecast_first_contact(oracle, state, config, start_step, end_step):
    cfg = config.physics
    path = Polynomial(oracle.path_coefficients(state.x, state.v))
    gravity = GravityContact(cfg)
    offset, scale = 0., 1.
    for step in range(start_step, end_step):
        beta = np.exp(-cfg.h * drag_rate("ramp", step * cfg.h, cfg.h))
        state = State(state.x, beta * state.v, state.z, beta * state.vz)
        scale *= beta
        local = path(Polynomial([offset, cfg.h * scale]))
        hit = ballistic_contact_time(_RestrictedPath(local.coef), state, cfg.h, cfg)
        if hit is not None:
            return {"time": float(step*cfg.h+hit), "loss": float(path(offset+scale*hit))}
        state = gravity._flight(state, cfg.h)
        offset += cfg.h * scale
    return None
