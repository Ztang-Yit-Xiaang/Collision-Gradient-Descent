"""Forecast one independent particle using paid line polynomials and contact geometry."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

from .cooling import drag_rate
from .core import State, specular_reflection
from .gravity import GravityConfig, GravityContact, ballistic_contact_time


@dataclass(frozen=True)
class _RestrictedPath:
    coefficients: np.ndarray

    def path_coefficients(self, _point, _displacement):
        return self.coefficients


def normal_quadratic_score(state, loss, initial_loss, event, curvature, start_step, end_step, h=.05):
    speed = float(np.linalg.norm(state.v))
    if speed == 0:
        return loss - initial_loss
    direction = state.v / speed
    normal = np.asarray(event["normal"])
    contact = np.asarray(event["point_i"])
    slope = float((-normal[:-1] / normal[-1]) @ direction)
    slope += curvature * float((state.x - contact[:-1]) @ direction)
    beta = [np.exp(-h * drag_rate("ramp", index * h, h)) for index in range(start_step, end_step)]
    distances = speed * h * np.cumsum(np.cumprod(beta))
    return float(loss - initial_loss + np.max(slope * distances + .5 * curvature * distances**2))


def forecast_after_bounce(oracle, state, initial_loss, limit, start_step, end_step, *, initial_path=None):
    """Use one path per uninterrupted flight and at most one future reflection."""
    cfg = GravityConfig()
    gravity = GravityContact(cfg)
    path = Polynomial(oracle.path_coefficients(state.x, state.v)) if initial_path is None else initial_path
    elapsed_parameter = 0.
    speed_factor = 1.
    observations = []
    next_contact = None
    for step in range(start_step + 1, end_step + 1):
        beta = np.exp(-cfg.h * drag_rate("ramp", (step - 1) * cfg.h, cfg.h))
        state = State(state.x, beta * state.v, state.z, beta * state.vz)
        speed_factor *= beta
        remaining = cfg.h
        while remaining > cfg.time_tolerance:
            subpath = path(Polynomial([elapsed_parameter, remaining * speed_factor]))
            hit = ballistic_contact_time(_RestrictedPath(subpath.coef), state, remaining, cfg)
            if hit is None:
                state = gravity._flight(state, remaining)
                elapsed_parameter += speed_factor * remaining
                break
            if next_contact is not None:
                return {"observations": observations, "next_contact": next_contact, "complete": False}
            next_contact = step * cfg.h - remaining + hit
            state = gravity._flight(state, hit)
            normal = np.r_[-oracle.gradient(state.x), 1.]
            after = specular_reflection(state.velocity, normal)
            state = State(state.x, after[:-1], state.z, after[-1])
            path = Polynomial(oracle.path_coefficients(state.x, state.v))
            elapsed_parameter = 0.
            speed_factor = 1.
            remaining -= hit
        rise = float(path(elapsed_parameter) - initial_loss)
        observations.append({"step": step, "loss_rise": rise, "after_next_contact": next_contact is not None})
        if rise > limit:
            break
    return {"observations": observations, "next_contact": next_contact, "complete": True}
