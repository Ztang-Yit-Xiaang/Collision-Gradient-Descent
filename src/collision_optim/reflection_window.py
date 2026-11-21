"""Forecast a single independent flight starting at an exact reflection."""

import numpy as np
from numpy.polynomial import Polynomial

from .cooling import drag_rate
from .core import State, specular_reflection
from .gravity import GravityContact, ballistic_contact_time
from .postbounce_warning import _RestrictedPath


def _reflect(oracle, gravity, state, duration):
    incoming = gravity._flight(state, duration)
    gradient = oracle.gradient(incoming.x)
    normal = np.r_[-gravity.config.height_scale*gradient, 1.]
    velocity = specular_reflection(incoming.velocity, normal)
    outgoing = State(incoming.x, velocity[:-1], incoming.z, velocity[-1])
    return incoming, outgoing, gradient


def _resume_second_saved_point(oracle, gravity, state, step, hit, path):
    cfg = gravity.config
    incoming, outgoing, _ = _reflect(oracle, gravity, state, hit)
    contact = {
        "time": (step - 1)*cfg.h + hit, "loss": float(path(hit/cfg.h)),
        "incoming_velocity": incoming.velocity.tolist(), "outgoing_velocity": outgoing.velocity.tolist(),
    }
    remaining = cfg.h - hit
    follow = Polynomial(oracle.path_coefficients(outgoing.x, remaining*outgoing.v))
    another = ballistic_contact_time(_RestrictedPath(follow.coef), outgoing, remaining, cfg)
    observation = None
    if another is None:
        observation = {"step": step, "time": step*cfg.h, "loss": float(follow(1.))}
    return contact, observation


def _saved_window_after_reflection(oracle, gravity, state, start, step, end_step, saved_limit,
                                    reflection_loss_bounds, fourth_saved_point):
    cfg = gravity.config
    kinetic = .5 * float(state.velocity @ state.velocity)
    upper_loss = (state.z + kinetic/cfg.gravity)/cfg.height_scale
    observations, contacts = [], []
    result = {"energy_upper_loss":upper_loss, "extended_observations":observations, "extended_contacts":contacts}
    if upper_loss < saved_limit-1e-12:
        return {**result, "window_stop":"energy_bound"}

    fourth_available = fourth_saved_point and step+3<=end_step
    last_step = step+3 if fourth_available else min(step+2, end_step)
    if reflection_loss_bounds is not None:
        bounds = reflection_loss_bounds(state, cfg, start, step, last_step)
        result["saved_loss_upper_bounds"] = bounds
        if all(bound is not None and bound<saved_limit-1e-12 for bound in bounds):
            return {**result, "window_stop":"loss_bounds"}
        if fourth_available and bounds[-1] is not None and bounds[-1]<saved_limit-1e-12:
            last_step -= 1

    path = Polynomial(oracle.path_coefficients(state.x, state.v))
    parameter, scale = 0., 1.
    for index in range(step, last_step+1):
        end = index*cfg.h
        if index != step:
            start = (index-1)*cfg.h
            beta = np.exp(-cfg.h*drag_rate("ramp", start, cfg.h))
            state = State(state.x, beta*state.v, state.z, beta*state.vz)
            scale *= beta
        while True:
            duration = end-start
            local = path(Polynomial([parameter, scale*duration]))
            hit = ballistic_contact_time(_RestrictedPath(local.coef), state, duration, cfg)
            if hit is None:
                state = gravity._flight(state, duration)
                parameter += scale*duration
                break
            additional_contacts = 3 if index==step+3 else 2
            if len(contacts)==additional_contacts:
                return {**result, "window_stop":"reflection_limit", "unresolved_contact_time":start+hit}
            incoming, state, _ = _reflect(oracle, gravity, state, hit)
            contacts.append({"time":start+hit, "loss":float(path(parameter+scale*hit)),
                             "incoming_velocity":incoming.velocity.tolist(), "outgoing_velocity":state.velocity.tolist()})
            path = Polynomial(oracle.path_coefficients(state.x, state.v))
            parameter, scale = 0., 1.
            start += hit
        loss = float(path(parameter))
        observations.append({"step":index, "time":end, "loss":loss})
        if loss>saved_limit:
            return {**result, "window_stop":"saved_crossing"}
    return {**result, "window_stop":"completed"}


def forecast_flight(oracle, state, config, time, boundary_step, end_step, *, saved_limit=None,
                    minimum_contact_loss=None, curvature_bound=None, second_saved_point=False,
                    follow_second_contact=False, third_saved_point=False, reflection_loss_bounds=None,
                    fourth_saved_point=False):
    """Reuse one paid polynomial through the next contact or last observation."""
    cfg = config.physics
    gravity = GravityContact(cfg)
    path = Polynomial(oracle.path_coefficients(state.x, state.v))
    parameter, scale = 0., 1.
    observations = []
    for step in range(boundary_step, end_step + 1):
        start = time if step == boundary_step else (step - 1) * cfg.h
        duration = step * cfg.h - start
        if step != boundary_step:
            beta = np.exp(-cfg.h * drag_rate("ramp", start, cfg.h))
            state = State(state.x, beta * state.v, state.z, beta * state.vz)
            scale *= beta
        local = path(Polynomial([parameter, scale * duration]))
        hit = ballistic_contact_time(_RestrictedPath(local.coef), state, duration, cfg)
        if hit is not None:
            result = {"observations": observations, "next_contact": {
                "time": start + hit, "loss": float(path(parameter + scale * hit))}}
            contact_loss = result["next_contact"]["loss"]
            near_limit = minimum_contact_loss is None or contact_loss>=minimum_contact_loss
            if saved_limit is not None and near_limit and not any(p["loss"]>saved_limit for p in observations):
                _, outgoing, gradient = _reflect(oracle, gravity, state, hit)
                if third_saved_point or fourth_saved_point:
                    return {**result, **_saved_window_after_reflection(
                        oracle, gravity, outgoing, start+hit, step, end_step, saved_limit,
                        reflection_loss_bounds, fourth_saved_point)}
                remaining = duration-hit
                displacement = remaining*outgoing.v
                extend = second_saved_point and step<end_step
                if extend:
                    beta = np.exp(-cfg.h * drag_rate("ramp",step*cfg.h,cfg.h))
                    next_duration = beta*cfg.h
                    second_displacement = displacement+next_duration*outgoing.v
                if curvature_bound is not None:
                    upper = contact_loss+max(0.,float(gradient@displacement)+.5*curvature_bound(outgoing.x,displacement))
                    result["post_contact_bound"] = {"upper_loss":upper,"remaining_time":remaining,
                                                    "outgoing_velocity":outgoing.velocity.tolist()}
                    if extend:
                        second_upper = contact_loss+max(
                            0.,float(gradient@second_displacement)+.5*curvature_bound(outgoing.x,second_displacement))
                        result["second_saved_upper_loss"] = second_upper
                        upper = max(upper,second_upper)
                    if upper < saved_limit-1e-12:
                        return result
                scale = remaining if remaining>0. else cfg.h
                follow = Polynomial(oracle.path_coefficients(outgoing.x,scale*outgoing.v))
                first_path = follow(Polynomial([0.,remaining/scale]))
                contact = ballistic_contact_time(_RestrictedPath(first_path.coef),outgoing,remaining,cfg)
                result["after_contact_saved"] = (None if contact is not None else
                                                  {"step":step,"time":step*cfg.h,"loss":float(first_path(1.))})
                if extend and contact is None and first_path(1.)<=saved_limit:
                    boundary = gravity._flight(outgoing,remaining)
                    damped = State(boundary.x,beta*boundary.v,boundary.z,beta*boundary.vz)
                    second_path = follow(Polynomial([remaining/scale,next_duration/scale]))
                    contact = ballistic_contact_time(_RestrictedPath(second_path.coef),damped,cfg.h,cfg)
                    result["second_after_contact_saved"] = (None if contact is not None else
                        {"step":step+1,"time":(step+1)*cfg.h,"loss":float(second_path(1.))})
                    if follow_second_contact and contact is not None:
                        bounce, saved = _resume_second_saved_point(oracle, gravity, damped, step+1, contact, second_path)
                        result["second_contact"] = bounce
                        result["second_after_contact_saved"] = saved
            return result
        parameter += scale * duration
        state = gravity._flight(state, duration)
        observations.append({"step": step, "time": step * cfg.h, "loss": float(path(parameter))})
    return {"observations": observations, "next_contact": None}
