"""Resume a fixed-step independent population immediately after its first reflection."""

from dataclasses import replace

import numpy as np
from numpy.polynomial.polynomial import polyval

from .cooling import drag_rate
from .core import State, specular_reflection
from .curvature_energy import IndependentEnergyPopulation
from .gravity import ballistic_contact_time
from .hybrid import ContactResolutionError
from .population import PopulationGravity, PopulationState
from .postbounce_warning import _RestrictedPath


def first_reflection(oracle, start, config, step_before):
    """Pay for the first iteration of the original step and stop after reflection."""
    cfg = config.physics
    beta = float(np.exp(-cfg.h * drag_rate("ramp", step_before * cfg.h, cfg.h)))
    damped = PopulationState(tuple(State(p.x, beta*p.v, p.z, beta*p.vz) for p in start.particles))
    method = IndependentEnergyPopulation(config)
    method._validate(oracle, damped.particles)
    contacts = [
        (time, i) for i, p in enumerate(damped.particles)
        if (time := ballistic_contact_time(oracle, p, cfg.h, cfg)) is not None
    ]
    if not contacts:
        raise ContactResolutionError("Expected a first terrain contact in this step.")
    time, selected = min(contacts)
    if sum(abs(t - time) <= cfg.time_tolerance for t, _ in contacts) != 1:
        raise ContactResolutionError("The replay requires a single earliest terrain contact.")
    incoming = PopulationState(tuple(method._advance(damped.particles, time, 0., None, [])))
    point = incoming.particles[selected]
    value = oracle.value(point.x)
    if abs(point.z - cfg.height_scale * value) > 10 * cfg.gap_tolerance:
        raise ContactResolutionError("The reconstructed contact is off the surface.")
    normal = np.r_[-cfg.height_scale * oracle.gradient(point.x), 1.]
    normal /= np.linalg.norm(normal)
    if point.velocity @ normal >= -cfg.normal_tolerance:
        raise ContactResolutionError("The reconstructed contact is not incoming.")
    after = specular_reflection(point.velocity, normal)
    particles = list(incoming.particles)
    particles[selected] = State(point.x, after[:-1], point.z, after[-1])
    outgoing = PopulationState(tuple(particles))
    return outgoing, {
        "offset": time, "index": selected, "normal": normal.tolist(), "contact_value": value,
        "incoming_velocity": point.velocity.tolist(), "outgoing_velocity": after.tolist(),
        "damping_beta": beta, "damping_energy": damped.kinetic - start.kinetic,
        "damping_impulse": (damped.momentum - start.momentum).tolist(),
        "gravity_impulse": [0., 0., -len(particles)*cfg.gravity*time],
        "terrain_impulse": (after - point.velocity).tolist(),
        "terrain_work": float(.5 * (after @ after - point.velocity @ point.velocity)),
        "gravity_work": -cfg.gravity * sum(p.z - q.z for p, q in zip(outgoing.particles, start.particles)),
    }


def forecast_to_boundary(oracle, particle, remaining, config):
    """Predict one saved value with one paid path query if the flight stays free."""
    cfg = config.physics
    path = oracle.path_coefficients(particle.x, remaining * particle.v)
    hit = ballistic_contact_time(_RestrictedPath(path), particle, remaining, cfg)
    return float(polyval(1., path)) if hit is None else None


def finish_fraction(oracle, outgoing, remaining, config):
    """Finish the same step without applying damping a second time."""
    fraction = replace(config, physics=replace(config.physics, h=remaining, beta=1.))
    method = IndependentEnergyPopulation(fraction)
    return PopulationGravity.step(method, oracle, outgoing)
