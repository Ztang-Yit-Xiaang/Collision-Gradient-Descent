"""Conservative horizontal velocity redistribution at a shared state."""

import numpy as np

from .core import State
from .initial_budget import transfer_velocity
from .population import PopulationState


def transfer_at_checkpoint(state, selected):
    horizontal = np.array([particle.v for particle in state.particles])
    center = horizontal.mean(axis=0)
    after = transfer_velocity(horizontal - center, selected) + center
    return PopulationState(tuple(
        State(particle.x, velocity, particle.z, particle.vz)
        for particle, velocity in zip(state.particles, after)
    ))


def capped_transfer_at_checkpoint(state, selected):
    """Minimize the selected horizontal speed at fixed total momentum and energy."""
    horizontal = np.array([particle.v for particle in state.particles])
    count = len(horizontal)
    if count < 3 or not 0 <= selected < count:
        raise ValueError("At least three particles and a valid selected index are required.")

    others = np.arange(count) != selected
    spectators = horizontal[others]
    center = horizontal.mean(axis=0)
    relative_energy = np.sum((horizontal - center)**2)
    radius = np.sqrt((count - 1) / count * relative_energy)
    center_speed = np.linalg.norm(center)
    target = np.zeros_like(center)
    if center_speed > radius:
        target = center * (1. - radius / center_speed)

    # Split the spectators' velocity into their new mean and zero-sum residuals.
    deviations = spectators - spectators.mean(axis=0)
    spread = np.sum(deviations**2)
    required_spread = count / (count - 1) * max(0., (radius-center_speed)*(radius+center_speed))
    if np.linalg.norm(target) > np.linalg.norm(horizontal[selected]) or (spread == 0. and required_spread > 0.):
        return state

    after = np.empty_like(horizontal)
    after[selected] = target
    other_mean = (horizontal.sum(axis=0) - target) / (count - 1)
    scale = np.sqrt(required_spread / spread) if spread > 0. else 0.
    after[others] = other_mean + scale * deviations
    return PopulationState(tuple(
        State(particle.x, velocity, particle.z, particle.vz)
        for particle, velocity in zip(state.particles, after)
    ))
