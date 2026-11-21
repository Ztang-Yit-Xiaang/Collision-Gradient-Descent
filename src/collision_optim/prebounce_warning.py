"""Loss extrapolation during the horizontally ballistic phase."""

import numpy as np

from .cooling import drag_rate


def flight_factors(steps=6, h=.05):
    """Return horizontal displacement per unit initial horizontal speed."""
    beta = [np.exp(-h * drag_rate("ramp", step * h, h)) for step in range(steps)]
    return np.r_[0., h * np.cumsum(np.cumprod(beta))]


def directional_quadratic(distance, increases):
    """Fit b and k in Δf(s) = b s + k s² / 2 through two points and the origin."""
    x1, x2 = distance
    f1, f2 = increases
    curvature = 2 * (f2 / x2 - f1 / x1) / (x2 - x1)
    slope = f1 / x1 - .5 * curvature * x1
    return slope, curvature


def forecast_rise(observed_increases, speed, factors, prior_curvature=None):
    """Largest forecast rise at a future saved step in the nominal window."""
    observed = np.asarray(observed_increases, dtype=float)
    distance = speed * np.asarray(factors, dtype=float)
    if speed == 0:
        return float(observed[-1])
    if len(observed) == 1 and prior_curvature is not None:
        curvature = prior_curvature
        slope = observed[0] / distance[1] - .5 * curvature * distance[1]
    elif len(observed) == 2:
        slope, curvature = directional_quadratic(distance[1:3], observed)
    else:
        raise ValueError("Supply one observed increase with a curvature prior, or two increases.")
    future = distance[len(observed) + 1:]
    return float(np.max(slope * future + .5 * curvature * future**2))
