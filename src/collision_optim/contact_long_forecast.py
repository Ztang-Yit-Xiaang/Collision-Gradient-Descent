"""Extend the exact contact warning by reusing its first paid line polynomial."""

from numpy.polynomial import Polynomial

from .gravity import GravityContact, ballistic_contact_time
from .postbounce_warning import _RestrictedPath, forecast_after_bounce


def forecast_from_reflection(oracle, particle, remaining, config, initial_loss, limit, boundary_step, end_step):
    """Predict the first saved crossing from the state immediately after reflection."""
    physics = config.physics
    path = Polynomial(oracle.path_coefficients(particle.x, remaining * particle.v))
    if ballistic_contact_time(_RestrictedPath(path.coef), particle, remaining, physics) is not None:
        return None, {"observations": [], "next_contact": None, "complete": False}
    value = float(path(1.))
    boundary = {"step": boundary_step, "loss_rise": value - initial_loss, "after_next_contact": False}
    if value - initial_loss > limit:
        return value, {"observations": [boundary], "next_contact": None, "complete": True}
    state = GravityContact(physics)._flight(particle, remaining)
    shifted_path = path(Polynomial([1., 1. / remaining]))
    tail = forecast_after_bounce(oracle, state, initial_loss, limit, boundary_step, end_step, initial_path=shifted_path)
    return value, {**tail, "observations": [boundary, *tail["observations"]]}
