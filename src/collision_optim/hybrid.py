"""Augmented smooth-force dynamics with geometric, fully elastic contacts.

q=(x,z), z>=f(x), U=lambda*f(x)+(1-lambda)*z. The unit-mass force is
(-lambda*grad(f), -(1-lambda)). One macro step applies damping, a force kick,
and a drift containing correctly timed specular contacts. The horizontal free
update is momentum SGD; the complete augmented method is not native SGD.

The drift geometry is exact up to polynomial/root tolerances for the supported
synthetic objectives. The force kick/drift integrator does NOT conserve total
mechanical energy exactly. The residual is measured, never silently removed.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial

from .core import ContactEvent, State, StepResult, specular_reflection
from .objectives import CountedObjective


class ContactResolutionError(RuntimeError):
    """Contact cannot be resolved without hiding an invalid trajectory."""


@dataclass(frozen=True)
class HybridConfig:
    h: float = 0.05
    beta: float = 0.99
    mixing: float = 0.5
    gap_tolerance: float = 1e-9
    time_tolerance: float = 1e-11
    normal_tolerance: float = 1e-12
    max_contacts_per_step: int = 64

    def __post_init__(self):
        numbers = (self.h, self.beta, self.mixing, self.gap_tolerance,
                   self.time_tolerance, self.normal_tolerance)
        if not np.all(np.isfinite(numbers)):
            raise ValueError("Configuration must be finite.")
        if self.h <= 0 or not 0 <= self.beta <= 1 or not 0 < self.mixing < 1:
            raise ValueError("Require h>0, beta in [0,1], and mixing in (0,1).")
        if min(self.gap_tolerance, self.time_tolerance, self.normal_tolerance) <= 0:
            raise ValueError("Contact tolerances must be positive.")
        if not isinstance(self.max_contacts_per_step, int) or self.max_contacts_per_step < 1:
            raise ValueError("max_contacts_per_step must be a positive integer.")


def _real_unit_roots(coefficients, tolerance):
    coefficients = np.polynomial.polynomial.polytrim(np.asarray(coefficients, dtype=float))
    if coefficients.size < 2:
        return []
    roots = Polynomial(coefficients).roots()
    real = [float(root.real) for root in roots
            if abs(root.imag) <= 1e-8 * (1 + abs(root.real))
            and -tolerance <= root.real <= 1 + tolerance]
    return sorted(np.clip(real, 0.0, 1.0))


def first_contact_time(oracle, x, z, vx, vz, duration, config):
    """First incoming graph contact during a constant-velocity drift.

    All roots of z(t)-f(x(t)) are considered, not just an endpoint sign change.
    An interior-penetration check guards against unresolvable numerical roots.
    A genuinely grazing trajectory requiring sustained contact is rejected
    explicitly: no unphysical projection or normal-velocity deletion is used.
    """
    if duration <= 0:
        return None
    coefficients = -oracle.path_coefficients(x, duration * vx)
    if coefficients.size < 2:
        coefficients = np.pad(coefficients, (0, 2 - coefficients.size))
    coefficients[0] += z
    coefficients[1] += duration * vz
    gap = Polynomial(coefficients)
    derivative = gap.deriv()
    cfg = config
    time_eps = min(0.25, cfg.time_tolerance / duration)
    gap0 = float(gap(0))
    if gap0 < -cfg.gap_tolerance:
        raise ContactResolutionError(f"Initial point penetrates the surface: gap={gap0:.3g}")

    if abs(gap0) <= cfg.gap_tolerance:
        slope0 = float(derivative(0))
        if gap0 <= 0 and slope0 < -cfg.normal_tolerance * duration:
            return 0.0
        if abs(slope0) <= cfg.normal_tolerance * duration:
            higher = coefficients[2:]
            significant = higher[np.abs(higher) > cfg.normal_tolerance * duration]
            if significant.size and significant[0] < 0:
                raise ContactResolutionError(
                    "Grazing contact requires a sustained-contact solver; "
                    "trajectory stopped rather than projected through the wall."
                )

    for fraction in _real_unit_roots(coefficients, time_eps):
        slope = float(derivative(fraction))
        if slope < -cfg.normal_tolerance * duration:
            # A zero root with an outgoing normal velocity is not a collision.
            if fraction <= time_eps and float(derivative(0)) > 0:
                continue
            if abs(float(gap(fraction))) > 10 * cfg.gap_tolerance:
                raise ContactResolutionError("Contact root has an excessive residual.")
            return float(fraction * duration)

    candidates = [0.0, 1.0] + _real_unit_roots(derivative.coef, time_eps)
    minimum = min(float(gap(t)) for t in candidates)
    if minimum < -cfg.gap_tolerance:
        raise ContactResolutionError(
            f"No incoming root was resolved although the drift penetrates: gap={minimum:.3g}"
        )
    return None


class HybridContact:
    """Contact-driven reflection, with no objective-improvement trigger."""

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()

    def initialize(self, oracle: CountedObjective, x, v, *, gap=0.1, vz=0.0):
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("Initial gap must be finite and nonnegative.")
        x = np.asarray(x, dtype=float)
        return State(x, v, oracle.value(x) + gap, vz)

    def step(self, oracle: CountedObjective, state: State) -> StepResult:
        if state.z is None:
            raise ValueError("Hybrid contact needs both height and vertical velocity.")
        cfg = self.config
        f0 = oracle.value(state.x)
        gap0 = state.z - f0
        if gap0 < -cfg.gap_tolerance:
            raise ContactResolutionError(f"Initial point penetrates the surface: gap={gap0:.3g}")
        gradient = oracle.gradient(state.x)
        force = np.r_[-cfg.mixing * gradient, -(1 - cfg.mixing)]
        damped = cfg.beta * state.velocity
        velocity = damped + cfg.h * force
        point = np.r_[state.x, state.z].astype(float)
        damping_energy = float(0.5 * damped @ damped - state.kinetic)
        remaining, elapsed = cfg.h, 0.0
        events = []
        while remaining > 0:
            contact = first_contact_time(
                oracle, point[:-1], point[-1], velocity[:-1], velocity[-1], remaining, cfg
            )
            if contact is None:
                point += remaining * velocity
                elapsed += remaining
                remaining = 0.0
                break
            if len(events) >= cfg.max_contacts_per_step:
                raise ContactResolutionError(
                    "Contact limit exceeded (possible chattering); no state was committed."
                )
            point += contact * velocity
            elapsed += contact
            remaining = max(0.0, cfg.h - elapsed)
            contact_f = oracle.value(point[:-1])
            if abs(point[-1] - contact_f) > 10 * cfg.gap_tolerance:
                raise ContactResolutionError("Resolved contact is not on the graph.")
            normal = np.r_[-oracle.gradient(point[:-1]), 1.0]
            normal /= np.linalg.norm(normal)
            if velocity @ normal >= -cfg.normal_tolerance:
                raise ContactResolutionError("Contact is not incoming or is numerically grazing.")
            before = velocity.copy()
            velocity = specular_reflection(before, normal)
            error = float(0.5 * (velocity @ velocity - before @ before))
            events.append(ContactEvent(elapsed, point.copy(), normal.copy(), before, velocity.copy(), error))
        result = State(point[:-1], velocity[:-1], float(point[-1]), float(velocity[-1]))
        value = oracle.value(result.x)
        gap = result.z - value
        if gap < -cfg.gap_tolerance:
            raise ContactResolutionError(f"End point penetrates the surface: gap={gap:.3g}")
        potential0 = cfg.mixing * f0 + (1 - cfg.mixing) * state.z
        potential = cfg.mixing * value + (1 - cfg.mixing) * result.z
        before_energy = state.kinetic + potential0
        after_energy = result.kinetic + potential
        impulse_error = float(sum(event.energy_error for event in events))
        return StepResult(result, value, {
            "gradient_norm": float(np.linalg.norm(gradient)),
            "direction_cosine": None,
            "paired_change": None,
            "diagnostic_se": 0.0,
            "loss_threshold": None,
            "streak": 0,
            "intervention": "contact" if events else "none",
            "contacts": len(events),
            "kinetic": result.kinetic,
            "potential": potential,
            "energy": after_energy,
            "damping_energy": damping_energy,
            "intervention_energy": impulse_error,
            "energy_residual": after_energy - before_energy - damping_energy - impulse_error,
            "gap": gap,
        }, tuple(events))

