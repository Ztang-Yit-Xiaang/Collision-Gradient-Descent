"""Unit-mass gravity above a fixed graph, with ballistic and sliding phases.

Only vertical gravity acts in flight. Horizontal acceleration during sustained
contact is a normal reaction, not an additional gradient force. Flight/contact
geometry uses the synthetic objective's exact line polynomial (degree <=4).
Sliding is integrated numerically and its energy error is measured explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial

from .core import ContactEvent, State, StepResult, specular_reflection
from .hybrid import ContactResolutionError, _real_unit_roots
from .objectives import CountedObjective


@dataclass(frozen=True)
class GravityConfig:
    h: float = 0.05
    beta: float = 0.995
    gravity: float = 0.5
    height_scale: float = 1.0
    gap_tolerance: float = 1e-9
    time_tolerance: float = 1e-11
    normal_tolerance: float = 1e-11
    rtol: float = 1e-7
    atol: float = 1e-10
    energy_tolerance: float = 1e-9
    max_events: int = 256
    max_substeps: int = 4096

    def __post_init__(self):
        positive = (self.h, self.gravity, self.height_scale, self.gap_tolerance,
                    self.time_tolerance, self.normal_tolerance, self.rtol,
                    self.atol, self.energy_tolerance)
        if not np.all(np.isfinite((*positive, self.beta))):
            raise ValueError("Gravity configuration must be finite.")
        if min(positive) <= 0 or not 0 <= self.beta <= 1:
            raise ValueError("Require positive scales/tolerances and beta in [0,1].")
        if any(not isinstance(n, int) or n < 1 for n in (self.max_events, self.max_substeps)):
            raise ValueError("Event and substep limits must be positive integers.")


def ballistic_contact_time(oracle, state, duration, config):
    """Earliest incoming contact along an exact constant-gravity flight."""
    cfg = config
    if duration <= 0:
        return None
    coefficients = -cfg.height_scale * oracle.path_coefficients(state.x, duration * state.v)
    coefficients = np.pad(coefficients, (0, max(0, 3 - coefficients.size)))
    coefficients[0] += state.z
    coefficients[1] += duration * state.vz
    coefficients[2] -= 0.5 * cfg.gravity * duration**2
    gap = Polynomial(coefficients)
    derivative = gap.deriv()
    # At near-zero horizontal speed, negligible cubic/quartic terms create
    # enormous irrelevant roots and an ill-conditioned companion matrix. Drop
    # only a tail whose TOTAL effect on [0,1] is bounded below gap tolerance.
    # Validate contacts against the original polynomial, not the reduced one.
    root_coefficients = coefficients.copy()
    tail_limit = min(1e-14*max(1., float(np.max(np.abs(coefficients)))),
                     .01*cfg.gap_tolerance)
    removed = 0.
    while root_coefficients.size > 3 and removed+abs(root_coefficients[-1]) <= tail_limit:
        removed += abs(root_coefficients[-1])
        root_coefficients = root_coefficients[:-1]
    eps = min(0.25, cfg.time_tolerance / duration)
    if gap(0) < -cfg.gap_tolerance:
        raise ContactResolutionError("Initial point penetrates the gravity terrain.")
    for fraction in _real_unit_roots(root_coefficients, eps):
        slope = float(derivative(fraction))
        if slope < -cfg.normal_tolerance * duration:
            if fraction <= eps and derivative(0) > 0:
                continue
            if abs(gap(fraction)) > 10 * cfg.gap_tolerance:
                raise ContactResolutionError("Ballistic contact root has excessive residual.")
            return float(fraction * duration)
    candidates = [0.0, 1.0] + _real_unit_roots(Polynomial(root_coefficients).deriv().coef, eps)
    if min(float(gap(t)) for t in candidates) < -cfg.gap_tolerance:
        raise ContactResolutionError("Unresolved ballistic contact or unsupported grazing event.")
    return None


class GravityContact:
    """Pure vertical gravity, elastic impacts, and unilateral sustained contact."""

    def __init__(self, config: GravityConfig | None = None):
        self.config = config or GravityConfig()

    def potential_energy(self, state, value):
        return self.config.gravity * state.z

    def initialize(self, oracle: CountedObjective, x, v, *, gap=0.2, vz=-0.1,
                   on_surface=False):
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("Initial clearance must be finite and nonnegative.")
        x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
        if on_surface:
            if gap != 0:
                raise ValueError("An on-surface initial state requires gap=0.")
            vz = self.config.height_scale * float(oracle.gradient(x) @ v)
        return State(x, v, self.config.height_scale * oracle.value(x) + gap, vz)

    def _geometry(self, oracle, x, v):
        """Graph gradient and directional curvature; no dense Hessian."""
        gradient = self.config.height_scale * oracle.gradient(x)
        coefficients = oracle.path_coefficients(x, v)
        curvature = (2 * self.config.height_scale * coefficients[2]
                     if coefficients.size > 2 else 0.0)
        support = self.config.gravity + curvature
        return gradient, float(support)

    def _rhs(self, oracle, x, v):
        gradient, support = self._geometry(oracle, x, v)
        acceleration = -support * gradient / (1 + gradient @ gradient)
        return v, acceleration, support

    def _rk4(self, oracle, x, v, dt):
        k1x, k1v, a = self._rhs(oracle, x, v)
        k2x, k2v, b = self._rhs(oracle, x + dt*k1x/2, v + dt*k1v/2)
        k3x, k3v, c = self._rhs(oracle, x + dt*k2x/2, v + dt*k2v/2)
        k4x, k4v, d = self._rhs(oracle, x + dt*k3x, v + dt*k3v)
        return (x + dt*(k1x+2*k2x+2*k3x+k4x)/6,
                v + dt*(k1v+2*k2v+2*k3v+k4v)/6, min(a, b, c, d))

    def _surface_state(self, oracle, x, v):
        gradient = self.config.height_scale * oracle.gradient(x)
        return State(x, v, self.config.height_scale * oracle.value(x), float(gradient @ v))

    def _slide(self, oracle, state, duration):
        """Adaptive surface integration until duration or zero support force.

        Negative reaction is used only to bracket a detachment event. No accepted
        sliding segment is allowed to have a resolved negative support force.
        """
        cfg = self.config
        x, v = state.x.copy(), state.v.copy()
        current, elapsed, dt = state, 0.0, duration
        accepted = rejected = 0
        while duration - elapsed > cfg.time_tolerance:
            if accepted + rejected >= cfg.max_substeps:
                raise ContactResolutionError("Sliding substep limit exceeded.")
            dt = min(dt, duration - elapsed)
            coarse_x, coarse_v, coarse_support = self._rk4(oracle, x, v, dt)
            half_x, half_v, first_support = self._rk4(oracle, x, v, dt/2)
            fine_x, fine_v, second_support = self._rk4(oracle, half_x, half_v, dt/2)
            fine = self._surface_state(oracle, fine_x, fine_v)
            gradient, end_support = self._geometry(oracle, fine_x, fine_v)
            error = max(np.linalg.norm(fine_x-coarse_x), np.linalg.norm(fine_v-coarse_v)) / 15
            scale = cfg.atol + cfg.rtol * max(np.linalg.norm(x), np.linalg.norm(v),
                                             np.linalg.norm(fine_x), np.linalg.norm(fine_v))
            energy0 = current.kinetic + cfg.gravity*current.z
            energy1 = fine.kinetic + cfg.gravity*fine.z
            energy_error = abs(energy1-energy0)
            negative_support = min(coarse_support, first_support, second_support, end_support) < -cfg.normal_tolerance
            if negative_support:
                # Refinement localizes detachment without continuing on a surface
                # that would require an attractive (negative) normal reaction.
                _, start_support = self._geometry(oracle, x, v)
                if dt <= 10*cfg.time_tolerance:
                    if start_support <= 100*cfg.normal_tolerance:
                        return current, elapsed, accepted, rejected, True
                    raise ContactResolutionError("Could not localize the detachment event.")
                dt /= 2
                rejected += 1
                continue
            if error > scale or energy_error > cfg.energy_tolerance*(1+abs(energy0)):
                if dt <= 10*cfg.time_tolerance:
                    raise ContactResolutionError("Sliding error exceeds tolerance at minimum time step.")
                dt /= 2
                rejected += 1
                continue
            current, x, v = fine, fine_x, fine_v
            elapsed += dt
            accepted += 1
            if error < scale/32 and energy_error < cfg.energy_tolerance*(1+abs(energy0))/32:
                dt *= 2
        return current, elapsed, accepted, rejected, False

    def _flight(self, state, dt):
        return State(state.x + dt*state.v, state.v,
                     state.z + dt*state.vz - 0.5*self.config.gravity*dt**2,
                     state.vz - self.config.gravity*dt)

    def step(self, oracle: CountedObjective, state: State) -> StepResult:
        if state.z is None:
            raise ValueError("Gravity contact requires a height and vertical velocity.")
        cfg = self.config
        f0 = oracle.value(state.x)
        if state.z-cfg.height_scale*f0 < -cfg.gap_tolerance:
            raise ContactResolutionError("Initial point penetrates the gravity terrain.")
        damped = State(state.x, cfg.beta*state.v, state.z, cfg.beta*state.vz)
        damping = damped.kinetic-state.kinetic
        current, elapsed = damped, 0.0
        events = []
        slide_steps = rejected_steps = detachments = 0
        slide_time = constraint_correction = 0.0
        skip_surface = False
        segments = 0
        gradient_norm = None
        while cfg.h-elapsed > cfg.time_tolerance:
            segments += 1
            if segments > cfg.max_events:
                raise ContactResolutionError("Gravity event limit exceeded.")
            remaining = cfg.h-elapsed
            value = oracle.value(current.x)
            gap = current.z-cfg.height_scale*value
            speed = None
            if not skip_surface and abs(gap) <= cfg.gap_tolerance:
                gradient = cfg.height_scale*oracle.gradient(current.x)
                gradient_norm = float(np.linalg.norm(gradient)/cfg.height_scale)
                normal = np.r_[-gradient, 1.0]
                normal /= np.linalg.norm(normal)
                speed = float(current.velocity @ normal)
            if speed is not None and abs(speed) <= cfg.normal_tolerance:
                _, support = self._geometry(oracle, current.x, current.v)
                if support >= -cfg.normal_tolerance:
                    snapped = State(current.x, current.v, cfg.height_scale*value,
                                    float(gradient @ current.v))
                    constraint_correction += (snapped.kinetic-current.kinetic
                                              + cfg.gravity*(snapped.z-current.z))
                    current, used, accepted, rejected, detached = self._slide(oracle, snapped, remaining)
                    elapsed += used
                    slide_time += used
                    slide_steps += accepted
                    rejected_steps += rejected
                    skip_surface = detached
                    detachments += int(detached)
                    if not detached:
                        break
                    continue
            skip_surface = False
            contact = ballistic_contact_time(oracle, current, remaining, cfg)
            if contact is None:
                current = self._flight(current, remaining)
                elapsed = cfg.h
                break
            current = self._flight(current, contact)
            elapsed += contact
            contact_value = oracle.value(current.x)
            if abs(current.z-cfg.height_scale*contact_value) > 10*cfg.gap_tolerance:
                raise ContactResolutionError("Ballistic impact is not on the terrain.")
            gradient = cfg.height_scale*oracle.gradient(current.x)
            gradient_norm = float(np.linalg.norm(gradient)/cfg.height_scale)
            normal = np.r_[-gradient, 1.0]
            normal /= np.linalg.norm(normal)
            before = current.velocity
            if before @ normal >= -cfg.normal_tolerance:
                raise ContactResolutionError("Ballistic impact is not incoming.")
            after = specular_reflection(before, normal)
            error = float(0.5*(after @ after-before @ before))
            events.append(ContactEvent(elapsed, np.r_[current.x, current.z], normal,
                                       before, after, error))
            current = State(current.x, after[:-1], current.z, after[-1])
        value = oracle.value(current.x)
        gap = current.z-cfg.height_scale*value
        if gap < -cfg.gap_tolerance:
            raise ContactResolutionError("Gravity end state penetrates the terrain.")
        energy0 = state.kinetic+cfg.gravity*state.z
        energy = current.kinetic+cfg.gravity*current.z
        impulse = float(sum(event.energy_error for event in events))
        return StepResult(current, value, {
            "gradient_norm": gradient_norm,
            "direction_cosine": None, "paired_change": None,
            "diagnostic_se": 0.0, "loss_threshold": None, "streak": 0,
            "intervention": "contact" if events else ("sliding" if slide_steps else "none"),
            "contacts": len(events), "kinetic": current.kinetic,
            "potential": cfg.gravity*current.z, "energy": energy,
            "damping_energy": damping, "intervention_energy": impulse,
            "constraint_correction_energy": constraint_correction,
            "energy_residual": energy-energy0-damping-impulse,
            "gravity_work": cfg.gravity*(state.z-current.z),
            "sliding_time": slide_time, "sliding_substeps": slide_steps,
            "rejected_substeps": rejected_steps, "detachments": detachments,
            "gap": gap,
        }, tuple(events))
