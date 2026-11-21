"""Shared-clock gravity with isolated binary particle and terrain impacts.

Masses are one. rho constrains center separation; the terrain constrains centers
independently. This initial population solver deliberately rejects sustained or
shared contacts instead of inventing an order-dependent multi-contact law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import numpy as np

from .core import State, elastic_pair_collision, specular_reflection
from .gravity import GravityConfig, GravityContact, ballistic_contact_time
from .hybrid import ContactResolutionError
from .objectives import CountedObjective, EvaluationBudgetExceeded


@dataclass(frozen=True)
class PopulationConfig:
    physics: GravityConfig = field(default_factory=lambda: GravityConfig(h=0.05, beta=0.995))
    rho: float = 0.12
    collisions: bool = True
    pair_tolerance: float = 1e-9
    max_events_per_step: int = 256
    max_pair_tests: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive.")
        if not np.isfinite(self.pair_tolerance) or self.pair_tolerance <= 0:
            raise ValueError("pair_tolerance must be finite and positive.")
        if not isinstance(self.max_events_per_step, int) or self.max_events_per_step < 1:
            raise ValueError("max_events_per_step must be a positive integer.")
        if self.max_pair_tests is not None and (not isinstance(self.max_pair_tests, int) or self.max_pair_tests < 0):
            raise ValueError("max_pair_tests must be a nonnegative integer or None.")
        if self.physics.h <= self.physics.time_tolerance:
            raise ValueError("The observation step must exceed the time tolerance.")


@dataclass(frozen=True)
class PopulationState:
    particles: tuple[State, ...]

    def __post_init__(self):
        particles = tuple(self.particles)
        if not particles or any(not isinstance(p, State) or p.z is None for p in particles):
            raise ValueError("A population needs particle states with heights.")
        if any(p.x.shape != particles[0].x.shape for p in particles):
            raise ValueError("All particles must have the same parameter dimension.")
        object.__setattr__(self, "particles", particles)

    @property
    def kinetic(self):
        return float(sum(p.kinetic for p in self.particles))

    @property
    def momentum(self):
        return np.sum([p.velocity for p in self.particles], axis=0)


class PopulationOracle(CountedObjective):
    """A shared population budget; initialization and posthoc scoring are outside."""

    def __init__(self, objective, budget=None, *, total_budget=None):
        super().__init__(objective, budget)
        if total_budget is not None and (not isinstance(total_budget, int) or total_budget < 0):
            raise ValueError("total_budget must be a nonnegative integer or None.")
        self.total_budget = total_budget

    def _reserve(self, kind, amount=1):
        total = self.values+self.gradients+self.paths
        if self.total_budget is not None and total+amount > self.total_budget:
            raise EvaluationBudgetExceeded(f"total oracle budget exhausted ({total}/{self.total_budget})")
        super()._reserve(kind, amount)


def pair_contact_time(left: State, right: State, duration: float, config: PopulationConfig):
    """First incoming hard-core contact when both particles have the same gravity.

    Relative acceleration cancels exactly between common damping ticks. The
    stable small-root formula avoids subtracting two nearly equal numbers.
    """
    if duration <= 0:
        return None
    r = np.r_[left.x-right.x, left.z-right.z]
    u = left.velocity-right.velocity
    distance = float(np.linalg.norm(r))
    if distance < config.rho-config.pair_tolerance:
        raise ContactResolutionError("Particles overlap before a binary flight.")
    a, b = float(u @ u), float(r @ u)
    if a <= 1e-30 or b >= 0:
        return None
    c = float(r @ r)-config.rho**2
    if c <= 0:
        return 0.0 if b/distance < -config.physics.normal_tolerance else None
    discriminant = b*b-a*c
    if discriminant <= 0:
        return None  # Exact tangency transfers no impulse.
    time = c/(-b+np.sqrt(discriminant))
    if time < -config.physics.time_tolerance or time > duration+config.physics.time_tolerance:
        return None
    time = float(np.clip(time, 0., duration))
    normal_speed = float((r+time*u) @ u)/config.rho
    return time if normal_speed < -config.physics.normal_tolerance else None


@dataclass
class PopulationStep:
    state: PopulationState
    values: list[float]
    diagnostics: dict
    events: list[dict]
    crossings: list[dict]


class PopulationGravity:
    def __init__(self, config: PopulationConfig | None = None):
        self.config = config or PopulationConfig()
        self.gravity = GravityContact(self.config.physics)
        self.pair_tests = 0

    def energy(self, state):
        return float(state.kinetic+self.config.physics.gravity*sum(p.z for p in state.particles))

    def _pair_test(self, left, right, duration):
        if self.config.max_pair_tests is not None and self.pair_tests >= self.config.max_pair_tests:
            raise EvaluationBudgetExceeded("particle-pair prediction budget exhausted")
        self.pair_tests += 1
        return pair_contact_time(left, right, duration, self.config)

    def _validate(self, oracle, particles, *, initial=False):
        cfg = self.config.physics
        values = [oracle.value(p.x) for p in particles]
        gaps = [p.z-cfg.height_scale*f for p, f in zip(particles, values)]
        if min(gaps) < -cfg.gap_tolerance:
            raise ContactResolutionError("A population particle penetrates the terrain.")
        if self.config.collisions or initial:
            for i, p in enumerate(particles):
                for q in particles[i+1:]:
                    separation = np.linalg.norm(np.r_[p.x-q.x, p.z-q.z])
                    if separation < self.config.rho-self.config.pair_tolerance:
                        raise ContactResolutionError("Population particles overlap.")
        return values, float(min(gaps))

    def _advance(self, particles, duration, offset, boundary, crossings):
        if boundary is not None:
            for i, p in enumerate(particles):
                if p.x[0] >= boundary and p.v[0] < 0:
                    crossing = (boundary-p.x[0])/p.v[0]
                    if 0 <= crossing <= duration:
                        crossings.append({"particle": i, "time": float(offset+crossing)})
        return [self.gravity._flight(p, duration) for p in particles]

    def step(self, oracle, state, *, basin_boundary=None):
        cfg = self.config.physics
        particles = [State(p.x, cfg.beta*p.v, p.z, cfg.beta*p.vz) for p in state.particles]
        damped_state = PopulationState(tuple(particles))
        damping_energy = damped_state.kinetic-state.kinetic
        damping_impulse = damped_state.momentum-state.momentum
        elapsed, iterations = 0.0, 0
        events, crossings = [], []
        wall_impulse = np.zeros_like(state.momentum)
        impulse_energy = 0.0
        minimum_pair_gap = np.inf
        while cfg.h-elapsed > cfg.time_tolerance:
            iterations += 1
            if iterations > self.config.max_events_per_step:
                raise ContactResolutionError("Population event limit exceeded.")
            _, _ = self._validate(oracle, particles)
            remaining = cfg.h-elapsed
            candidates = []
            for i, p in enumerate(particles):
                hit = ballistic_contact_time(oracle, p, remaining, cfg)
                if hit is not None:
                    candidates.append((hit, "wall", i, -1))
            if self.config.collisions:
                for i, p in enumerate(particles):
                    for j in range(i+1, len(particles)):
                        q = particles[j]
                        gap = float(np.linalg.norm(np.r_[p.x-q.x, p.z-q.z])-self.config.rho)
                        minimum_pair_gap = min(minimum_pair_gap, gap)
                        hit = self._pair_test(p, q, remaining)
                        if hit is not None:
                            candidates.append((hit, "pair", i, j))
            if not candidates:
                particles = self._advance(particles, remaining, elapsed, basin_boundary, crossings)
                elapsed = cfg.h
                break
            first = min(item[0] for item in candidates)
            simultaneous = [item for item in candidates if abs(item[0]-first) <= cfg.time_tolerance]
            participants = []
            for _, kind, i, j in simultaneous:
                participants.extend([i] if kind == "wall" else [i, j])
            if len(set(participants)) != len(participants):
                raise ContactResolutionError("Unsupported simultaneous contacts share a particle.")
            particles = self._advance(particles, first, elapsed, basin_boundary, crossings)
            elapsed += first
            for _, kind, i, j in simultaneous:
                p = particles[i]
                if kind == "wall":
                    value = oracle.value(p.x)
                    if abs(p.z-cfg.height_scale*value) > 10*cfg.gap_tolerance:
                        raise ContactResolutionError("Population wall root is off the terrain.")
                    normal = np.r_[-cfg.height_scale*oracle.gradient(p.x), 1.]
                    normal /= np.linalg.norm(normal)
                    before = p.velocity
                    if before @ normal >= -cfg.normal_tolerance:
                        raise ContactResolutionError("Population wall event is not incoming.")
                    after = specular_reflection(before, normal)
                    error = float(0.5*(after @ after-before @ before))
                    wall_impulse += after-before
                    particles[i] = State(p.x, after[:-1], p.z, after[-1])
                    event = {"time": elapsed, "kind": kind, "i": i, "j": None,
                             "normal": normal.tolist(), "point_i": np.r_[p.x,p.z].tolist(),
                             "before_i": before.tolist(), "after_i": after.tolist(),
                             "energy_error": error, "momentum_error": None,
                             "normal_speed_before": float(before @ normal)}
                else:
                    q = particles[j]
                    difference = np.r_[p.x-q.x, p.z-q.z]
                    distance = np.linalg.norm(difference)
                    if abs(distance-self.config.rho) > 10*self.config.pair_tolerance:
                        raise ContactResolutionError("Binary event is not at the contact distance.")
                    normal = difference/distance
                    before_i, before_j = p.velocity, q.velocity
                    relative_speed = float((before_i-before_j) @ normal)
                    if relative_speed >= -cfg.normal_tolerance:
                        raise ContactResolutionError("Binary event is not approaching.")
                    after_i, after_j = elastic_pair_collision(before_i, before_j, normal)
                    # A binary map cannot silently replace a coupled wall impulse.
                    for particle, after in [(p, after_i), (q, after_j)]:
                        value = oracle.value(particle.x)
                        if abs(particle.z-cfg.height_scale*value) <= cfg.gap_tolerance:
                            wall_normal = np.r_[-cfg.height_scale*oracle.gradient(particle.x), 1.]
                            wall_normal /= np.linalg.norm(wall_normal)
                            if after @ wall_normal < -cfg.normal_tolerance:
                                raise ContactResolutionError("Binary impact requires a simultaneous terrain impulse.")
                    error = float(0.5*(after_i @ after_i+after_j @ after_j
                                       -before_i @ before_i-before_j @ before_j))
                    momentum_error = float(np.linalg.norm(after_i+after_j-before_i-before_j))
                    particles[i] = State(p.x, after_i[:-1], p.z, after_i[-1])
                    particles[j] = State(q.x, after_j[:-1], q.z, after_j[-1])
                    minimum_pair_gap = min(minimum_pair_gap, float(distance-self.config.rho))
                    event = {"time": elapsed, "kind": kind, "i": i, "j": j,
                             "normal": normal.tolist(), "point_i": np.r_[p.x,p.z].tolist(),
                             "point_j": np.r_[q.x,q.z].tolist(),
                             "before_i": before_i.tolist(), "before_j": before_j.tolist(),
                             "after_i": after_i.tolist(), "after_j": after_j.tolist(),
                             "energy_error": error, "momentum_error": momentum_error,
                             "normal_speed_before": relative_speed,
                             "contact_distance": float(distance),
                             "horizontal_kinetic_gain_i": float(0.5*(after_i[:-1] @ after_i[:-1]-before_i[:-1] @ before_i[:-1]))}
                events.append(event)
                impulse_energy += error
        final = PopulationState(tuple(particles))
        values, minimum_gap = self._validate(oracle, particles)
        gravity_impulse = np.zeros_like(state.momentum)
        gravity_impulse[-1] = -len(particles)*cfg.gravity*cfg.h
        momentum_residual = final.momentum-state.momentum-damping_impulse-gravity_impulse-wall_impulse
        return PopulationStep(final, values, {
            "energy": self.energy(final), "kinetic": final.kinetic,
            "potential": cfg.gravity*sum(p.z for p in particles),
            "damping_energy": damping_energy, "impulse_energy": impulse_energy,
            "energy_residual": self.energy(final)-self.energy(state)-damping_energy-impulse_energy,
            "momentum_residual": float(np.linalg.norm(momentum_residual)),
            "momentum": final.momentum.tolist(), "wall_impulse": wall_impulse.tolist(),
            "gravity_impulse": gravity_impulse.tolist(), "damping_impulse": damping_impulse.tolist(),
            "minimum_surface_gap": minimum_gap,
            "minimum_pair_gap": None if not np.isfinite(minimum_pair_gap) else float(minimum_pair_gap),
            "pair_collisions": sum(e["kind"] == "pair" for e in events),
            "wall_collisions": sum(e["kind"] == "wall" for e in events),
            "pair_tests": self.pair_tests,
        }, events, crossings)


@dataclass
class PopulationRun:
    rows: list[dict]
    events: list[dict]
    final_state: PopulationState
    best_value: float
    best_x: np.ndarray
    stop_reason: str
    counts: dict
    wall_seconds: float
    first_crossing_times: list[float | None]
    first_target_times: list[float | None]


def run_population(method, oracle, initial, steps, *, basin_boundary=None, target_value=None,
                   store_states=True):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    started = perf_counter()
    state = initial
    values, gap = method._validate(oracle, state.particles, initial=True)
    best_i = int(np.argmin(values))
    best_value, best_x = values[best_i], state.particles[best_i].x.copy()
    count = len(state.particles)
    crossing_times = [0. if basin_boundary is not None and p.x[0] < basin_boundary else None for p in state.particles]
    target_times = [0. if target_value is not None and f <= target_value
                    and (basin_boundary is None or p.x[0] < basin_boundary) else None
                    for p, f in zip(state.particles, values)]
    def positions(population):
        return [np.r_[p.x, p.z].tolist() for p in population.particles]
    rows = [{"step": 0, "time": 0., "best_value": best_value, "minimum_value": min(values),
             "values": values, "energy": method.energy(state), "kinetic": state.kinetic,
             "potential": method.energy(state)-state.kinetic,
             "cumulative_damping": 0., "cumulative_impulse_error": 0., "cumulative_residual": 0.,
             "momentum_residual": 0., "momentum": state.momentum.tolist(),
             "minimum_surface_gap": gap, "pair_collisions": 0, "wall_collisions": 0,
             "pair_tests": method.pair_tests, "crossed_particles": sum(t is not None for t in crossing_times),
             "target_particles": sum(t is not None for t in target_times),
             **({"positions": positions(state), "velocities": [p.velocity.tolist() for p in state.particles]} if store_states else {}),
             **oracle.counts()}]
    events, damping, impulse, residual = [], 0., 0., 0.
    reason = "completed"
    for index in range(1, steps+1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = method.step(oracle, state, basin_boundary=basin_boundary)
            numeric = [v for v in result.diagnostics.values() if isinstance(v, (int,float,np.number))]
            if not np.all(np.isfinite(numeric)):
                raise FloatingPointError("Nonfinite population diagnostics.")
        except EvaluationBudgetExceeded as exc:
            reason = f"budget: {exc}"
            break
        except ContactResolutionError as exc:
            reason = f"contact_error: {exc}"
            break
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            reason = f"numerical_error: {exc}"
            break
        state, values = result.state, result.values
        best_i = int(np.argmin(values))
        if values[best_i] < best_value:
            best_value, best_x = values[best_i], state.particles[best_i].x.copy()
        offset = (index-1)*method.config.physics.h
        for crossing in result.crossings:
            i = crossing["particle"]
            if crossing_times[i] is None:
                crossing_times[i] = offset+crossing["time"]
        for i, (p, f) in enumerate(zip(state.particles, values)):
            if target_times[i] is None and target_value is not None and f <= target_value and (basin_boundary is None or p.x[0] < basin_boundary):
                target_times[i] = index*method.config.physics.h
        damping += result.diagnostics["damping_energy"]
        impulse += result.diagnostics["impulse_energy"]
        residual += result.diagnostics["energy_residual"]
        rows.append({"step": index, "time": index*method.config.physics.h,
                     "best_value": best_value, "minimum_value": min(values), "values": values,
                     **result.diagnostics, "cumulative_damping": damping,
                     "cumulative_impulse_error": impulse, "cumulative_residual": residual,
                     "crossed_particles": sum(t is not None for t in crossing_times),
                     "target_particles": sum(t is not None for t in target_times),
                     **({"positions": positions(state), "velocities": [p.velocity.tolist() for p in state.particles]} if store_states else {}),
                     **oracle.counts()})
        events.extend([{**event, "time": offset+event["time"], "step": index} for event in result.events])
    counts = {**oracle.counts(), "pair_tests": method.pair_tests}
    counts["total_oracle_calls"] = sum(oracle.counts().values())
    return PopulationRun(rows, events, state, best_value, best_x, reason, counts,
                         perf_counter()-started, crossing_times, target_times)
