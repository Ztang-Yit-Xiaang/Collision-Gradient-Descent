"""Frozen population scheduler with a swappable isolated pair-impact law.

The step body mirrors population.PopulationGravity.step; only pair map and event
metrics differ. Historical source is preserved and equivalence is regression-tested.
Random scattering is an ablation preserving linear P/K, not angular momentum.
"""

from __future__ import annotations

import numpy as np
from .core import State, elastic_pair_collision, specular_reflection, vector
from .gravity import ballistic_contact_time
from .hybrid import ContactResolutionError
from .population import PopulationGravity, PopulationState, PopulationStep


ORIGINAL_SOURCE_SHA256 = "2a531fa2f76c224003667743595c87c091e723ceb84a3bc54a05463c07ce75f0"


def random_pair_collision(velocity_i, velocity_j, normal, rng):
    vi, vj, n = vector(velocity_i), vector(velocity_j), vector(normal)
    if vi.shape != vj.shape or vi.shape != n.shape or np.linalg.norm(n) == 0:
        raise ValueError("Pair velocity/normal dimensions or normal are invalid.")
    n /= np.linalg.norm(n)
    if (vi-vj) @ n >= 0:
        raise ValueError("Random scattering requires an incoming pair.")
    center, relative = (vi+vj)/2, (vi-vj)/2
    d = rng.normal(size=vi.size)
    length = np.linalg.norm(d)
    if length == 0:
        raise FloatingPointError("Zero random scattering direction.")
    d /= length
    if d @ n < 0:
        d = -d
    outgoing = np.linalg.norm(relative)*d
    return center+outgoing, center-outgoing


def pair_diagnostics(vi, vj, ai, aj, n, distance=1.):
    center, u, after_u = (vi+vj)/2, (vi-vj)/2, (ai-aj)/2
    impulse = ai-vi; r = distance*n
    angular = np.outer(r, impulse)-np.outer(impulse, r)
    gain = float(.5*(ai @ ai-vi @ vi))
    predicted = float(center @ (after_u-u))
    return {"center_speed": float(np.linalg.norm(center)),
            "relative_half_speed": float(np.linalg.norm(u)),
            "incidence_cosine": float(-(u @ n)/np.linalg.norm(u)),
            "outgoing_normal_speed": float((ai-aj) @ n),
            "kinetic_gain_i": gain, "predicted_kinetic_gain_i": predicted,
            "transfer_formula_error": abs(gain-predicted),
            "angular_impulse_frobenius": float(np.linalg.norm(angular)),
            "impulse_i": impulse.tolist()}


class ScatteringPopulation(PopulationGravity):
    def __init__(self, config=None, *, rule="elastic", seed=0):
        super().__init__(config)
        if rule not in {"elastic", "random"}:
            raise ValueError("Unknown pair-scattering rule.")
        self.rule = rule
        self.rng = np.random.default_rng(seed)
        self.map_attempts = []

    def _pair_map(self, vi, vj, normal):
        if self.rule == "elastic":
            ai, aj = elastic_pair_collision(vi, vj, normal)
        else:
            ai, aj = random_pair_collision(vi, vj, normal, self.rng)
        self.map_attempts.append({"before_i": vi.tolist(), "before_j": vj.tolist(),
                                  "after_i": ai.tolist(), "after_j": aj.tolist(),
                                  "normal": normal.tolist()})
        return ai, aj

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
                    after_i, after_j = self._pair_map(before_i, before_j, normal)
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
                if kind == "pair":
                    event.update(pair_diagnostics(before_i, before_j, after_i, after_j, normal, distance))
                    event["scattering_rule"] = self.rule
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
