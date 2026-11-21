"""Query-free momentum control; deterministic mechanics is an explicit ledger.

The scalar router has no objective/model reference. Warnings are heuristics, not
certificates of overshoot or of a nonglobal minimum. Velocity has update units.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .optimizer_controls import OptimizerState, inject_optimizer, optimizer_step


MODES = ("none", "restart", "damp", "random", "adaptive")


@dataclass(frozen=True)
class RoutingConfig:
    window: int = 10
    patience: int = 2
    epsilon_gain: float = 1e-4
    epsilon_move: float = 1e-3
    noise_aware: bool = False
    adverse_cosine: float = .1
    speed_ratio: float = 2.
    damping_factor: float = .5
    damping_cooldown: int = 10
    max_dampings: int = 20
    radii: tuple[float, float] = (.05, .15)
    max_shots: int = 2

    def __post_init__(self):
        if self.window < 2 or self.patience < 1 or self.max_shots < 0 or self.max_dampings < 0:
            raise ValueError("Invalid routing counts.")
        scales = (*self.radii, self.epsilon_gain, self.epsilon_move, self.speed_ratio,
                  self.adverse_cosine, self.damping_factor)
        if not np.all(np.isfinite(scales)) or not 0 <= self.damping_factor < 1:
            raise ValueError("Invalid routing scales.")
        if len(self.radii) != 2 or not 0 < self.radii[0] <= self.radii[1]:
            raise ValueError("Two positive ordered pulse radii required.")
        if min(self.epsilon_gain, self.epsilon_move, self.speed_ratio) < 0 or self.damping_cooldown < 0:
            raise ValueError("Negative routing tolerance.")


@dataclass(frozen=True)
class Action:
    kind: str = "continue"
    factor: float = 1.
    radius_index: int = 0
    reverse: bool = False


class MomentumRouter:
    """Use only already observed scalars. decide() never evaluates a candidate."""

    def __init__(self, mode="adaptive", config=None):
        if mode not in MODES:
            raise ValueError("Unknown routing mode.")
        self.mode, self.config = mode, config or RoutingConfig()
        self.warning = deque(maxlen=3)
        self.losses, self.moves = [], []
        self.best = np.inf
        self.previous_score = None
        self.score = None
        self.threshold = self.config.epsilon_gain
        self.progress = None
        self.quiet = self.shots = self.dampings = 0
        self.next_damping = self.next_pulse = self.grace_until = 0
        self.feedback = None
        self.last_failed = False
        self.events = []

    def decide(self, step, *, loss, movement, relative_velocity, cosine, speed_ratio):
        if self.mode == "none":
            return Action()
        c = self.config
        scalars = (loss, movement, relative_velocity, cosine, speed_ratio)
        if not np.all(np.isfinite(scalars)) or min(movement, relative_velocity, speed_ratio) < 0:
            raise ValueError("Nonfinite or negative routing input.")
        self.best = min(self.best, loss)
        self.losses.append(float(loss)); self.moves.append(float(movement))
        boundary = len(self.losses) == c.window
        if boundary:
            mean = float(np.mean(self.losses))
            variance = float(np.var(self.losses, ddof=1)) / c.window
            score = mean if c.noise_aware else self.best
            self.score = score
            if self.previous_score is not None:
                old, old_variance = self.previous_score
                self.threshold = c.epsilon_gain + (2*np.sqrt(variance+old_variance)/(1+abs(old)) if c.noise_aware else 0.)
                self.progress = (old-score)/(1+abs(old))
                low = float(np.mean(self.moves)) <= c.epsilon_move
                self.quiet = self.quiet+1 if low and self.progress <= self.threshold else 0
            self.previous_score = (score, variance)
            self.losses.clear(); self.moves.clear()
        if self.feedback is not None and step >= self.feedback["due"]:
            reference = self.feedback["reference"]
            gain = (reference-self.score)/(1+abs(reference))
            self.last_failed = gain <= self.threshold
            self.events.append({"kind": "feedback", "step": step, "observed_gain": gain,
                                "failed": self.last_failed, "causal_counterfactual": False})
            self.feedback = None
        warning = cosine > c.adverse_cosine and speed_ratio > c.speed_ratio and relative_velocity > c.epsilon_move
        self.warning.append(bool(warning))
        if (self.mode in {"restart", "damp", "adaptive"} and sum(self.warning) >= 2
                and step >= max(self.next_damping, self.grace_until) and self.dampings < c.max_dampings):
            self.dampings += 1
            self.next_damping = step+c.damping_cooldown
            self.warning.clear()
            return Action("restart" if self.mode == "restart" else "damp",
                          0. if self.mode == "restart" else c.damping_factor)
        if (boundary and self.mode in {"random", "adaptive"} and self.quiet >= c.patience
                and step >= self.next_pulse and self.feedback is None and self.shots < c.max_shots):
            index = 1 if self.mode == "random" or self.last_failed else 0
            reverse = self.mode == "adaptive" and self.shots > 0 and self.last_failed
            self.shots += 1; self.quiet = 0
            self.next_pulse = step+4*c.window
            self.grace_until = step+2*c.window
            self.warning.clear()
            self.feedback = {"due": step+2*c.window, "reference": self.score}
            return Action("pulse", radius_index=index, reverse=reverse)
        return Action()


def scale_momentum(state, factor):
    """Scale all inherited velocity components, preserving x and step number."""
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("Dissipation factor must lie in [0,1].")
    return OptimizerState(state.x, factor*state.first, state.second,
                          factor*state.external, factor*state.base_step, state.step)


def velocity_features(x, velocity, gradient, lr):
    vn, gn = float(np.linalg.norm(velocity)), float(np.linalg.norm(gradient))
    cosine = float(velocity @ gradient)/(vn*gn) if vn*gn > 1e-300 else 0.
    return {"relative_velocity": vn/(1+float(np.linalg.norm(x))), "cosine": cosine,
            "speed_ratio": vn/max(lr*gn, 1e-300)}


def run_routed(oracle, x0, optimizer, *, mode="adaptive", routing=None, seed=0, max_steps=10000):
    """Every loss/gradient is paid. No planning query, including diagnostics."""
    c = routing or RoutingConfig()
    if mode != "none" and optimizer.kind != "momentum":
        raise ValueError("Routing acts on Momentum SGD only.")
    if optimizer.kind == "momentum" and (optimizer.weight_decay != 0 or optimizer.control_decay != optimizer.beta1):
        raise ValueError("The deterministic ledger requires zero decay and matched momentum components.")
    router = MomentumRouter(mode, c)
    state = OptimizerState.initialize(x0)
    rng = np.random.default_rng(20261106+seed)
    radii = np.array(c.radii)*(1+np.linalg.norm(state.x))
    supply_cap = c.max_shots*.5*radii[-1]**2
    started = perf_counter(); feature_seconds = action_seconds = 0.
    value = float(oracle.value(state.x)); initial_value = best = value
    rows = []; actions = []; direction = None; movement = supplied = dissipated = 0.
    balance_energy = max_energy_error = max_momentum_error = 0.
    balance_momentum = np.zeros_like(state.x)
    natural_damping = gradient_work = residual_total = 0.
    reason = "steps"

    def record():
        rows.append({"step": state.step, "value": value, "best_value": best, "x": state.x.tolist(),
                     "relative_movement": movement, "kinetic": float(.5*state.velocity @ state.velocity),
                     "shots": router.shots, "dampings": router.dampings,
                     "supplied": supplied, "dissipated": dissipated,
                     "natural_damping": natural_damping, "gradient_work": gradient_work,
                     "integration_residual": residual_total,
                     "total_oracle_calls": sum(oracle.counts().values())})
    record()
    for _ in range(max_steps):
        remaining = None if oracle.total_budget is None else oracle.total_budget-sum(oracle.counts().values())
        if remaining is not None and remaining < 2:
            # A last paid gradient cannot be followed by a paid value observation.
            if remaining == 1:
                oracle.gradient(state.x)
            reason = "budget"; break
        try:
            g = oracle.gradient(state.x)
            old_value = value
            original_v = state.velocity.copy()
            stamp = perf_counter()
            features = velocity_features(state.x, original_v, g, optimizer.lr) if mode != "none" else {}
            action = router.decide(state.step, loss=value, movement=movement, **features) if mode != "none" else Action()
            feature_seconds += perf_counter()-stamp
            if action.kind != "continue":
                stamp = perf_counter()
                if action.kind == "pulse":
                    if action.reverse:
                        direction = -direction
                    else:
                        direction = rng.normal(size=state.x.size); direction /= np.linalg.norm(direction)
                    energy = float(.5*radii[action.radius_index]**2)
                    if supplied+energy > supply_cap+1e-12:
                        raise FloatingPointError("Supply cap exceeded.")
                    changed, impulse = inject_optimizer(state, energy, direction)
                else:
                    changed = scale_momentum(state, action.factor)
                    impulse = changed.velocity-state.velocity
                delta_k = float(.5*(changed.velocity @ changed.velocity-original_v @ original_v))
                if action.kind == "pulse": supplied += delta_k
                else: dissipated -= delta_k
                actions.append({"step": state.step, **asdict(action), "delta_kinetic": delta_k,
                                "impulse": impulse.tolist(), "direction": None if direction is None else direction.tolist()})
                state = changed; balance_energy += delta_k; balance_momentum += impulse
                action_seconds += perf_counter()-stamp
            old = state
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate = optimizer_step(old, g, optimizer)
                candidate_value = float(oracle.value(candidate.x))
            state = candidate; value = candidate_value; best = min(best, value)
            movement = float(np.linalg.norm(state.x-old.x)/(1+np.linalg.norm(old.x)))
            if optimizer.kind == "momentum":
                before_k = float(.5*old.velocity @ old.velocity)
                after_k = float(.5*state.velocity @ state.velocity)
                damp = (optimizer.beta1**2-1)*before_k
                work = -optimizer.lr*(value-old_value)
                residual = after_k-before_k-damp-work
                natural_damping += damp; gradient_work += work; residual_total += residual
                balance_energy += damp+residual
                balance_momentum += (optimizer.beta1-1)*old.velocity-optimizer.lr*g
                actual_h = after_k+optimizer.lr*(value-initial_value)
                max_energy_error = max(max_energy_error, abs(actual_h-balance_energy))
                max_momentum_error = max(max_momentum_error, float(np.linalg.norm(state.velocity-balance_momentum)))
            record()
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            reason = f"numerical_error: {exc}"; break
    summary = {"mode": mode, "stop_reason": reason, "failure": reason.startswith("numerical_error"),
               "initial_f": initial_value, "best_f": best, "final_f": value, "steps": state.step,
               "shots": router.shots, "dampings": router.dampings, "supplied": supplied,
               "supply_cap": float(supply_cap), "dissipated": dissipated,
               "max_energy_balance_error": max_energy_error if optimizer.kind == "momentum" else None,
               "max_momentum_balance_error": max_momentum_error if optimizer.kind == "momentum" else None,
               "feature_seconds": feature_seconds, "action_seconds": action_seconds,
               "wall_seconds": perf_counter()-started, "planning_calls": 0,
               "total_oracle_calls": sum(oracle.counts().values()), **oracle.counts()}
    return summary, rows, actions, router.events, state
