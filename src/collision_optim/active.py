"""Momentum SGD, matched-trigger restart, and active gradient reflection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .core import State, StepResult, specular_reflection
from .objectives import CountedObjective, Vector


@dataclass(frozen=True)
class ActiveConfig:
    h: float = 0.05
    beta: float = 0.99
    gradient_scale: float = 1.0
    epsilon_c: float = 0.1
    epsilon_f: float = 1e-3
    patience: int = 3
    adaptive: bool = False
    kappa: float = 2.0
    diagnostic_pairs: int = 4
    zero_norm: float = 1e-12

    def __post_init__(self):
        numeric = (self.h, self.beta, self.gradient_scale, self.epsilon_c,
                   self.epsilon_f, self.kappa, self.zero_norm)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Configuration must be finite.")
        if self.h <= 0 or not 0 <= self.beta <= 1 or self.gradient_scale <= 0:
            raise ValueError("Require h>0, beta in [0,1], and gradient_scale>0.")
        if not 0 <= self.epsilon_c < 1 or self.epsilon_f < 0 or self.kappa < 0 or self.zero_norm <= 0:
            raise ValueError("Invalid tolerance configuration.")
        if not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError("patience must be a positive integer.")
        if not isinstance(self.diagnostic_pairs, int) or self.diagnostic_pairs < 2:
            raise ValueError("diagnostic_pairs must be at least two.")


class MomentumSGD:
    """Velocity form; ordinary SGD learning rate is gradient_scale*h**2.

    mode='reflect' implements the active control strategy. mode='restart' uses
    the same trigger and recomputes the gradient-only velocity. No mode claims
    that the trigger proves an uphill step is unhelpful.
    """

    def __init__(
        self,
        config: ActiveConfig | None = None,
        mode: str = "none",
        *,
        seed: int = 0,
        paired_sampler: Callable[[Vector, Vector, np.random.Generator], tuple[float, float]] | None = None,
    ):
        if mode not in {"none", "reflect", "restart"}:
            raise ValueError("mode must be none, reflect, or restart")
        self.config = config or ActiveConfig()
        self.mode = mode
        self.streak = 0
        self.rng = np.random.default_rng(seed)
        self.paired_sampler = paired_sampler

    def step(self, oracle: CountedObjective, state: State) -> StepResult:
        if state.z is not None:
            raise ValueError("Momentum SGD expects a horizontal-only state.")
        cfg = self.config
        f0 = oracle.value(state.x)
        gradient = oracle.gradient(state.x)
        damped = cfg.beta * state.v
        trial_v = damped - cfg.h * cfg.gradient_scale * gradient
        trial_x = state.x + cfg.h * trial_v
        f_trial = oracle.value(trial_x)
        kinetic_trial = float(0.5 * trial_v @ trial_v)
        c = None
        g_norm, v_norm = np.linalg.norm(gradient), np.linalg.norm(trial_v)
        if g_norm > cfg.zero_norm and v_norm > cfg.zero_norm:
            c = float(np.clip((gradient / g_norm) @ (trial_v / v_norm), -1, 1))
        paired_change = (f_trial - f0) / (1 + abs(f0))
        se = 0.0
        if self.mode != "none" and self.paired_sampler is not None:
            count = cfg.diagnostic_pairs if cfg.adaptive else 1
            changes = []
            for _ in range(count):
                a, b = oracle.sampled_pair(state.x, trial_x, self.paired_sampler, self.rng)
                changes.append((b - a) / (1 + abs(a)))
            paired_change = float(np.mean(changes))
            if cfg.adaptive:
                se = float(np.std(changes, ddof=1) / np.sqrt(count))
        # A deterministic full-batch objective has no minibatch sampling error.
        threshold = cfg.epsilon_f + (cfg.kappa * se if cfg.adaptive else 0.0)
        adverse = c is not None and c > cfg.epsilon_c and paired_change > threshold
        streak = self.streak + 1 if adverse and self.mode != "none" else 0
        intervention = self.mode != "none" and streak >= cfg.patience
        final_v = trial_v
        if intervention:
            if self.mode == "reflect":
                final_v = specular_reflection(trial_v, gradient)
            else:
                final_v = -cfg.h * cfg.gradient_scale * gradient
            streak = 0
        final_x = state.x + cfg.h * final_v
        value = oracle.value(final_x) if intervention else f_trial
        next_state = State(final_x, final_v)
        damping_energy = float(0.5 * (damped @ damped) - state.kinetic)
        intervention_energy = next_state.kinetic - kinetic_trial
        before = state.kinetic + cfg.gradient_scale * f0
        after = next_state.kinetic + cfg.gradient_scale * value
        self.streak = streak  # Commit only after all oracle calls succeeded.
        return StepResult(next_state, value, {
            "gradient_norm": float(g_norm),
            "direction_cosine": c,
            "paired_change": paired_change,
            "diagnostic_se": se,
            "loss_threshold": threshold,
            "streak": streak,
            "intervention": self.mode if intervention else "none",
            "contacts": 0,
            "kinetic": next_state.kinetic,
            "potential": cfg.gradient_scale * value,
            "energy": after,
            "damping_energy": damping_energy,
            "intervention_energy": intervention_energy,
            "energy_residual": after - before - damping_energy - intervention_energy,
        })


class ActiveReflection(MomentumSGD):
    def __init__(self, config: ActiveConfig | None = None, **kwargs):
        super().__init__(config, mode="reflect", **kwargs)


class MomentumRestart(MomentumSGD):
    def __init__(self, config: ActiveConfig | None = None, **kwargs):
        super().__init__(config, mode="restart", **kwargs)

