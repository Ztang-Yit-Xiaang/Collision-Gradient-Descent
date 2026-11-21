"""Budget-aware execution. Failed macro steps never become valid trajectory rows."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .core import State
from .hybrid import ContactResolutionError
from .objectives import CountedObjective, EvaluationBudgetExceeded


@dataclass
class RunResult:
    rows: list[dict]
    contacts: list[dict]
    best_x: np.ndarray
    best_value: float
    final_state: State
    stop_reason: str
    wall_seconds: float
    counts: dict[str, int]


def run(method, oracle: CountedObjective, initial: State, steps: int, *, store_states=True):
    """Run a trajectory; compact mode uses O(d + steps) retained memory.

    store_states=False omits full per-step state/contact vectors, while preserving
    costs, energies, norms, and scalar impulse diagnostics. The final and best
    states remain available. The numerical update is identical in both modes.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    started = perf_counter()
    value = oracle.value(initial.x)
    if hasattr(method, "potential_energy"):
        potential = method.potential_energy(initial, value)
    else:
        scale = getattr(method.config, "gradient_scale", None)
        if scale is None:
            scale = method.config.mixing
        potential = scale * value
        if initial.z is not None:
            potential += (1 - scale) * initial.z
    initial_energy = initial.kinetic + potential
    state, best_value, best_x = initial, value, initial.x.copy()
    rows = [{
        "step": 0, "time": 0.0,
        **({"x": state.x.tolist(), "v": state.v.tolist()} if store_states else {}),
        "x_norm": float(np.linalg.norm(state.x)), "v_norm": float(np.linalg.norm(state.v)),
        "z": state.z, "vz": state.vz, "value": value, "best_value": value,
        "kinetic": initial.kinetic, "potential": potential, "energy": initial_energy,
        "contacts": 0, "intervention": "none", "direction_cosine": None,
        "paired_change": None, "diagnostic_se": 0.0, "loss_threshold": None,
        "streak": 0, "damping_energy": 0.0, "intervention_energy": 0.0,
        "energy_residual": 0.0, "cumulative_damping": 0.0,
        "cumulative_intervention": 0.0, "cumulative_residual": 0.0,
        "elapsed_seconds": perf_counter() - started, **oracle.counts(),
    }]
    contacts, damping, interventions, residual = [], 0.0, 0.0, 0.0
    reason = "completed"
    for index in range(1, steps + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                outcome = method.step(oracle, state)
            numeric = [outcome.value] + [v for v in outcome.diagnostics.values()
                                         if isinstance(v, (int, float, np.number))]
            if not np.all(np.isfinite(numeric)):
                raise FloatingPointError("Nonfinite step diagnostics; state was not committed.")
        except EvaluationBudgetExceeded as exc:
            reason = f"budget: {exc}"
            break
        except ContactResolutionError as exc:
            reason = f"contact_error: {exc}"
            break
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            reason = f"numerical_error: {exc}"
            break
        state, value = outcome.state, outcome.value
        if value < best_value:
            best_value, best_x = value, state.x.copy()
        diag = outcome.diagnostics
        damping += diag["damping_energy"]
        interventions += diag["intervention_energy"]
        residual += diag["energy_residual"]
        rows.append({
            "step": index, "time": index * method.config.h,
            **({"x": state.x.tolist(), "v": state.v.tolist()} if store_states else {}),
            "x_norm": float(np.linalg.norm(state.x)), "v_norm": float(np.linalg.norm(state.v)),
            "z": state.z, "vz": state.vz,
            "value": value, "best_value": best_value, **diag,
            "cumulative_damping": damping, "cumulative_intervention": interventions,
            "cumulative_residual": residual,
            "elapsed_seconds": perf_counter() - started, **oracle.counts(),
        })
        for event in outcome.events:
            contacts.append({
                "step": index, "time": (index - 1) * method.config.h + event.time,
                **({"point": event.point.tolist(), "normal": event.normal.tolist(),
                    "before": event.before.tolist(), "after": event.after.tolist()}
                   if store_states else {
                       "normal_vertical": float(event.normal[-1]),
                       "horizontal_energy_transfer": float(0.5 * (
                           event.after[:-1] @ event.after[:-1] - event.before[:-1] @ event.before[:-1])),
                       "normal_speed_before": float(event.before @ event.normal),
                   }),
                "energy_error": event.energy_error,
            })
    return RunResult(rows, contacts, best_x, best_value, state, reason,
                     perf_counter() - started, oracle.counts())

