"""Common state, impulse maps, and diagnostics. Mass is one throughout."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


def vector(value) -> Vector:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError("Expected a finite, nonempty, one-dimensional vector.")
    return result.copy()


@dataclass(frozen=True)
class State:
    x: Vector
    v: Vector
    z: float | None = None
    vz: float | None = None

    def __post_init__(self):
        x, v = vector(self.x), vector(self.v)
        if x.shape != v.shape:
            raise ValueError("Position and velocity dimensions must match.")
        if (self.z is None) != (self.vz is None):
            raise ValueError("Height and vertical velocity must be supplied together.")
        if self.z is not None and not np.all(np.isfinite([self.z, self.vz])):
            raise ValueError("Height state must be finite.")
        x.setflags(write=False)
        v.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "v", v)

    @property
    def velocity(self):
        return self.v.copy() if self.vz is None else np.r_[self.v, self.vz]

    @property
    def kinetic(self):
        w = self.velocity
        return float(0.5 * (w @ w))


def specular_reflection(velocity, normal):
    """An instantaneous fixed-wall reflection, not a momentum-conserving pair."""
    v, n = vector(velocity), vector(normal)
    if v.shape != n.shape:
        raise ValueError("Velocity and normal dimensions must match.")
    length = np.linalg.norm(n)
    if length <= 1e-15:
        raise ValueError("A reflection normal cannot be zero.")
    n /= length
    return v - 2 * (v @ n) * n


def elastic_pair_collision(velocity_i, velocity_j, normal_ij):
    """Equal unit masses: conserve pair momentum and kinetic energy.

    This is the impulse map only. A population simulation must separately locate
    approaching contacts; the single-particle experiments do not schedule pairs.
    """
    vi, vj, n = vector(velocity_i), vector(velocity_j), vector(normal_ij)
    if vi.shape != vj.shape or vi.shape != n.shape:
        raise ValueError("Pair collision dimensions must match.")
    length = np.linalg.norm(n)
    if length <= 1e-15:
        raise ValueError("A pair normal cannot be zero.")
    n /= length
    transfer = ((vi - vj) @ n) * n
    return vi - transfer, vj + transfer


@dataclass(frozen=True)
class ContactEvent:
    time: float
    point: Vector
    normal: Vector
    before: Vector
    after: Vector
    energy_error: float


@dataclass(frozen=True)
class StepResult:
    state: State
    value: float
    diagnostics: dict[str, float | int | str | None]
    events: tuple[ContactEvent, ...] = field(default_factory=tuple)

