"""Differentiable objectives, evaluation budgets, and analytic drift geometry.

Path coefficients describe f(x + t * displacement), in increasing powers of t.
They let the synthetic contact tests find *all* crossings, including trajectories
whose two endpoints are feasible but which tunnel through the intervening wall.
This additional analytic access is counted separately and is not a black-box
contact detector suitable for a neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class Objective(Protocol):
    def value(self, x: Vector) -> float: ...
    def gradient(self, x: Vector) -> Vector: ...


class EvaluationBudgetExceeded(RuntimeError):
    """A real oracle call would exceed a configured budget."""


@dataclass(frozen=True)
class Budget:
    values: int | None = None
    gradients: int | None = None
    paths: int | None = None

    def __post_init__(self):
        for bound in (self.values, self.gradients, self.paths):
            if bound is not None and (not isinstance(bound, int) or bound < 0):
                raise ValueError("Evaluation budgets must be nonnegative integers.")


class CountedObjective:
    """Counts value/gradient/path requests, including diagnostics and root checks."""

    def __init__(self, objective: Objective, budget: Budget | None = None):
        self.objective = objective
        self.budget = budget or Budget()
        self.values = self.gradients = self.paths = 0

    def _reserve(self, kind: str, amount: int = 1):
        count = getattr(self, kind)
        limit = getattr(self.budget, kind)
        if limit is not None and count + amount > limit:
            raise EvaluationBudgetExceeded(f"{kind} budget exhausted ({count}/{limit})")
        setattr(self, kind, count + amount)

    def value(self, x: Vector) -> float:
        self._reserve("values")
        value = float(self.objective.value(x))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite objective value.")
        return value

    def gradient(self, x: Vector) -> Vector:
        self._reserve("gradients")
        g = np.asarray(self.objective.gradient(x), dtype=float)
        if g.shape != x.shape:
            raise ValueError("Gradient must match the parameter shape.")
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("Nonfinite gradient.")
        return g

    def path_coefficients(self, x: Vector, displacement: Vector) -> Vector:
        path = getattr(self.objective, "path_coefficients", None)
        if path is None:
            raise NotImplementedError(
                "Exact contact detection needs path_coefficients. This synthetic "
                "prototype does not silently replace them with endpoint-only tests."
            )
        self._reserve("paths")
        coefficients = np.asarray(path(x, displacement), dtype=float)
        if coefficients.ndim != 1 or not coefficients.size:
            raise ValueError("Invalid path polynomial.")
        if not np.all(np.isfinite(coefficients)):
            raise FloatingPointError("Nonfinite path polynomial.")
        return np.polynomial.polynomial.polytrim(coefficients)

    def sampled_pair(
        self,
        x: Vector,
        y: Vector,
        sampler: Callable[[Vector, Vector, np.random.Generator], tuple[float, float]],
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """One minibatch/randomness realization must be shared within each pair.

        sampler is responsible for evaluating both points on that same realization.
        It must not additionally route its two calls through this counted wrapper.
        """
        self._reserve("values", 2)
        a, b = map(float, sampler(x, y, rng))
        if not np.isfinite(a) or not np.isfinite(b):
            raise FloatingPointError("Nonfinite paired diagnostic values.")
        return a, b

    def counts(self) -> dict[str, int]:
        return {"value_calls": self.values, "gradient_calls": self.gradients, "path_calls": self.paths}


@dataclass(frozen=True)
class Affine:
    slope: Vector
    offset: float = 0.0

    def __post_init__(self):
        slope = np.asarray(self.slope, dtype=float)
        if slope.ndim != 1 or not slope.size or not np.all(np.isfinite(slope)):
            raise ValueError("slope must be a finite nonempty vector")
        if not np.isfinite(self.offset):
            raise ValueError("offset must be finite")
        object.__setattr__(self, "slope", slope.copy())

    def value(self, x):
        return float(self.slope @ x + self.offset)

    def gradient(self, x):
        return self.slope.copy()

    def path_coefficients(self, x, displacement):
        return np.array([self.value(x), self.slope @ displacement])


@dataclass(frozen=True)
class Quadratic:
    matrix: Vector | None = None

    def _matrix(self, x):
        matrix = np.eye(x.size) if self.matrix is None else np.asarray(self.matrix, dtype=float)
        if matrix.shape != (x.size, x.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("Quadratic matrix has an invalid shape or value.")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Quadratic matrix must be symmetric.")
        return matrix

    def value(self, x):
        if self.matrix is None:
            return float(0.5 * (x @ x))
        return float(0.5 * x @ self._matrix(x) @ x)

    def gradient(self, x):
        if self.matrix is None:
            return np.array(x, dtype=float, copy=True)
        return self._matrix(x) @ x

    def path_coefficients(self, x, displacement):
        if self.matrix is None:
            return np.array([self.value(x), x @ displacement, 0.5 * (displacement @ displacement)])
        matrix = self._matrix(x)
        return np.array([self.value(x), x @ matrix @ displacement, 0.5 * displacement @ matrix @ displacement])


@dataclass(frozen=True)
class TiltedDoubleWell:
    tilt: float = 0.05

    def value(self, x):
        return float(0.25 * (x[0] ** 2 - 1.0) ** 2 + self.tilt * x[0] + 0.5 * (x[1:] @ x[1:]))

    def gradient(self, x):
        g = np.array(x, dtype=float, copy=True)
        g[0] = x[0] ** 3 - x[0] + self.tilt
        return g

    def path_coefficients(self, x, displacement):
        first = Polynomial([x[0], displacement[0]])
        result = 0.25 * (first * first - 1) ** 2 + self.tilt * first
        for a, b in zip(x[1:], displacement[1:]):
            result += 0.5 * Polynomial([a, b]) ** 2
        return result.coef


@dataclass(frozen=True)
class Rosenbrock:
    def value(self, x):
        if x.size < 2:
            raise ValueError("Rosenbrock requires at least two parameters.")
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def gradient(self, x):
        if x.size < 2:
            raise ValueError("Rosenbrock requires at least two parameters.")
        g = np.zeros_like(x)
        g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) + 2 * (x[:-1] - 1)
        g[1:] += 200 * (x[1:] - x[:-1] ** 2)
        return g

    def path_coefficients(self, x, displacement):
        if x.size < 2:
            raise ValueError("Rosenbrock requires at least two parameters.")
        result = Polynomial([0.0])
        p = [Polynomial([a, b]) for a, b in zip(x, displacement)]
        for left, right in zip(p[:-1], p[1:]):
            result += 100 * (right - left * left) ** 2 + (1 - left) ** 2
        return result.coef

