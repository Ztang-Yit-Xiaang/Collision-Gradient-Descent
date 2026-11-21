"""Matrix-free, dimension-scaled objectives with genuine all-coordinate activity.

The nonlinear cases use r=sqrt(d)*Qx and average their local energies. This keeps
objective magnitude, local curvature, gradient norms, and total initial kinetic
energy comparable across dimensions. Q is an implicit orthogonal map, not a
dense matrix; neighbor interactions create genuine nonseparability independently
of that rotation. All drift polynomials are vectorized and have degree <=4.
"""

from __future__ import annotations

import numpy as np


class StructuredRotation:
    def __init__(self, dimension, *, seed=5239, reflections=3):
        if not isinstance(dimension, int) or dimension < 2:
            raise ValueError("dimension must be an integer >=2")
        if not isinstance(reflections, int) or reflections < 0:
            raise ValueError("reflections must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        self.dimension = dimension
        self.vectors = []
        for _ in range(reflections):
            v = rng.normal(size=dimension)
            self.vectors.append(v / np.linalg.norm(v))

    def forward(self, x):
        result = np.array(x, dtype=float, copy=True)
        if result.shape != (self.dimension,):
            raise ValueError("Invalid parameter dimension")
        for v in self.vectors:
            result -= 2 * float(np.sum(v * result)) * v
        return result

    def inverse(self, x):
        result = np.array(x, dtype=float, copy=True)
        if result.shape != (self.dimension,):
            raise ValueError("Invalid parameter dimension")
        for v in reversed(self.vectors):
            result -= 2 * float(np.sum(v * result)) * v
        return result


class DiagonalEllipsoid:
    """Rotated ill-conditioned quadratic; no d-by-d allocation."""
    def __init__(self, dimension, *, condition=100.0, seed=5239):
        if not np.isfinite(condition) or condition < 1:
            raise ValueError("condition must be finite and >=1")
        self.rotation = StructuredRotation(dimension, seed=seed)
        self.dimension = dimension
        self.eigenvalues = np.geomspace(1 / condition, 1.0, dimension)
        self.minimum = 0.0

    def coordinates(self, x):
        return np.sqrt(self.dimension) * self.rotation.forward(x)

    def from_coordinates(self, r):
        return self.rotation.inverse(np.asarray(r) / np.sqrt(self.dimension))

    def value(self, x):
        y = self.rotation.forward(x)
        return float(0.5 * np.sum(self.eigenvalues * y * y))

    def gradient(self, x):
        return self.rotation.inverse(self.eigenvalues * self.rotation.forward(x))

    def path_coefficients(self, x, displacement):
        y, s = self.rotation.forward(x), self.rotation.forward(displacement)
        return np.array([0.5 * np.sum(self.eigenvalues * y*y),
                         np.sum(self.eigenvalues * y*s),
                         0.5 * np.sum(self.eigenvalues * s*s)])


class CoupledDoubleWell:
    """Every coordinate is a well, with cyclic neighbor coupling."""
    def __init__(self, dimension, *, coupling=0.2, tilt=0.05, seed=5239):
        if not np.isfinite(coupling) or coupling < 0 or not 0 <= tilt < 2 / np.sqrt(27):
            raise ValueError("Invalid well coupling or tilt")
        self.rotation = StructuredRotation(dimension, seed=seed)
        self.dimension, self.coupling, self.tilt = dimension, coupling, tilt
        roots = np.roots([1.0, 0.0, -1.0, tilt])
        stationary = roots[np.abs(roots.imag) < 1e-10].real
        self.minimum = float(min(0.25 * (a*a - 1)**2 + tilt*a for a in stationary))

    def coordinates(self, x):
        return np.sqrt(self.dimension) * self.rotation.forward(x)

    def from_coordinates(self, r):
        return self.rotation.inverse(np.asarray(r) / np.sqrt(self.dimension))

    def value(self, x):
        r = self.coordinates(x)
        difference = r - np.roll(r, -1)
        return float(np.mean(0.25 * (r*r - 1)**2 + self.tilt*r
                             + 0.5*self.coupling*difference*difference))

    def gradient(self, x):
        r = self.coordinates(x)
        local = (r*r*r - r + self.tilt
                 + self.coupling * (2*r - np.roll(r, 1) - np.roll(r, -1)))
        return self.rotation.inverse(local / np.sqrt(self.dimension))

    def path_coefficients(self, x, displacement):
        r, s = self.coordinates(x), self.coordinates(displacement)
        a, b = r - np.roll(r, -1), s - np.roll(s, -1)
        return np.array([
            np.mean(0.25*(r*r-1)**2 + self.tilt*r + 0.5*self.coupling*a*a),
            np.mean((r*r*r-r+self.tilt)*s + self.coupling*a*b),
            np.mean((1.5*r*r-0.5)*s*s + 0.5*self.coupling*b*b),
            np.mean(r*s*s*s),
            0.25*np.mean(s*s*s*s),
        ])


class CoupledRosenbrock:
    """Curved chain valley, normalized to keep curvature scales comparable."""
    def __init__(self, dimension, *, seed=5239):
        self.rotation = StructuredRotation(dimension, seed=seed)
        self.dimension = dimension
        self.minimum = 0.0

    def coordinates(self, x):
        return np.sqrt(self.dimension) * self.rotation.forward(x)

    def from_coordinates(self, r):
        return self.rotation.inverse(np.asarray(r) / np.sqrt(self.dimension))

    def value(self, x):
        r = self.coordinates(x)
        return float(np.mean((r[1:] - r[:-1]**2)**2 + 0.01*(1-r[:-1])**2))

    def gradient(self, x):
        r = self.coordinates(x)
        g = np.zeros_like(r)
        delta = r[1:] - r[:-1]*r[:-1]
        g[:-1] += -4*r[:-1]*delta + 0.02*(r[:-1]-1)
        g[1:] += 2*delta
        return self.rotation.inverse(g*np.sqrt(self.dimension)/(self.dimension-1))

    def path_coefficients(self, x, displacement):
        r, s = self.coordinates(x), self.coordinates(displacement)
        a = r[1:] - r[:-1]*r[:-1]
        b = s[1:] - 2*r[:-1]*s[:-1]
        c = -s[:-1]*s[:-1]
        return np.array([
            np.mean(a*a + 0.01*(1-r[:-1])**2),
            np.mean(2*a*b - 0.02*(1-r[:-1])*s[:-1]),
            np.mean(b*b + 2*a*c + 0.01*s[:-1]*s[:-1]),
            np.mean(2*b*c),
            np.mean(c*c),
        ])

