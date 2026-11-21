"""E23 terrain/motion adapters; E22 initialization policies are unchanged."""
import numpy as np
from .curvature_energy import RotatedEnergyWells, explorer_velocities, score

SCALES = (.5, 1., 2.)
INCUMBENT_ENERGIES = (0., 1e-5, 1e-3)


class ScaledEnergyWells(RotatedEnergyWells):
    def __init__(self, scale, seed):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Positive longitudinal scale required.')
        super().__init__(8., seed)
        self.scale = float(scale)

    def value(self, x):
        r = self.coordinates(x)
        return self.scale*self.base.value(r) + .5*(self.stiffness-self.scale)*r[1]**2

    def gradient(self, x):
        r = self.coordinates(x)
        g = self.scale*self.base.gradient(r)
        g[1] += (self.stiffness-self.scale)*r[1]
        return self.rotation.T@g

    def path_coefficients(self, x, displacement):
        r, s = self.coordinates(x), self.coordinates(displacement)
        c = self.scale*self.base.path_coefficients(r, s)
        c = np.pad(c, (0, max(0, 3-len(c))))
        c[:3] += (self.stiffness-self.scale)*np.array([.5*r[1]**2, r[1]*s[1], .5*s[1]**2])
        return c


def controlled_velocities(seed, incumbent_energy):
    e = float(incumbent_energy)
    if not np.isfinite(e) or e < 0 or e >= .45:
        raise ValueError('Incumbent energy must be in [0,.45).')
    velocities = explorer_velocities(seed)
    direction = np.random.default_rng(np.random.SeedSequence([20261110, seed, 23])).normal(size=2)
    direction /= np.linalg.norm(direction)
    u = np.sqrt(2*e)*direction
    velocities[0] = u
    velocities[1:] = np.sqrt(1-4*e/(3*.6))*velocities[1:] - u/3
    return velocities


def generalization_score(rows, context, objective):
    result = score(rows, context, objective)
    last = rows[-1]
    result['particle0_final_hit'] = bool(last['values'][0] <= context['minimum']+1e-4
        and objective.coordinates(np.array(last['positions'][0][:-1]))[0] < context['boundary'])
    return result
