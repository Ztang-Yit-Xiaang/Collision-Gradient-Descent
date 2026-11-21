import numpy as np

from collision_optim.core import State
from collision_optim.population import PopulationState
from collision_optim.shared_prefix import transfer_at_checkpoint


def test_checkpoint_transfer_conserves_energy_and_nonzero_momentum():
    before = PopulationState(tuple(
        State([i, -i], velocity, i + 2., .4 * i - .3)
        for i, velocity in enumerate(([1, 1], [0, 4], [3, 2], [2, 0]))
    ))
    after = transfer_at_checkpoint(before, 1)
    assert np.linalg.norm(before.momentum) > 1.
    np.testing.assert_allclose(after.momentum, before.momentum, rtol=0, atol=1e-13)
    assert abs(after.kinetic - before.kinetic) < 1e-13
    np.testing.assert_allclose(after.particles[1].v, before.momentum[:2] / 4)
    for a, b in zip(after.particles, before.particles):
        np.testing.assert_array_equal(a.x, b.x)
        assert (a.z, a.vz) == (b.z, b.vz)
