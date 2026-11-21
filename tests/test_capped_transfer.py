import numpy as np
import pytest

from collision_optim.core import State
from collision_optim.population import PopulationState
from collision_optim.shared_prefix import capped_transfer_at_checkpoint, transfer_at_checkpoint


@pytest.mark.parametrize("center_shift", [0., 2.])
def test_capped_transfer_uses_minimum_feasible_speed_with_conservation(center_shift):
    velocities = np.array([[.01,0.], [1.,0.], [.3,.4], [-.5,-.2]])+center_shift
    state = PopulationState(tuple(State([i,0.],v,.1,.2*i) for i,v in enumerate(velocities)))
    capped = capped_transfer_at_checkpoint(state,0)
    original = transfer_at_checkpoint(state,0)
    assert np.linalg.norm(original.particles[0].v)>np.linalg.norm(state.particles[0].v) or center_shift>0.
    assert np.linalg.norm(capped.particles[0].v)<=np.linalg.norm(state.particles[0].v)
    center = velocities.mean(axis=0)
    radius = np.sqrt(.75*np.sum((velocities-center)**2))
    assert np.linalg.norm(capped.particles[0].v)==pytest.approx(max(0.,np.linalg.norm(center)-radius))
    assert capped.kinetic==pytest.approx(state.kinetic,abs=1e-13,rel=0)
    np.testing.assert_allclose(capped.momentum,state.momentum,atol=1e-13,rtol=0)
    for before,after in zip(state.particles,capped.particles):
        np.testing.assert_array_equal(after.x,before.x)
        assert (after.z,after.vz)==(before.z,before.vz)
