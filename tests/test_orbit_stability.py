import numpy as np
import pytest
from collision_optim.core import State
from collision_optim.event_replay import pack
from collision_optim.objectives import Quadratic
from collision_optim.population import PopulationOracle,PopulationState
from collision_optim.orbit_stability import (bowl_launch,next_bounce,bowl_returns,rosen_basis,controlled_initial)
from collision_optim.stability_experiment import map_cell_task


def test_launches_share_exact_mechanical_energy():
    for x,vx in ((0.,0.),(1e-6,0.),(0.,-1e-6)):
        oracle=PopulationOracle(Quadratic(np.array([[4.]])))
        state,_=bowl_launch(oracle,4.,.05,x,vx)
        assert state.kinetic+.5*state.z==pytest.approx(.025,abs=1e-15)
        assert oracle.values==1


def test_launch_rejects_impossible_energy_surface():
    with pytest.raises(ValueError,match='energy'):
        bowl_launch(PopulationOracle(Quadratic()),1.,.01,x=1.)


def test_vertical_reference_and_external_impulse_balance():
    oracle=PopulationOracle(Quadratic(np.array([[4.]])))
    initial,_=bowl_launch(oracle,4.,.05)
    final,event=next_bounce(oracle,initial,.05)
    assert event['duration']==pytest.approx(2*np.sqrt(2*.05/.5))
    assert final.x[0]==0. and final.v[0]==0.
    assert final.vz==pytest.approx(initial.vz)
    assert abs(event['energy_residual'])<1e-14
    assert event['momentum_residual']<1e-14
    assert sum(oracle.counts().values())==5


def test_bowl_mirror_symmetry_after_multiple_impacts():
    a=bowl_returns(3.,.08,x=1e-7,bounces=5)
    b=bowl_returns(3.,.08,x=-1e-7,bounces=5)
    assert not a['failed'] and not b['failed']
    for x,y in zip(a['rows'],b['rows']):
        assert x['x']==pytest.approx(-y['x'],abs=1e-13)
        assert x['vx']==pytest.approx(-y['vx'],abs=1e-13)
        assert x['vz']==pytest.approx(y['vz'],abs=1e-13)
        assert x['time']==pytest.approx(y['time'],abs=1e-13)


def test_basis_setup_probe_counts_and_orientation():
    basis=rosen_basis(Quadratic(np.diag([1.,5.])),np.zeros(2))
    np.testing.assert_allclose(basis['eigenvalues'],[1.,5.])
    np.testing.assert_allclose(basis['high_direction'],[0.,1.])
    assert basis['setup_gradient_calls']==5 and basis['setup_value_calls']==1


def test_controlled_initial_preserves_spectators_and_shared_target():
    parent=pack(PopulationState(tuple(State([x,0.],[.1,0.],1.,0.) for x in (0.,1.,2.,3.))))
    basis={'optimum':[0.,0.],'high_direction':[0.,1.],'low_direction':[1.,0.]}
    obj=Quadratic()
    single=controlled_initial(parent,basis,.02,'high_plus','single',obj)
    joint=controlled_initial(parent,basis,.02,'high_plus','joint',obj)
    assert pack(single)['positions'][0]==pack(joint)['positions'][0]
    assert pack(single)['velocities'][0]==[0.,0.,0.]
    assert joint.particles[0].z-obj.value(joint.particles[0].x)==pytest.approx(.02)
    assert pack(joint)['positions'][1:]==parent['positions'][1:]
    assert pack(joint)['velocities'][1:]==parent['velocities'][1:]


def test_budget_failure_preserves_paid_uncommitted_probe():
    result=bowl_returns(1.,.05,x=1e-7,bounces=2,budget=6)
    assert result['stop_reason'].startswith('budget:') and not result['failed']
    assert len(result['events'])==1
    assert result['counts']['total_oracle_calls']==6
    assert result['rows'][-1]['total_oracle_calls']==5
    assert result['uncommitted']['attempted_bounce']==2


def test_cell_executes_one_zero_and_eight_signed_probes(tmp_path):
    (tmp_path/'return_traces').mkdir()
    result=map_cell_task(tmp_path,2.,.35)
    assert len(result['return_probes'])==9 and len(result['return_matrices'])==2
    assert sum(r['axis']=='zero' for r in result['return_probes'])==1
    assert result['runs']==[]
    assert result['solver_queries']==45
