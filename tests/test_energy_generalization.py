import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval
from collision_optim.curvature_energy import (
    RotatedEnergyWells, explorer_velocities, parameter_cloud, measured_curvatures, allocate, initial_state)
from collision_optim.energy_generalization import ScaledEnergyWells, controlled_velocities, generalization_score
from collision_optim.energy_generalization_experiment import task
from collision_optim.population import PopulationOracle


def test_scale_one_matches_frozen_parent_exactly():
    parent=RotatedEnergyWells(8.,1480);new=ScaledEnergyWells(1.,1480)
    x=np.array([.3,-.2]);d=np.array([.1,.4])
    assert new.value(x)==parent.value(x)
    np.testing.assert_array_equal(new.gradient(x),parent.gradient(x))
    np.testing.assert_array_equal(new.path_coefficients(x,d),parent.path_coefficients(x,d))


def test_scaled_geometry_gradient_and_full_path():
    for a in (.5,2.):
        obj=ScaledEnergyWells(a,1481);x=np.array([.3,-.2]);d=np.array([.4,.1]);eps=1e-6
        fd=[(obj.value(x+eps*e)-obj.value(x-eps*e))/(2*eps) for e in np.eye(2)]
        np.testing.assert_allclose(obj.gradient(x),fd,rtol=1e-8,atol=1e-9)
        for t in (-1.,0.,.3,1.):assert polyval(t,obj.path_coefficients(x,d))==pytest.approx(obj.value(x+t*d),abs=1e-12)
    with pytest.raises(ValueError):ScaledEnergyWells(0,1481)


def test_cloud_is_shared_and_barrier_really_changes():
    clouds=[];barriers=[]
    for a in (.5,1.,2.):
        obj=ScaledEnergyWells(a,1482);xs,roots=parameter_cloud(obj,'good_present',1482)
        clouds.append(xs);barriers.append(obj.value(obj.from_coordinates([roots[1],0.]))-obj.value(obj.from_coordinates([roots[-1],0.])))
    np.testing.assert_array_equal(clouds[0],clouds[1]);np.testing.assert_array_equal(clouds[0],clouds[2])
    np.testing.assert_allclose(barriers,np.array([.5,1.,2.])*barriers[1],atol=1e-14)


def test_moving_incumbent_preserves_total_K_P_and_its_assigned_K():
    for seed in (1483,1484):
        for e in (0.,1e-5,1e-3):
            v=controlled_velocities(seed,e)
            np.testing.assert_array_equal(v,controlled_velocities(seed,e))
            np.testing.assert_allclose(v.sum(axis=0),np.zeros(2),atol=1e-14)
            assert .5*np.sum(v*v)==pytest.approx(.6,abs=1e-14)
            assert .5*np.sum(v[0]*v[0])==pytest.approx(e,abs=1e-14)
    for e in (-1.,.45,np.nan):
        with pytest.raises(ValueError):controlled_velocities(1483,e)


def test_stationary_incumbent_reproduces_parent_velocity_recipe():
    np.testing.assert_array_equal(controlled_velocities(1485,0.),explorer_velocities(1485))


def test_curvature_probes_and_equal_energy_with_moving_incumbent():
    obj=ScaledEnergyWells(2.,1486);xs,_=parameter_cloud(obj,'good_present',1486);oracle=PopulationOracle(obj)
    vals,lambdas,_=measured_curvatures(oracle,xs)
    assert oracle.counts()=={'value_calls':4,'gradient_calls':16,'path_calls':0}
    for x,lam in zip(xs,lambdas):assert lam==pytest.approx(max(2*(3*obj.coordinates(x)[0]**2-1),8.),abs=1e-7)
    energies=[]
    for policy in ('fixed_high','reserve_best','reserve_random'):
        state=initial_state(xs,vals,allocate(vals,lambdas,policy,1486),controlled_velocities(1486,1e-3))
        energies.append(state.kinetic+.5*sum(p.z for p in state.particles))
    np.testing.assert_allclose(energies,energies[0],atol=1e-14)


def test_temporary_excursion_and_final_recovery_are_distinct():
    obj=ScaledEnergyWells(1.,1487);xs,roots=parameter_cloud(obj,'good_present',1487)
    low=obj.from_coordinates([roots[0],0.]);high=obj.from_coordinates([roots[0]+.1,0.])
    minimum=obj.value(low);ctx={'condition':'good_present','values':[minimum]*4,'minimum':minimum,
        'target':minimum+1e-4,'boundary':roots[1],'reference_point0':low.tolist()}
    def row(x,t):return {'time':t,'positions':[list(x)+[obj.value(x)]]*4,'values':[obj.value(x)]*4}
    result=generalization_score([row(low,0.),row(high,1.),row(low,2.)],ctx,obj)
    assert result['particle0_harmed'] and result['particle0_final_hit'] and result['particle0_final_rise']==0.


def test_new_task_keeps_five_policy_accounting_and_motion_metadata(tmp_path):
    for name in ('contexts','traces','events','maps','uncommitted','initial_states'):(tmp_path/name).mkdir()
    result=task(tmp_path,'good_present',.5,1e-3,1488,100,2)
    assert len(result['runs'])==5 and result['shared_probe_calls']==20
    assert result['contexts'][0]['incumbent_energy']==1e-3
    for r in result['runs']:
        assert r['longitudinal_scale']==.5 and r['incumbent_energy']==1e-3 and r['initial_kinetic']==pytest.approx(.6)
        assert r['actual_branch_calls']==28 and 'particle0_final_hit' in r
    assert sum(r['total_oracle_calls'] for r in result['runs'])-20-result['actual_branch_calls']==48
