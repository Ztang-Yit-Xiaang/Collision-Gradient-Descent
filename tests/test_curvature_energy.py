import numpy as np
import pytest
from numpy.polynomial import polynomial
from collision_optim.core import State
from collision_optim.curvature_energy import (POLICIES,RotatedEnergyWells,parameter_cloud,measured_curvatures,
    allocate,initial_state,IndependentEnergyPopulation,random_reserve_index,explorer_velocities)
from collision_optim.curvature_energy_experiment import task
from collision_optim.gravity import GravityConfig
from collision_optim.hybrid import ContactResolutionError
from collision_optim.population import PopulationConfig,PopulationOracle,PopulationState,run_population


def test_rotated_value_gradient_and_path_geometry():
    obj=RotatedEnergyWells(8.,1321);x=np.array([.3,-.2]);d=np.array([.4,.1]);eps=1e-6
    fd=np.array([(obj.value(x+eps*e)-obj.value(x-eps*e))/(2*eps) for e in np.eye(2)])
    np.testing.assert_allclose(obj.gradient(x),fd,rtol=1e-8,atol=1e-9)
    coefficients=obj.path_coefficients(x,d)
    for t in (0.,.3,1.,-1.):assert polynomial.polyval(t,coefficients)==pytest.approx(obj.value(x+t*d),abs=1e-12)


def test_cloud_rotation_and_perturbations_are_shared_across_stiffnesses():
    clouds=[]
    for k in (1.,8.,32.):
        obj=RotatedEnergyWells(k,1321);cloud,roots=parameter_cloud(obj,'good_present',1321)
        coords=np.array([obj.coordinates(x) for x in cloud]);clouds.append(coords)
        for i,r in enumerate(coords):assert np.linalg.norm(r-[roots[0] if i==0 else roots[-1],0.])==pytest.approx(.001,abs=1e-14)
    np.testing.assert_allclose(clouds[0],clouds[1],atol=1e-14)
    np.testing.assert_allclose(clouds[0],clouds[2],atol=1e-14)


def test_curvature_estimation_pays_all_gradient_queries():
    obj=RotatedEnergyWells(8.,1322);points,_=parameter_cloud(obj,'bad_only',1322);oracle=PopulationOracle(obj)
    _,curvatures,records=measured_curvatures(oracle,points)
    assert oracle.values==4 and oracle.gradients==16 and oracle.paths==0
    for x,k,r in zip(points,curvatures,records):
        expected=max(3*obj.coordinates(x)[0]**2-1,8.)
        assert k==pytest.approx(expected,abs=1e-7)
        assert len(r['probes'])==2


def test_height_cap_and_unequal_energy_baselines_are_explicit():
    values=[.1,.2,.3,.4];curvatures=[2.,8.,16.,32.]
    record=allocate(values,curvatures,'cap_all',1)
    np.testing.assert_allclose(record['heights'],.2/np.array(curvatures))
    assert record['clearance_potential']<allocate(values,curvatures,'fixed_high',1)['clearance_potential']
    fallback=allocate(values,[-2.,0.,2.,8.],'cap_all',1)
    assert fallback['heights'][:2]==[.8,.8]


def test_redistribution_preserves_total_energy_and_initial_momentum():
    obj=RotatedEnergyWells(32.,1323);points,_=parameter_cloud(obj,'good_present',1323)
    values=[obj.value(x) for x in points];curvatures=[32.]*4
    energies=[]
    for policy in ('fixed_high','reserve_best','reserve_random'):
        record=allocate(values,curvatures,policy,1323);state=initial_state(points,values,record)
        assert sum(record['heights'])==pytest.approx(3.2,abs=1e-14)
        np.testing.assert_array_equal(state.momentum,np.zeros(3))
        energies.append(state.kinetic+.5*sum(p.z for p in state.particles))
    np.testing.assert_allclose(energies,energies[0],atol=1e-14)


def test_current_best_selection_and_random_control_use_only_allowed_inputs():
    values=[.2,-.1,.3,.4];curvatures=[8.]*4
    assert allocate(values,curvatures,'reserve_best',1324)['selected_index']==1
    a=allocate(values,curvatures,'reserve_random',1324)['selected_index']
    b=allocate(list(reversed(values)),[32.]*4,'reserve_random',1324)['selected_index']
    assert a==b==random_reserve_index(1324)


def test_overlap_adapter_does_not_enable_binary_collisions():
    with pytest.raises(ValueError,match='forbids'):
        IndependentEnergyPopulation(PopulationConfig(collisions=True))
    obj=RotatedEnergyWells(8.,1325);x=obj.from_coordinates([1.,0.]);f=obj.value(x)
    state=PopulationState(tuple(State(x,[0.,0.],f+.02,0.) for _ in range(4)))
    method=IndependentEnergyPopulation(PopulationConfig(physics=GravityConfig(),collisions=False))
    run=run_population(method,PopulationOracle(obj),state,1)
    assert run.stop_reason=='completed' and not run.events


def test_overlap_adapter_still_rejects_terrain_penetration():
    obj=RotatedEnergyWells(1.,1326);x=np.zeros(2)
    state=PopulationState((State(x,[0.,0.],obj.value(x)-.01,0.),))
    method=IndependentEnergyPopulation(PopulationConfig(collisions=False))
    with pytest.raises(ContactResolutionError,match='terrain'):
        run_population(method,PopulationOracle(obj),state,1)


def test_shared_prefix_counts_charge_fixed_and_curvature_methods_differently(tmp_path):
    for name in ('contexts','traces','events','maps','uncommitted','initial_states'):(tmp_path/name).mkdir()
    result=task(tmp_path,'good_present',32.,1327,100,2)
    assert result['shared_probe_calls']==20 and len(result['runs'])==5
    for r in result['runs']:
        assert r['charged_prefix_calls']==(4 if r['policy'].startswith('fixed_') else 20)
        assert r['actual_branch_calls']==28
    charged=sum(r['total_oracle_calls'] for r in result['runs'])
    actual=result['shared_probe_calls']+result['actual_branch_calls']
    assert charged-actual==48


def test_moving_explorers_have_reproducible_K_point6_zero_total_momentum():
    v=explorer_velocities(1440)
    np.testing.assert_array_equal(v,explorer_velocities(1440))
    np.testing.assert_array_equal(v[0],np.zeros(2))
    np.testing.assert_allclose(v.sum(axis=0),np.zeros(2),atol=1e-14)
    assert .5*np.sum(v*v)==pytest.approx(.6,abs=1e-14)


def test_height_redistribution_keeps_equal_energy_with_moving_explorers():
    obj=RotatedEnergyWells(32.,1441);points,_=parameter_cloud(obj,'good_present',1441)
    values=[obj.value(x) for x in points];v=explorer_velocities(1441);energies=[]
    for policy in ('fixed_high','reserve_best','reserve_random'):
        state=initial_state(points,values,allocate(values,[32.]*4,policy,1441),v)
        assert state.kinetic==pytest.approx(.6,abs=1e-14)
        np.testing.assert_allclose(state.momentum,np.zeros(3),atol=1e-14)
        assert state.particles[0].kinetic==0.
        energies.append(state.kinetic+.5*sum(p.z for p in state.particles))
    np.testing.assert_allclose(energies,energies[0],atol=1e-14)
