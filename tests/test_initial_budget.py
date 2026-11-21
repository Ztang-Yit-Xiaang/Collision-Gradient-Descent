import numpy as np
import pytest
from collision_optim.curvature_energy import RotatedEnergyWells,parameter_cloud,allocate,initial_state
from collision_optim.energy_generalization import controlled_velocities
from collision_optim.initial_budget import POLICIES,allocate_initial,transfer_velocity,angular_allocation
from collision_optim.initial_budget_experiment import task


def test_velocity_transfer_preserves_K_and_P_for_every_selected_index():
    v=controlled_velocities(1580,.001)
    for j in range(4):
        w=transfer_velocity(v,j)
        np.testing.assert_array_equal(w[j],np.zeros(2))
        np.testing.assert_allclose(w.sum(axis=0),v.sum(axis=0),atol=1e-14)
        assert np.sum(w*w)==pytest.approx(np.sum(v*v),abs=1e-14)
    np.testing.assert_array_equal(v,controlled_velocities(1580,.001))


def test_transfer_handles_zero_and_rejects_infeasible_domain_without_reseeding():
    np.testing.assert_array_equal(transfer_velocity(np.zeros((4,2)),2),np.zeros((4,2)))
    with pytest.raises(ValueError,match='zero total'):transfer_velocity(np.ones((4,2)),0)
    with pytest.raises(ValueError,match='Degenerate'):transfer_velocity([[3.,0.],[-1.,0.],[-1.,0.],[-1.,0.]],0)
    with pytest.raises(ValueError):transfer_velocity(np.zeros((3,2)),0)


def test_selection_uses_actual_current_best_including_nonzero_index():
    v=controlled_velocities(1581,.001);values=[.4,.3,-.1,.2]
    for policy in POLICIES:
        r=allocate_initial(values,[8.]*4,policy,v)
        assert r['selected_index']==2
        if policy.endswith('transfer'):assert r['selected_kinetic_after']==0.
        else:np.testing.assert_array_equal(r['horizontal_velocities'],v)


def test_every_policy_keeps_equal_initial_E_K_and_P():
    obj=RotatedEnergyWells(32.,1582);xs,_=parameter_cloud(obj,'good_present',1582)
    values=[obj.value(x) for x in xs];v=controlled_velocities(1582,.001);energies=[]
    for policy in POLICIES:
        allocation=allocate_initial(values,[32.]*4,policy,v)
        assert sum(allocation['heights'])==pytest.approx(3.2,abs=1e-14)
        state=initial_state(xs,values,allocation,allocation['horizontal_velocities'])
        assert state.kinetic==pytest.approx(.6,abs=1e-14)
        np.testing.assert_allclose(state.momentum,np.zeros(3),atol=1e-14)
        energies.append(state.kinetic+.5*sum(p.z for p in state.particles))
    np.testing.assert_allclose(energies,energies[0],atol=1e-14)


def test_keep_policies_reproduce_parent_high_and_curve():
    vals=[-.1,.2,.3,.4];curv=[2.,8.,16.,32.];v=controlled_velocities(1583,.001)
    for new,old in [('high_keep','fixed_high'),('curve_keep','reserve_best')]:
        a=allocate_initial(vals,curv,new,v);b=allocate(vals,curv,old,1583)
        np.testing.assert_array_equal(a['heights'],b['heights'])
        np.testing.assert_array_equal(a['horizontal_velocities'],v)


def test_fixed_controls_ignore_curvature_and_k8_matches_curve():
    vals=[-.1,.2,.3,.4];v=controlled_velocities(1584,.001)
    for mode,height in [('fixed_ref',.025),('fixed_safe',.00625)]:
        a=allocate_initial(vals,[8.]*4,mode+'_transfer',v)
        b=allocate_initial(vals,[1.,2.,3.,4.],mode+'_transfer',v)
        assert a['heights']==b['heights'] and a['heights'][0]==height
    np.testing.assert_array_equal(allocate_initial(vals,[8.]*4,'curve_transfer',v)['heights'],
                                  allocate_initial(vals,[8.]*4,'fixed_ref_transfer',v)['heights'])


def test_angular_change_is_recorded_without_conservation_claim():
    points=[[0.,0.],[1.,0.],[0.,1.],[1.,1.]];vals=[0.,1.,2.,3.];v=controlled_velocities(1585,.001)
    allocation=allocate_initial(vals,[8.]*4,'fixed_ref_transfer',v)
    r=angular_allocation(points,vals,allocation,v)
    assert np.linalg.norm(r['angular_change_at_same_positions'])>1e-5
    assert not allocation['angular_momentum_conservation_claimed'] and not allocation['dynamical_collision_implemented']


def test_eight_branch_prefix_accounting_saves44_not48(tmp_path):
    for name in ('contexts','traces','events','maps','uncommitted','initial_states'):(tmp_path/name).mkdir()
    result=task(tmp_path,'good_present',8.,1586,100,2)
    assert len(result['runs'])==8 and result['shared_probe_calls']==20
    for r in result['runs']:
        assert r['charged_prefix_calls']==(20 if r['policy'].startswith('curve_') else 4)
        assert r['actual_branch_calls']==28
    assert sum(r['total_oracle_calls'] for r in result['runs'])-20-result['actual_branch_calls']==44
