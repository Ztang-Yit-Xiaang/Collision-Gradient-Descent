"""E24 initialization ablation; no online impulses or binary collisions."""
import numpy as np
from .curvature_energy import allocate

HEIGHT_MODES=('high','curve','fixed_ref','fixed_safe')
POLICIES=tuple(h+'_'+v for h in HEIGHT_MODES for v in ('keep','transfer'))


def transfer_velocity(velocities,pivot):
    v=np.asarray(velocities,float)
    if v.shape!=(4,2) or not np.all(np.isfinite(v)) or pivot not in range(4):
        raise ValueError('Four finite velocities and valid selected index required.')
    if np.linalg.norm(v.sum(axis=0))>1e-12:
        raise ValueError('This initialization map requires zero total linear momentum.')
    total=float(np.sum(v*v));out=np.zeros_like(v)
    if total==0:return out
    indices=[i for i in range(4) if i!=pivot]
    centered=v[indices]+v[pivot]/3
    spread=float(np.sum(centered*centered))
    if spread<=1e-24:
        raise ValueError('Degenerate spectator spread; no reseeding or invented transfer direction.')
    out[indices]=centered*np.sqrt(total/spread)
    return out


def allocate_initial(values,curvatures,policy,velocities):
    if policy not in POLICIES or len(values)!=4 or len(curvatures)!=4:
        raise ValueError('Known four-particle initialization policy required.')
    height_mode,velocity_mode=policy.rsplit('_',1);pivot=int(np.argmin(values))
    if height_mode=='curve':
        record=allocate(values,curvatures,'reserve_best',0)
    else:
        selected_height={'high':.8,'fixed_ref':.025,'fixed_safe':.00625}[height_mode]
        H=np.full(4,.8+(.8-selected_height)/3);H[pivot]=selected_height
        record={'heights':H.tolist(),'clearance_potential':float(.5*sum(H)),
                'charged_value_prefix':4,'charged_gradient_prefix':0}
    v=np.asarray(velocities,float)
    if v.shape!=(4,2) or not np.all(np.isfinite(v)):raise ValueError('Finite4x2 velocities required.')
    after=transfer_velocity(v,pivot) if velocity_mode=='transfer' else v.copy()
    return {**record,'policy':policy,'height_mode':height_mode,'velocity_mode':velocity_mode,
            'selected_index':pivot,'horizontal_velocities':after.tolist(),
            'selected_kinetic_before':float(.5*np.sum(v[pivot]**2)),
            'selected_kinetic_after':float(.5*np.sum(after[pivot]**2)),
            'total_kinetic_before':float(.5*np.sum(v*v)),
            'total_kinetic_after':float(.5*np.sum(after*after)),
            'linear_momentum_before':v.sum(axis=0).tolist(),'linear_momentum_after':after.sum(axis=0).tolist(),
            'no_post_initialization_control':True,'dynamical_collision_implemented':False,
            'angular_momentum_conservation_claimed':False}


def angular_allocation(points,values,allocation,before):
    q=np.column_stack([points,np.array(values)+allocation['heights']])
    original=np.column_stack([before,np.zeros(4)])
    after=np.column_stack([allocation['horizontal_velocities'],np.zeros(4)])
    L0=np.cross(q,original).sum(axis=0);L1=np.cross(q,after).sum(axis=0)
    return {'angular_before_velocity_allocation':L0.tolist(),'angular_after_velocity_allocation':L1.tolist(),
            'angular_change_at_same_positions':(L1-L0).tolist(),
            'role':'Initialization comparison; not a claimed internal collision or runtime torque'}
