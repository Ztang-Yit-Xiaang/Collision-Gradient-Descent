"""Empirical initial-condition and bounce-return diagnostics, not a proof."""

import numpy as np
from .core import State, specular_reflection
from .event_replay import pack, unpack
from .gravity import GravityConfig, GravityContact, ballistic_contact_time
from .hybrid import ContactResolutionError
from .objectives import EvaluationBudgetExceeded, Quadratic
from .population import PopulationOracle, PopulationState

CURVATURES = (1.,4.,16.)
PRODUCTS = (.02,.08,.2,.35,.45,.49,.51,.55,.7,1.)
ROLLOUT_PRODUCTS = (.2,.49,.51,.7)
PERTURBATIONS = ('zero','high_plus','high_minus','low_plus')


def rosen_basis(objective, optimum, delta=1e-5):
    columns=[];probes=[]
    for i in range(len(optimum)):
        direction=np.eye(len(optimum))[i]
        plus=objective.gradient(optimum+delta*direction)
        minus=objective.gradient(optimum-delta*direction)
        columns.append((plus-minus)/(2*delta))
        probes.append({'axis':i,'delta':delta,'plus_gradient':plus.tolist(),'minus_gradient':minus.tolist()})
    matrix=np.column_stack(columns);matrix=(matrix+matrix.T)/2
    values,vectors=np.linalg.eigh(matrix)
    for j in range(vectors.shape[1]):
        nz=np.flatnonzero(np.abs(vectors[:,j])>1e-14)
        if len(nz) and vectors[nz[0],j]<0:vectors[:,j]*=-1
    return {'optimum':np.asarray(optimum).tolist(),'hessian':matrix.tolist(),
            'eigenvalues':values.tolist(),'low_direction':vectors[:,0].tolist(),
            'high_direction':vectors[:,-1].tolist(),'probes':probes,
            'baseline_gradient':objective.gradient(optimum).tolist(),
            'baseline_value':objective.value(optimum),'setup_gradient_calls':5,'setup_value_calls':1}


def controlled_initial(parent,basis,height,perturbation,layout,objective):
    if perturbation not in PERTURBATIONS or layout not in ('single','joint') or height<=0:
        raise ValueError('Invalid diagnostic initial condition.')
    original=unpack(parent);x=np.array(basis['optimum'],float)
    if perturbation!='zero':
        direction=np.array(basis['low_direction'] if perturbation=='low_plus' else basis['high_direction'])
        x=x+( -1e-8 if perturbation=='high_minus' else 1e-8)*direction
    particle=State(x,np.zeros_like(x),objective.value(x)+height,0.)
    return PopulationState((particle,)+original.particles[1:] if layout=='joint' else (particle,))


def bowl_launch(oracle,kappa,height,x=0.,vx=0.,gravity=.5):
    if not np.all(np.isfinite([kappa,height,x,vx,gravity])) or min(kappa,height,gravity)<=0:
        raise ValueError('Invalid bowl launch.')
    value=oracle.value(np.array([x]));remaining=2*gravity*(height-value)-vx*vx
    if remaining<=0:
        raise ValueError('Launch exceeds the fixed mechanical-energy surface.')
    vz=float(np.sqrt(remaining))
    if vz-kappa*x*vx<=0:
        raise ValueError('Bowl launch must be outgoing.')
    return State([x],[vx],value,vz),value


def next_bounce(oracle,state,height,gravity=.5):
    cfg=GravityConfig(beta=1.,gravity=gravity);method=GravityContact(cfg)
    value0=oracle.value(state.x)
    if state.z-value0 < -cfg.gap_tolerance:
        raise ContactResolutionError('Return starts inside terrain.')
    bound=max(1.,8*np.sqrt(height/gravity))
    duration=ballistic_contact_time(oracle,state,bound,cfg)
    if duration is None:
        raise ContactResolutionError('No incoming bounce inside conservative flight bound.')
    incoming=method._flight(state,duration);value=oracle.value(incoming.x)
    if abs(incoming.z-value)>10*cfg.gap_tolerance:
        raise ContactResolutionError('Bowl return is off the terrain.')
    normal=np.r_[-oracle.gradient(incoming.x),1.];normal/=np.linalg.norm(normal)
    before=incoming.velocity
    if before@normal>=-cfg.normal_tolerance:
        raise ContactResolutionError('Bowl return is not incoming.')
    after=specular_reflection(before,normal)
    final=State(incoming.x,after[:-1],incoming.z,after[-1])
    wall=after-before;gravity_impulse=np.array([0.,-gravity*duration])
    e0=state.kinetic+gravity*state.z;e1=final.kinetic+gravity*final.z
    record={'duration':duration,'flight_bound':bound,'point':np.r_[incoming.x,incoming.z].tolist(),
            'normal':normal.tolist(),'incoming_velocity':before.tolist(),'outgoing_velocity':after.tolist(),
            'wall_impulse':wall.tolist(),'gravity_impulse':gravity_impulse.tolist(),
            'energy_residual':e1-e0,'reflection_energy_error':float(.5*(after@after-before@before)),
            'momentum_residual':float(np.linalg.norm(after-state.velocity-gravity_impulse-wall)),
            'loss':value,'energy':e1,'kinetic':final.kinetic}
    return final,record


def bowl_returns(kappa,height,*,x=0.,vx=0.,bounces=1,budget=2000):
    oracle=PopulationOracle(Quadratic(np.array([[kappa]])),total_budget=budget)
    state,value=bowl_launch(oracle,kappa,height,x,vx)
    initial=state;time=0.;best=value;rows=[];events=[];reason='completed';uncommitted=None
    tau=2*np.sqrt(2*height/.5)
    def row(index,loss):
        counts=oracle.counts()
        return {'bounce':index,'time':time,'x':float(state.x[0]),'vx':float(state.v[0]),
                'z':state.z,'vz':state.vz,'loss':loss,'best_loss':best,
                'phase_norm':float(np.hypot(state.x[0],tau*state.v[0])),
                'energy':state.kinetic+.5*state.z,'kinetic':state.kinetic,
                **counts,'total_oracle_calls':sum(counts.values())}
    rows.append(row(0,value))
    for index in range(1,bounces+1):
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                final,event=next_bounce(oracle,state,height)
        except (EvaluationBudgetExceeded,ContactResolutionError,FloatingPointError,np.linalg.LinAlgError) as exc:
            kind='budget' if isinstance(exc,EvaluationBudgetExceeded) else 'contact_error' if isinstance(exc,ContactResolutionError) else 'numerical_error'
            reason=f'{kind}: {exc}'
            uncommitted={'input':pack(PopulationState((state,))),'attempted_bounce':index,'counts':oracle.counts()}
            break
        state=final;time+=event['duration'];best=min(best,event['loss']);events.append(event)
        rows.append(row(index,event['loss']))
    counts=oracle.counts()
    return {'initial':pack(PopulationState((initial,))),'rows':rows,'events':events,
            'stop_reason':reason,'failed':reason.startswith(('contact_error:','numerical_error:')),
            'counts':{**counts,'total_oracle_calls':sum(counts.values())},'uncommitted':uncommitted}


def empirical_matrices(probes):
    bykey={(p['axis'],p['epsilon'],p['sign']):p for p in probes}
    matrices=[]
    for eps in (1e-6,1e-7):
        columns=[]
        for axis in ('x','vx'):
            plus=bykey[(axis,eps,1)];minus=bykey[(axis,eps,-1)]
            if plus['failed'] or minus['failed'] or plus['completed_bounces']!=1 or minus['completed_bounces']!=1:
                columns=[];break
            columns.append((np.array(plus['return_coordinates'])-minus['return_coordinates'])/(2*eps))
        if not columns:
            matrices.append({'epsilon':eps,'available':False});continue
        matrix=np.column_stack(columns);eigenvalues=np.linalg.eigvals(matrix)
        radius=float(np.max(np.abs(eigenvalues)))
        matrices.append({'epsilon':eps,'available':True,'matrix':matrix.tolist(),
                         'eigenvalues_real':eigenvalues.real.tolist(),'eigenvalues_imag':eigenvalues.imag.tolist(),
                         'spectral_radius':radius,
                         'screen':'amplifying' if radius>1.01 else 'nonexpanding' if radius<=1.0001 else 'ambiguous'})
    return matrices
