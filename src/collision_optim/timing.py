"""Observation-only checkpoints, past-only physics features and local baselines."""

from copy import deepcopy
from dataclasses import replace
import numpy as np

from .cooling import CoolingPopulation
from .handoff import HandoffPopulation
from .core import State
from .population import PopulationState
from .population_experiment import initial_population
from .objectives import EvaluationBudgetExceeded

FRACTIONS=(0.,.25,.5,.75,1.)
BASE_FEATURES=('budget_fraction','current_loss','recent_loss_improvement')
PHYSICS_FEATURES=('kinetic','clearance_potential','center_speed','relative_speed_rms',
                  'position_dispersion','recent_pair_count','mean_abs_kinetic_transfer')


class CheckpointPopulation(CoolingPopulation):
    def __init__(self, config, *, rule, seed, thresholds, budget, initial_value):
        super().__init__(config,schedule='ramp',rule=rule,seed=seed)
        self.thresholds=tuple(thresholds);self.budget=budget;self.scale=max(1.,abs(initial_value))
        self.snapshots={};self.history=[];self.pair_history=[]

    def step(self, oracle, state, *, basin_boundary=None):
        result=super().step(oracle,state,basin_boundary=basin_boundary)
        calls=sum(oracle.counts().values());index=self.committed_steps
        self.history.append({'step':index,'minimum':min(result.values)})
        self.history=self.history[-20:]
        self.pair_history.extend({**e,'global_step':index} for e in result.events if e['kind']=='pair')
        self.pair_history=[e for e in self.pair_history if e['global_step']>=index-19]
        for threshold in self.thresholds:
            if calls>=threshold and threshold not in self.snapshots:
                v=np.array([p.velocity for p in result.state.particles]);q=np.array([np.r_[p.x,p.z] for p in result.state.particles])
                center=v.mean(axis=0)
                features={'budget_fraction':calls/self.budget,'current_loss':min(result.values)/self.scale,
                    'recent_loss_improvement':(self.history[0]['minimum']-min(result.values))/self.scale,
                    'kinetic':result.state.kinetic,'clearance_potential':.5*sum(p.z-f for p,f in zip(result.state.particles,result.values)),
                    'center_speed':float(np.linalg.norm(center)),
                    'relative_speed_rms':float(np.sqrt(np.mean(np.sum((v-center)**2,axis=1)))),
                    'position_dispersion':float(np.sqrt(np.mean(np.sum((q-q.mean(axis=0))**2,axis=1)))),
                    'recent_pair_count':len(self.pair_history),
                    'mean_abs_kinetic_transfer':float(np.mean([abs(e['kinetic_gain_i']) for e in self.pair_history])) if self.pair_history else 0.}
                self.snapshots[threshold]={'state':result.state,'values':list(result.values),'oracle':deepcopy(oracle),
                    'config':self.config,'rng':deepcopy(self.rng.bit_generator.state),'step':index,'calls':calls,
                    'pair_tests':self.pair_tests,'features':features,
                    'history_step_range':[self.history[0]['step'],index],
                    'feature_last_observation_calls':calls}
        return result


def resume_off(snapshot):
    method=HandoffPopulation(replace(snapshot['config'],collisions=False),schedule='ramp',rule='elastic',seed=0)
    method.resuming=True;method.committed_steps=snapshot['step'];method.pair_tests=snapshot['pair_tests']
    method.rng.bit_generator.state=deepcopy(snapshot['rng'])
    return method,deepcopy(snapshot['oracle'])


def protected_population(case, objective, minimum, seed):
    if case=='quadratic':x=np.zeros(2)
    elif case=='double_well':x=np.array([min(np.roots([1.,0.,-1.,objective.tilt]).real),0.])
    else:x=objective.from_coordinates(np.ones(2))
    helper=x+np.array([.26,0.]);height=max(objective.value(x),objective.value(helper))+.02
    fixed=[State(x,[0.,0.],height,0.),State(helper,[-.4,0.],height,0.)]
    for attempt in range(100):
        cloud=initial_population(case,objective,seed+20000+attempt,rho_max=.24)
        rest=cloud.particles[2:]
        if all(np.linalg.norm(np.r_[a.x-b.x,a.z-b.z])>=.26 for a in rest for b in fixed):break
    else:raise ValueError('Could not separate protected-particle stress cloud.')
    rng=np.random.default_rng(seed+20261109);w=rng.normal(size=3);w*=np.sqrt(.48)/np.linalg.norm(w)
    velocities=(np.array([.2,0.,0.])+w,np.array([.2,0.,0.])-w)
    return PopulationState(tuple(fixed+[State(p.x,v[:-1],p.z,v[-1]) for p,v in zip(rest,velocities)]))


def multistart(oracle, initial, *, family, eta, beta=.9):
    x=np.array([p.x for p in initial.particles]);buffers=np.zeros_like(x)
    values=[oracle.value(p) for p in x]
    def counts():return {**oracle.counts(),'total_oracle_calls':sum(oracle.counts().values())}
    rows=[{'step':0,'positions':x.tolist(),'values':list(values),'particle':None,**counts()}]
    attempts=[];reason='completed';index=0
    while True:
        i=index%len(x);before=x[i].copy();old_buffer=buffers[i].copy();g=None
        try:
            g=oracle.gradient(before)
            direction=g if family=='gd' else beta*old_buffer+g
            with np.errstate(over='raise',invalid='raise',divide='raise'):candidate=before-eta*direction
            value=oracle.value(candidate)
        except EvaluationBudgetExceeded as exc:
            reason='budget: '+str(exc)
            if g is not None:attempts.append({'particle':i,'before':before.tolist(),'gradient':g.tolist(),**counts()})
            break
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            reason='numerical_error: '+str(exc);break
        x[i]=candidate;values[i]=value;buffers[i]=direction
        rows.append({'step':index+1,'positions':x.tolist(),'values':list(values),'particle':i,'before':before.tolist(),
            'gradient':g.tolist(),'buffer_before':old_buffer.tolist(),'direction':direction.tolist(),'eta':eta,**counts()})
        index+=1
    return rows,reason,counts(),attempts


def fit_ridge(rows, keys, penalty=1.):
    x=np.array([[r['features'][k] for k in keys] for r in rows],float);y=np.array([r['value'] for r in rows])
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-12]=1.
    design=np.c_[np.ones(len(x)),(x-mean)/scale]
    reg=np.diag([0.]+[penalty]*len(keys));coef=np.linalg.solve(design.T@design+reg,design.T@y)
    return {'keys':list(keys),'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'training_rows':len(rows)}


def predict_ridge(model, rows):
    x=np.array([[r['features'][k] for k in model['keys']] for r in rows]);z=(x-np.array(model['mean']))/np.array(model['scale'])
    return np.c_[np.ones(len(x)),z]@np.array(model['coef'])
