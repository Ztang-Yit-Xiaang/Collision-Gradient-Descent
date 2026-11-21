"""E22 observed-curvature initialization; no state changes after launch."""
import numpy as np
from .core import State
from .cooling import CoolingPopulation
from .objectives import TiltedDoubleWell
from .population import PopulationState

POLICIES=('fixed_high','fixed_low','cap_all','reserve_best','reserve_random')
STIFFNESSES=(1.,8.,32.)
CONDITIONS=('good_present','bad_only')


class RotatedEnergyWells:
    def __init__(self,stiffness,seed):
        if not np.isfinite(stiffness) or stiffness<=0:raise ValueError('Positive stiffness required.')
        self.stiffness=float(stiffness);self.seed=int(seed);self.base=TiltedDoubleWell()
        angle=np.random.default_rng(20261109+seed).uniform(0,2*np.pi)
        self.rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    def coordinates(self,x):return self.rotation@np.asarray(x)
    def from_coordinates(self,r):return self.rotation.T@np.asarray(r)

    def value(self,x):
        r=self.coordinates(x)
        return self.base.value(r)+.5*(self.stiffness-1)*r[1]**2

    def gradient(self,x):
        r=self.coordinates(x);g=self.base.gradient(r);g[1]+=(self.stiffness-1)*r[1]
        return self.rotation.T@g

    def path_coefficients(self,x,displacement):
        r,s=self.coordinates(x),self.coordinates(displacement)
        coefficients=self.base.path_coefficients(r,s)
        coefficients=np.pad(coefficients,(0,max(0,3-len(coefficients))))
        coefficients[:3]+=(self.stiffness-1)*np.array([.5*r[1]**2,r[1]*s[1],.5*s[1]**2])
        return coefficients


def parameter_cloud(objective,condition,seed):
    if condition not in CONDITIONS:raise ValueError('Unknown condition.')
    roots=np.sort(np.roots([1.,0.,-1.,objective.base.tilt]).real)
    rng=np.random.default_rng(20261110+seed);cloud=[]
    for i in range(4):
        delta=rng.normal(size=2);delta*=.001/np.linalg.norm(delta)
        center=roots[0] if condition=='good_present' and i==0 else roots[-1]
        cloud.append(objective.from_coordinates(np.array([center,0.])+delta))
    return cloud,roots


def measured_curvatures(oracle,points,step=1e-5):
    if not np.isfinite(step) or step<=0:raise ValueError('Positive probe step required.')
    values=[oracle.value(x) for x in points];curvatures=[];records=[]
    for x in points:
        columns=[];probes=[]
        for axis in range(2):
            displacement=np.eye(2)[axis]*step
            plus=oracle.gradient(x+displacement);minus=oracle.gradient(x-displacement)
            columns.append((plus-minus)/(2*step))
            probes.append({'axis':axis,'step':step,'plus_gradient':plus.tolist(),'minus_gradient':minus.tolist()})
        matrix=np.column_stack(columns);matrix=(matrix+matrix.T)/2;eigenvalues=np.linalg.eigvalsh(matrix)
        curvatures.append(float(eigenvalues[-1]));records.append({'hessian':matrix.tolist(),'eigenvalues':eigenvalues.tolist(),'probes':probes})
    return values,curvatures,records


def random_reserve_index(seed):
    return int(np.random.default_rng(np.random.SeedSequence([20261109,seed])).integers(0,4))


def allocate(values,curvatures,policy,seed):
    if policy not in POLICIES or len(values)!=4 or len(curvatures)!=4:raise ValueError('Four-particle policy required.')
    caps=np.minimum(.8,.2/np.maximum(np.array(curvatures),1e-6));pivot=None
    if policy=='fixed_high':heights=np.full(4,.8)
    elif policy=='fixed_low':heights=np.full(4,.02)
    elif policy=='cap_all':heights=caps.copy()
    else:
        pivot=int(np.argmin(values)) if policy=='reserve_best' else random_reserve_index(seed)
        heights=np.full(4,.8+(.8-caps[pivot])/3);heights[pivot]=caps[pivot]
    return {'policy':policy,'heights':heights.tolist(),'caps':caps.tolist(),'selected_index':pivot,
            'clearance_potential':float(.5*sum(heights)),
            'curvature_height_products':(np.array(curvatures)*heights).tolist(),
            'charged_value_prefix':4,'charged_gradient_prefix':0 if policy.startswith('fixed_') else 16,
            'no_post_initialization_control':True}


def initial_state(points,values,allocation,velocities=None):
    velocities=np.zeros((4,2)) if velocities is None else np.asarray(velocities,float)
    if velocities.shape!=(4,2) or not np.all(np.isfinite(velocities)):raise ValueError('Four finite horizontal velocities required.')
    return PopulationState(tuple(State(x,v,f+h,0.) for x,v,f,h in zip(points,velocities,values,allocation['heights'])))


def explorer_velocities(seed):
    rng=np.random.default_rng(np.random.SeedSequence([20261110,seed,60]));v=rng.normal(size=(3,2));v-=v.mean(axis=0)
    v*=np.sqrt(1.2/np.sum(v*v))
    return np.vstack([np.zeros(2),v])


class IndependentEnergyPopulation(CoolingPopulation):
    def __init__(self,config):
        if config.collisions:raise ValueError('Independent point-cloud adapter forbids binary interactions.')
        super().__init__(config,schedule='ramp',rule='elastic')

    def _validate(self,oracle,particles,*,initial=False):
        # Independent point particles may overlap; terrain feasibility stays checked.
        return super()._validate(oracle,particles,initial=False)


def score(rows,context,objective):
    minimum=context['minimum'];target=context['target'];boundary=context['boundary'];initial=context['values']
    def hit(row):
        return any(f<=target and objective.coordinates(np.array(q[:-1]))[0]<boundary
                   for f,q in zip(row['values'],row['positions']))
    def left(row):return any(objective.coordinates(np.array(q[:-1]))[0]<boundary for q in row['positions'])
    p0=[r['values'][0] for r in rows]
    best=min(min(r['values']) for r in rows)
    return {'final_value':min(rows[-1]['values']),'best_value':best,
            'final_hit':hit(rows[-1]),'historical_hit':any(hit(r) for r in rows),
            'ever_left':any(left(r) for r in rows),'first_left_time':next((r['time'] for r in rows if left(r)),None),
            'relative_final_gap':(min(rows[-1]['values'])-minimum)/(min(initial)-minimum) if context['condition']=='bad_only' else None,
            'particle0_final_value':p0[-1],'particle0_final_rise':p0[-1]-initial[0],
            'particle0_max_rise':max(0.,max(p0)-initial[0]),'particle0_harmed':max(p0)-initial[0]>1e-4,
            'particle0_max_above_global':max(p0)-minimum,
            'particle0_final_distance':float(np.linalg.norm(np.array(rows[-1]['positions'][0][:-1])-context['reference_point0']))}
