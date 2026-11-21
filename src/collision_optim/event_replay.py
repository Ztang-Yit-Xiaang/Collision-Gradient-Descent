"""Isolated contact counterfactuals; frozen population dynamics remain untouched."""

from copy import deepcopy
from dataclasses import replace
import numpy as np

from .core import State, elastic_pair_collision
from .cooling import drag_rate
from .gravity import GravityConfig, GravityContact
from .hybrid import ContactResolutionError
from .objectives import EvaluationBudgetExceeded
from .population import PopulationConfig, PopulationOracle, PopulationState
from .scattering_population import ScatteringPopulation, random_pair_collision

MAGNITUDE_KEYS = ('source_time', 'pair_loss', 'pair_kinetic', 'relative_speed',
                  'center_speed', 'impulse_norm', 'abs_energy_transfer',
                  'gradient_norm_i', 'gradient_norm_j', 'gradient_zero_i', 'gradient_zero_j')
DIRECTION_KEYS = ('gradient_impulse_cos_i', 'gradient_impulse_cos_j',
                  'velocity_impulse_cos_i', 'velocity_impulse_cos_j',
                  'normal_gradient_cos_i', 'normal_gradient_cos_j')


def pack(state):
    return {'positions': [np.r_[p.x, p.z].tolist() for p in state.particles],
            'velocities': [p.velocity.tolist() for p in state.particles]}


def unpack(record):
    return PopulationState(tuple(State(q[:-1], v[:-1], q[-1], v[-1])
                                 for q, v in zip(record['positions'], record['velocities'])))


def reconstruct_contact(rows, events, event_index, h=.05):
    """Reconstruct one committed contact, auditing its whole observation step."""
    target = events[event_index]
    if target['kind'] != 'pair':
        raise ValueError('A binary event is required.')
    step = target['step']; start = (step-1)*h
    previous = unpack(rows[step-1]); beta = rows[step]['drag_beta']
    particles = [State(p.x, beta*p.v, p.z, beta*p.vz) for p in previous.particles]
    gravity = GravityContact(GravityConfig(h=h)); elapsed = 0.; error = 0.; incoming = None
    for index, event in enumerate(events):
        if event['step'] != step:
            continue
        time = event['time']-start
        if time < elapsed-1e-10:
            raise ValueError('Saved events are not chronological.')
        particles = [gravity._flight(p, max(0., time-elapsed)) for p in particles]
        elapsed = time
        for suffix, particle in [('i', event['i']), ('j', event.get('j'))]:
            if particle is None:
                continue
            p = particles[particle]; q = np.array(event['point_'+suffix])
            v = np.array(event['before_'+suffix])
            error = max(error, float(np.max(np.abs(np.r_[p.x, p.z]-q))),
                        float(np.max(np.abs(p.velocity-v))))
            particles[particle] = State(q[:-1], v[:-1], q[-1], v[-1])
        if index == event_index:
            incoming = PopulationState(tuple(particles))
        for suffix, particle in [('i', event['i']), ('j', event.get('j'))]:
            if particle is not None:
                p = particles[particle]; v = np.array(event['after_'+suffix])
                particles[particle] = State(p.x, v[:-1], p.z, v[-1])
    particles = [gravity._flight(p, max(0., h-elapsed)) for p in particles]
    rebuilt, saved = pack(PopulationState(tuple(particles))), pack(unpack(rows[step]))
    error = max(error, *(float(np.max(np.abs(np.array(rebuilt[k])-saved[k])))
                         for k in ('positions', 'velocities')))
    if incoming is None or error > 1e-9:
        raise ValueError(f'Contact reconstruction failed: {error}')
    return incoming, {'source_step': step, 'source_time': target['time'],
                      'local_time': target['time']-start,
                      'source_event_index': event_index,
                      'i': target['i'], 'j': target['j'], 'normal': target['normal'],
                      'reconstruction_error': error, 'source_h': h}


def mapped_state(incoming, context, rule, seed=0):
    i, j = context['i'], context['j']; p, q = incoming.particles[i], incoming.particles[j]
    n = np.array(context['normal']); vi, vj = p.velocity, q.velocity
    if rule == 'elastic':
        ai, aj = elastic_pair_collision(vi, vj, n)
    elif rule == 'random':
        ai, aj = random_pair_collision(vi, vj, n, np.random.default_rng(seed))
    elif rule == 'none':
        ai, aj = vi.copy(), vj.copy()
    else:
        raise ValueError('Unknown map.')
    particles = list(incoming.particles)
    particles[i] = State(p.x, ai[:-1], p.z, ai[-1])
    particles[j] = State(q.x, aj[:-1], q.z, aj[-1])
    state = PopulationState(tuple(particles)); impulse = ai-vi
    separation = np.r_[p.x-q.x, p.z-q.z]
    record = {'rule': rule, 'before': pack(incoming), 'after': pack(state),
              'impulse_i': impulse.tolist(), 'impulse_j': (aj-vj).tolist(),
              'kinetic_gain_i': float(.5*(ai@ai-vi@vi)),
              'energy_error': state.kinetic-incoming.kinetic,
              'momentum_error': float(np.linalg.norm(state.momentum-incoming.momentum)),
              'angular_impulse_norm': float(np.linalg.norm(np.outer(separation, impulse)-np.outer(impulse, separation))),
              'external_energy_supply': 0., 'external_net_impulse': [0.]*len(vi)}
    return state, record


def contact_features(incoming, context, values, gradients, initial_best):
    """Inputs are exclusively the current incoming state and paid probes."""
    i, j = context['i'], context['j']; vi, vj = (incoming.particles[k].velocity for k in (i, j))
    gi, gj = (np.r_[g, 0.] for g in gradients); n = np.array(context['normal'])
    ai, aj = elastic_pair_collision(vi, vj, n); impulse = ai-vi
    def cosine(a, b):
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        return float(a@b/denom) if denom > 1e-12 else 0.
    return {'source_time': context['source_time'],
            'pair_loss': min(values[i], values[j])/max(1., abs(initial_best)),
            'pair_kinetic': float(.5*(vi@vi+vj@vj)),
            'relative_speed': float(np.linalg.norm(vi-vj)),
            'center_speed': float(np.linalg.norm((vi+vj)/2)),
            'impulse_norm': float(np.linalg.norm(impulse)),
            'abs_energy_transfer': abs(float(.5*(ai@ai-vi@vi))),
            'gradient_norm_i': float(np.linalg.norm(gi)), 'gradient_norm_j': float(np.linalg.norm(gj)),
            'gradient_zero_i': float(np.linalg.norm(gi) <= 1e-12),
            'gradient_zero_j': float(np.linalg.norm(gj) <= 1e-12),
            'gradient_impulse_cos_i': cosine(gi, impulse),
            'gradient_impulse_cos_j': cosine(gj, -impulse),
            'velocity_impulse_cos_i': cosine(vi, impulse),
            'velocity_impulse_cos_j': cosine(vj, -impulse),
            'normal_gradient_cos_i': cosine(n, gi), 'normal_gradient_cos_j': cosine(-n, gj)}


class ReplayPopulation(ScatteringPopulation):
    def __init__(self, context):
        self.source_h = context['source_h']; self.source_step = context['source_step']
        self.source_time = context['source_time']; self.elapsed = 0.; self.steps = 0
        self.remaining = max(0., self.source_h-context['local_time'])
        self.partial = self.remaining > 1e-11
        super().__init__(PopulationConfig(physics=GravityConfig(h=self.source_h), rho=.24,
                                          collisions=False), rule='elastic')

    def _validate(self, oracle, particles, *, initial=False):
        # Ghost counterfactuals legitimately overlap after the isolated event.
        return super()._validate(oracle, particles, initial=False)

    def step(self, oracle, state, *, basin_boundary=None):
        partial = self.steps == 0 and self.partial
        duration = self.remaining if partial else self.source_h
        clock = self.source_time+self.elapsed
        rate = 0. if partial else drag_rate('ramp', clock, duration)
        beta = 1. if partial else float(np.exp(-rate*duration))
        self.config = replace(self.config, physics=replace(self.config.physics, h=duration, beta=beta))
        result = super().step(oracle, state, basin_boundary=basin_boundary)
        result.diagnostics.update(duration=duration, absolute_start_time=clock,
                                  drag_rate=rate, drag_beta=beta, partial_source_step=partial)
        self.elapsed += duration; self.steps += 1
        return result


def run_replay(objective, incoming, context, rule, *, seed=0, budget=2000, steps=100,
               boundary=None):
    state, collision = mapped_state(incoming, context, rule, seed)
    method = ReplayPopulation(context); oracle = PopulationOracle(objective, total_budget=budget)
    # Derived shared charge; these six probes were actually paid once per context.
    oracle.values = 4; oracle.gradients = 2
    values, gap = method._validate(oracle, state.particles)
    def counts():
        c = oracle.counts(); total = sum(c.values())
        return {**c, 'total_oracle_calls': total, 'actual_branch_calls': total-6}
    def row(index, values, diagnostics):
        return {'step': index, 'time': method.elapsed,
                'absolute_time': context['source_time']+method.elapsed,
                'values': list(values), 'minimum_value': min(values),
                **pack(state), **diagnostics, **counts()}
    rows = [row(0, values, {'energy': method.energy(state), 'kinetic': state.kinetic,
                           'momentum': state.momentum.tolist(), 'minimum_surface_gap': gap})]
    events = []; reason = 'completed'; uncommitted = None
    for index in range(1, steps+1):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                result = method.step(oracle, state, basin_boundary=boundary)
            if not np.all(np.isfinite([result.diagnostics['energy'], result.diagnostics['momentum_residual']])):
                raise FloatingPointError('Nonfinite replay diagnostics.')
        except (EvaluationBudgetExceeded, ContactResolutionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            kind = 'budget' if isinstance(exc, EvaluationBudgetExceeded) else 'contact_error' if isinstance(exc, ContactResolutionError) else 'numerical_error'
            reason = f'{kind}: {exc}'
            uncommitted = {'attempted_step': index, 'starting_state': pack(state), **counts(),
                           'partial_intermediate_states': 'Unavailable in frozen event solver; not reconstructed'}
            break
        state, values = result.state, result.values
        rows.append(row(index, values, result.diagnostics))
        events.extend({**e, 'step': index, 'time': method.elapsed-result.diagnostics['duration']+e['time']}
                      for e in result.events)
    return {'rows': rows, 'events': events, 'map': collision, 'stop_reason': reason,
            'failed': reason.startswith(('contact_error:', 'numerical_error:')),
            'counts': counts(), 'uncommitted': uncommitted}


def outcomes(rows, context, initial_best, minimum, target, boundary):
    i, j = context['i'], context['j']; scale = max(initial_best-minimum, 1e-12)
    first_pair = min(rows[0]['values'][i], rows[0]['values'][j])
    def evaluate(points):
        last = points[-1]; values = last['values']; positions = last['positions']
        pair_values = [min(r['values'][i], r['values'][j]) for r in points]
        def hit(r):
            return any(r['values'][k] <= target and (boundary is None or r['positions'][k][0] < boundary)
                       for k in (i, j))
        return {'pair_final_loss': min(values[i], values[j]), 'group_final_loss': min(values),
                'pair_best_loss': min(pair_values), 'group_best_loss': min(min(r['values']) for r in points),
                'pair_final_gap': (min(values[i], values[j])-minimum)/scale,
                'pair_max_rise': max(0., max(pair_values)-first_pair),
                'group_max_rise': max(0., max(min(r['values']) for r in points)-min(rows[0]['values'])),
                'pair_final_hit': hit(last), 'pair_historical_hit': any(hit(r) for r in points),
                'pair_crossed': boundary is not None and any(r['positions'][k][0] < boundary for r in points for k in (i, j)),
                'time': last['time'], 'charged_calls': last['total_oracle_calls'], 'step': last['step']}
    views = {'end': evaluate(rows)}
    under = [r for r in rows if r['total_oracle_calls'] <= 1000]
    views['budget1000'] = evaluate(under) if under else None
    for index in (10, 50, 100):
        views['boundary'+str(index)] = evaluate(rows[:index+1]) if len(rows) > index else None
    return views
