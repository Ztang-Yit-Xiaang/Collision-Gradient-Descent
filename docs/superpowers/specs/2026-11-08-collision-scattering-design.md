# Frozen experiment: physical collision geometry versus conserving random scattering

User approved the physical-mechanism exploration route. This batch implements the
first two stages: isolated impact diagnostics and new two-dimensional populations.
Higher-dimensional or neural-network experiments depend on the findings below.

## Hypotheses and decisions fixed before execution

- H1: with unit equal masses and pair center velocity V=0, both elastic and
  conserving random maps preserve each particle's speed. With V nonzero, the
  transfer is delta K_i=V dot (u_after-u_before), u=(v_i-v_j)/2.
- H2: the actual contact-normal law improves population search relative to four
  independent particles, at the same initial state/energy and shared query cap.
- H3: any H2 gain exceeds that of conserving random outgoing directions. Without
  H3 evidence, describe redistribution/exploration rather than a demonstrated
  advantage of the full contact-normal law.
- No new loss triggers, helper balls, heat, projection, best-state resets or tuning.
  Best output and final current population minimum are both recorded.

## Three arms and two radii

Independent population: existing solver, interparticle contacts disabled.
Elastic: same solver scheduling, original equal-mass normal impulse.
Random: same scheduling and contact detection; hold V and ||u|| fixed and sample
u_after uniformly over the outgoing hemisphere in the full (x,z) velocity space.
Fold a normalized Gaussian direction into the hemisphere. Do not resample based
on loss, terrain or later outcome. Unsupported coupled wall impacts fail explicitly.

Random scattering conserves linear momentum and kinetic energy but generally
violates central-impulse/angular-momentum constraints. It is an ablation, not a
complete frictionless rigid-body law. Record orbital angular impulse separately.
Contact schedules may diverge after the first differing event.

Preserve old source and artifacts. A separate subclass will use a readable copy
of the frozen step scheduler with only the pair map and diagnostics changed.
Check equivalence to the original elastic solver before the formal batch.

## Mechanism diagnostic

One instantaneous collision in three velocity dimensions (2 parameters+height).
u=(-0.4,0,0); V in {(0,0,0),(0.4,0,0),(0,0.4,0)}; horizontal normal angles
{-60,-30,0,30,60} degrees. For each of 15 incoming states: one physical result
and 100 independent random results, 1,515 map records total. No optimization
trajectory or target-hit claim from these constructed cases.
For the random map in 3D, E[u_after]=||u||*n/2; record predicted mean and sample SE.

## Population protocol

- Reuse E06 initialization, objectives and all fixed physical scales: four unit
  particles, initial total kinetic=0.6, initial total momentum approximately zero,
  beta=.995, h=.05, gravity=.5, height_scale=1.
- Cases: quadratic, tilted double well, scaled Rosenbrock; all are 2D parameter
  problems plus height. Initial positions are generated using rho_max=.24 for
  every arm, making initial states shared across radii.
- Formal seeds 500--519; rho=.12/.24, frozen without new search. Independent arm
  is run once and shared across radii: 3 cases*20 seeds*(1+2*2)=300 new runs.
- Preflight seed 490, 80 steps, same three cases/five arms: 15 development runs.
- Formal maximum 500 observation steps/time 25; **actual shared query cap 6,000**
  from the start. Every value/gradient/analytic path costs one and is separated.
  Particle-pair tests cap=100,000 and are also logged. This is a cap on query
  accounting, not a wall-time equality claim.
- Score last complete observation before budget exhaustion, retaining all spent
  queries. No interpolation, unpaid future observations, or resuming a failed step.
- Initial known minimum and basin boundary are setup/scoring only, not map inputs.
  Target=min+0.1*(initial population best-min), and left basin for double well.
  Report historical hit, final hit, current loss, archived best, event counts,
  incoming/outgoing geometry, actual energy transfer, costs and runtime.
- Numerical/contact failures remain in all denominators; no-event runs remain.
  Budget stops are valid partial trajectories, not numerical failures. Report
  valid-only gap summaries and unavailable paired comparisons explicitly.
- Primary comparisons: elastic-independent and elastic-random, per case/rho.
  Report paired improvement/worse/tie (tolerance1e-6), target counts and complete
  per-seed data. No cross-case p-value or universal superiority claim.

## Verification and preservation

Test momentum/K conservation, outgoing direction, null V case, analytic transfer,
uniform-hemisphere mean, 1D map equivalence, seed reproducibility, unchanged inputs,
and original-solver equivalence including query counts. Run the full suite before
formal data. Snapshot exact code/tests/spec/launch command and environment before
the batch, store all trajectories/events and any uncommitted map attempts.
Audit raw records independently and update experiment log/registry after completion.
