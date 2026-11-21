# E17 — physical dissipation after exploration

User authorized analysis and experiments following E16. This protocol fixes the
next comparison before inspecting new outcomes. Existing code/data stay intact.

## Question and alternatives

Can a predetermined increase in physical drag improve final retention without
losing exploration, and is timing useful beyond a stronger constant drag?
Choose bulk velocity damping first. Alternatives deferred: restitution<1 changes
impact physics and needs resting-contact treatment; loss-triggered stopping adds
an information-based controller. Neither is part of this batch.

All instantaneous pair/wall collisions remain elastic (the random scattering arm
is the existing linear-momentum/K-conserving, generally noncentral ablation).
At each observation step apply v <- exp(-gamma*h)*v to every full velocity.
The removed kinetic energy Q=(1-beta^2)*K is heat to the environment; the drag
impulse is external to the particle subsystem. No supply and no resets.
This is the existing split-step drag discretization, not an exact continuous-drag
ballistic trajectory. Time-dependent gamma does not itself inject mechanical work.

## Frozen schedules

h=.05, nominal horizon T=25, gamma0=-log(.995)/h, gamma1=.5.
Three schedules, common across all particle methods:

1. weak: gamma=gamma0 (exact E16 beta=.995).
2. constant: gamma=(gamma0+gamma1)/2.
3. ramp: gamma=gamma0+(gamma1-gamma0)*clip((t+h/2)/T,0,1).

The midpoint rule makes constant and ramp integrated drag equal over the full
500-step nominal horizon. Under the true query cap they can stop at different
times; actual integrated gamma and dissipated Q must be reported. Equal nominal
drag is NOT equal actual dissipated energy or equal CPU time.
No parameters selected by new outcome search. These are an initial fixed screen,
not a convergence theorem or a claim of universally optimal annealing.

## Analysis and experiment

First derive hit/exit counts, final K, removed energy and late target occupancy
from all E16 raw traces; this is a derived view, no new trajectories. Known
minima/basins/targets remain scoring-only, never schedule inputs.

Fresh E17 seeds600–609, three E16 terrains, four unit particles, initial K=.6,
rho=.24 for shared initialization and colliding methods; other scales unchanged.
Methods independent/elastic/random × schedules weak/constant/ramp ×3cases×10seeds
=270 unique new trajectories. One rho is retained to study dissipation, not
to claim it was independently selected as the optimal geometry.
Preflight seed590, all27 combinations,80steps. Full tests before formal runs.
Actual shared query cap6000 from the start, pair-test cap100000, max500steps.
Four disjoint seed shards may run concurrently with numeric threads=1.
All results, including errors/no-contact/budget stops, remain in denominators.
No stopping rule based on loss, K, known targets or best checkpoint is added.

Primary: final relative gap and final target hit; exploratory secondary: historical
hit, first hit, subsequent exits, target occupancy over the last20% of observed
intervals (population any-particle; discrete observations, not continuous proof),
final K/clearance potential, integrated gamma, Q, events and query/runtime costs.
Report schedule-vs-weak and ramp-vs-constant paired differences per method/case,
and elastic-vs-independent/random within schedule; tolerance1e-6 on final gap.
No pooled cross-terrain significance or positive selection after the results.
Failed runs cannot count as successes; gap summaries are valid-only with counts.
This is not a near-rest good-minimum safety experiment.

## Implementation and verification

New schedule subclass delegates the unchanged scattering step, replacing only
the immutable config beta per step. Increment time only after committed steps.
Tests: nominal integrated drag, monotonic ramp, weak exact state/event/query
equivalence, discrete Q and momentum balances, step counter on budget failure,
metrics hit/exit semantics. Independent saved-data audit reconstructs per-step
beta, Q and total energy/momentum ledgers and final scoring. Preserve exact
sources/tests/spec/commands/environment before launching each batch.
Snapshot the new spec separately alongside the existing runner snapshot.
Append E17 results/reasons and update registry/inventory. Never overwrite E16.

## Literature scope

Su/Boyd/Candes (2016) provides an inertial-ODE perspective on acceleration,
not evidence for this ramp or for our hybrid collision model:
https://stanford.edu/~boyd/papers/ode_nest_grad.html (author summary read).
Cherubini/Metafune/Paparella, arXiv0706.1978 studies a deformable dissipative
bouncing ball avoiding inelastic collapse, illustrating model dependence of
settling, not proving collapse in our elastic-impact bulk-drag solver:
https://arxiv.org/abs/0706.1978 (abstract read).

Self-review: schedules, budgets, splits, scoring, failures, accounting and limits
are specified; no tuning or placeholder remains. No git repository is available;
source archives provide preservation. User's authorization covers this direction.
