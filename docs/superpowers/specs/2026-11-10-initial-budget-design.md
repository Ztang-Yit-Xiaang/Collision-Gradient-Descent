# E24 — initial height and momentum allocation at matched total E/P

User approved this empirical follow-up on 2026-11-10. Parent E23 showed moving
incumbents can recover with capped height but still exceed the original harm
threshold. Fixed low height was also effective, and the curvature fee had no
identified value at constant maximum curvature8. Isolate those two issues.

Use a paired2×4 initialization factorial, not an online controller. Alternatives
were a strict tiny total-per-body-energy cap (likely changes contact/query cost
drastically) or adding actual binary collisions (mixes a new mechanism). Defer
both until momentum allocation and inexpensive height controls are understood.

## Frozen setup

Reuse E22 rotated terrain f(r)=.25(r0²−1)²+.05r0+.5*k*r1², k=1/8/32;
original seeded rotations and norm.001 four-point clouds; good_present and
bad_only. Reuse E23 controlled initial velocities with particle0 K=.001,
total K=.6 and total linear P=0. Current paid values select j=argmin f(x_i)
in ALL methods; no global-well label used. In bad_only j need not be particle0.
Known minima/saddle serve setup/scoring only. Same input cloud/velocities for
all methods and stiffnesses per seed. No new terrain family/high-dimensional
or neural ranking. Independent unit point particles, no binary interactions.

## Eight fixed policies

Height mode high: all H=.8. curve: selected j gets unchanged E22
min(.8,.2/max(lambda_j,1e-6)), others .8+(.8-H_j)/3. fixed_ref:
selected H=.025 (E23's curvature8 cap), same redistribution. fixed_safe:
selected H=.00625 (E22's curvature32 cap), same redistribution. Both constants
come from past results, not E24 outcomes. All sum H=3.2, clearance potential1.6.

Each height has keep and transfer velocity modes. keep leaves input velocities
unchanged. transfer sets selected v_j'=0; for other i let w_i=v_i+v_j/3 and
v_i'=w_i*sqrt(sum_l||v_l||² / sum_{i!=j}||w_i||²). Inputs have zero total P;
this preserves K and linear P within numerical tolerance and transfers the
selected kinetic budget to the others, without extra oracle queries. Require
finite4×2 zero-P input; if spectator relative spread degenerates with nonzero
K, raise/record an error instead of inventing directions or reseeding. Zero-K
input maps to zero. Vertical velocity stays0. Save the actual initial K/P and
selected/spectator K before/after, and angular momentum change at the SAME
post-height positions. This map does NOT in general conserve angular momentum.

This is a matched family of chosen initial states, NOT a dynamically realized
internal collision or free online momentum reset. No event/impulse is inserted
after t0. If deployed in motion it would need a separately specified physical
interaction or external work/impulse/torque account. All eight initial total
E and linear P match, but per-body E, angular momentum, later heat/contact
cost and elapsed time need not match. high_transfer tests whether removing
selected velocity alone suffices despite large height.

## Cost, seeds and stopping

New seeds1600–1619, two conditions×three stiffnesses×eight policies×20=
960 formal trajectories,120 shared contexts,20 independent seed groups.
Engineering1590, same48 trajectories/6contexts, query1200/max80steps; excluded
from efficacy and no efficacy-based tuning. Formal cap6000 from initialization,
max500steps, g=.5/h=.05/original ramp. Four processes, numeric threads1.
No target early stopping, fitting/tuning allowance0; no extra height/velocity
search or new policies after outcomes. Preserve failures and last committed
states, including charged unfinished steps. Fresh output roots/source archives.

Shared probe20 calls (4values+16gradients) actually executed once/context.
Only the TWO curve policies pay20 prefix; SIX others pay4. Each pays actual
initialization revalidation and subsequent solver calls. Thus actual shared
execution saves44 calls/context versus independent policy charges, NOT the
48 of the old five-policy experiments. Three scoring reference values/context
separate. transfer itself uses existing values and velocities, oracle cost0.
Save exact source/data hashes, restorable code archives, commands/environment,
per-run costs/outcomes, initial allocations and physical ledgers.

## Frozen metrics and comparisons

Reuse E23 scores: good particle0 ever loss rise>1e-4, continuous max/final rise,
particle0 final target f<=global min+1e-4 and left of saddle, group current and
historical targets separately. Bad-only target unchanged, left crossing,
historical/final success and seed-paired switches. All failures in denominators.
Original20-seed threshold not relaxed; transient movement alone is not a
numerical-instability diagnosis. Report all eight methods in all stiffnesses.

Primary causal contrast is transfer vs keep at the SAME height mode. Main
descriptive joint screen: curve_transfer vs curve_keep reduces good-particle
harm>=.25 in>=2/3 stiffnesses; no stiffness loses>2/20 bad-only final hits;
no solver failure; baseline curve_keep has>=5/20 bad-only final hits in EVERY
stiffness for joint interpretation. Report gates separately, no statistical
noninferiority claim. Fixed controls are NOT selected on these outcomes:
compare curve_transfer against BOTH fixed_ref_transfer and fixed_safe_transfer,
including paid fee, continuous effects and final success, without declaring
curvature necessary solely because .025 fails at32.

Secondary common-time view uses minimum last committed .05 grid time across
all8 methods within a context, no interpolation/new executions. At k8 report
how close curve/ref initial heights/states are and their16-call prefix fee
difference; FD roundoff/chaotic amplification mean approximate initialization
equivalence is not a promise of identical floating-point trajectories.

## Verification and interpretation

Unit tests for matched E/K/P, actual selected index, velocity transfer,
degenerate rejection, angular change disclosure, correct4/20 prefix, source
policy equivalence and k8 same-height control. Full tests then raw engineering
audit before formal launch. Independently reconstruct original inputs, finite
probes, ALL eight allocations and velocity maps, E/P/angular changes, terrain
reflections, per-step heat/gravity/wall momentum, query costs and raw/new metrics
before inference. Export committed-trajectory heat/work residuals without new
queries. Preserve engineering failures, update log/registry/inventory.

Self-review: no unresolved choices; all8 E/P match; linear P distinct from
angular P;44-call accounting checked; no good-index leak; independent units20;
the test is initialization ablation, not an intervention/optimizer claim.
No git repository: source archives provide restorable versions. Prior user
approval covers this bounded follow-up; no redundant approval loop.
