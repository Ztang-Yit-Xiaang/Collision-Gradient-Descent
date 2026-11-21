# E19 — timing curve, collision-value data and protection stress test

User approved the staged timing sweep, data for physical marginal-value signals,
good-solution protection and equal-budget baselines. Existing E00--E18 artifacts
and model code remain intact. This protocol fixes choices before new outcomes.

## Efficacy and data split

Three unchanged 2D terrains, four particles, E18 initialization, h=.05, g=.5,
rho=.24, initial K=.6, unchanged E17 ramp and global500 physical-step limit.
Total query cap6000, pair-test cap100000, original E18 scoring. One call per
value/gradient/analytic path, wall-time and pair-test work also recorded.
Development seeds800--809; untouched confirmation840--859; preflight790.

Fractions0,.25,.5,.75,1 specify cost thresholds to disable binary interactions
at a committed observation. No states/velocities/RNG/drag clocks are reset.
Fraction0 is a single independent run shared across pair rules; fraction1 is
an uninterrupted all-on run. For elastic/random, capture .25/.5/.75 committed
checkpoints during one all-on execution and fork an off suffix from each.
Record the four paid revalidation values on forks. Avoid discarded partial
steps at thresholds. A failed/unreached checkpoint remains unavailable/failed.
Methods do not know global minima/basins; these remain setup/scoring only.

Per case/seed: one independent all-run, two interacting all-runs, six off suffixes
=9 execution segments and9 compared policies (shared-prefix full-path views).
No duplicate separate prefix execution. Each conceptual policy pays its all-on
prefix, while actual shared execution cost is counted only once. Main matrix:
270 development policies and540 confirmation policies (90 case/seed groups).
Report fractions for both pair rules; select only elastic's global fraction.

Select a single elastic fraction on development by equal-case mean of mean
log1p(final relative gap); failures get log1p(1e6). Ties1e-12 prefer closest to.5,
then smaller fraction. Save selection+development hashes BEFORE confirmation.
Run/show the full confirmation curve but never reselect based on it. Primary
confirmation contrasts: selected vs.5/0/1 and random at the same chosen fraction.
Final loss/hit and historical best/hit separate; tolerance1e-6 for paired gaps.
No pooled significance/universal superiority claim.

## Cheap physical features and offline labels

At each committed checkpoint save full population/RNG/config/counters, current
and last20 committed-step statistics, pair-event summaries and exact row indices.
Base features: fraction of budget, current observed minimum loss divided by
max(1,abs(initial observed minimum)), and recent loss improvement on that scale.
Additional features: K, clearance potential, center speed, relative velocity RMS,
position dispersion, recent pair count and mean absolute single-particle kinetic
transfer. All arise from paid past observations; no minimum, basin, future loss
or future event count is a feature. Instantaneous total K conservation does not
imply that a transfer is useful for optimization.

Labels are derived comparisons of all-on vs off from the same checkpoint:
V=L_off-L_on, normalized by original initialization gap, at last complete saved
states under an additional1000-query cap; also save a common physical20-step
comparison when available. These are derived budget views, not new executions
or online oracle access. Costs of the full offline trajectories are retained.
For the predictive budget label, reserve the same four-value resume validation
charge on the reused on curve as the off fork actually pays; save both source
and hypothetical charged counts, and the unadjusted budget label separately.
This is derived cost matching, not four actually executed extra on queries.

Fit ridge regression lambda=1, train-standardized features, intercept unpenalized,
on development only, with two feature sets base vs base+physics. No tuning.
Evaluate held-out MAE and sign accuracy outside1e-6 deadband, per terrain/rule
and by seed; report class balance and a train-majority sign baseline. Prediction
is exploratory/offline, NOT an implemented adaptive controller or online saving.

## Equal-budget local baselines

Four independent current parameter vectors, total6000 shared queries, round-robin
updates. GD and Momentum(beta=.9, zero initial buffer), learning rates
.005,.02,.05,.1; tune per terrain/family on the10 development seeds, same
log1p criterion, ties smaller rate. Confirm chosen rates on20 untouched seeds.
Every initialization value, gradient and candidate value is paid; no target stop.
Fixed initial physical K/height reserves are disclosed, not claimed equivalent
to gradient-method momentum. These are not AdamW/NanoGPT rankings.

## Protected-particle stress diagnostic

Separate seeds880--889, three terrains, five elastic fractions; no selection or
tuning from these outcomes. Put particle0 at a known optimum in parameter space,
zero velocity, and particle1 at x0+.26e1 at a common clear height, approaching at
-.4e1. Others have momentum+.4e1 and relative random velocities so total P=0,
K=.6; they start separated using rejected E18 clouds. Common height is
max(f(x0),f(x1))+.02, so this is a good-parameter test with disclosed vertical
reserve, NOT a static resting-contact model. Initial intended pair contact near
t=.05 is constructed; label this a stress test, not unbiased encounter frequency.
Physical forks/all-on/independent use shared initial states.

Report particle0's maximum/final absolute loss rise, fraction ever exceeding
1e-4, group final and archived best separately, and actual contacts involving0.
Do not use near-zero initial-gap normalization for harm. Fraction0 is the
control. Snapshot captures in this condition are not used to train predictors.
150 policy views,150 execution segments; denominator10 per terrain/fraction.

## Checks, limits and preservation

Preflight seed790, cap1200/global80steps, proportional quarter checkpoints,
all3 terrains/rules/fractions; both gradient families at eta.02; same790 safety.
Tests: checkpoint recorder identical to frozen dynamics; incoming/outgoing state
and RNG continuity; legal independent overlap on resume; exact paid counters;
window features contain past only; deterministic selection; ridge train-only
standardization; valid stress initialization and intended approaching pair;
gradient update/buffer and unpaid candidate handling.
Full suite before formal launch, raw energy/momentum/heat and state audit before
reading inference. Archive exact sources/tests/spec/launch/environment, all
traces/events/maps/checkpoints and selection files. Four workers, numeric threads1.
Append results, limitations and reasons to experiment log and update registry.
No changes motivated by held-out efficacy; failures remain in denominators.
No git repo; source archives provide exact preservation. This implements the
approved experiments, not an unapproved new online intervention policy.
