# E20 — isolated natural-contact replay and numerical sensitivity

User approved the proposed single-event paired replay and preliminary Rosenbrock
sensitivity check. Parent E19 showed exploration benefits without unique normal
geometry gains; aggregate physical features did not beat a clock control. This
protocol is fixed before E20 results. Existing E00–E19 files remain unchanged.

## Questions and scope

1. Conditional on an elastic population's naturally encountered contact, does its
   outgoing direction beat a linear-momentum/kinetic-energy matched random map?
2. Do contact-direction features predict that marginal value beyond magnitude
   and elapsed time on new seeds?
3. Is E19 independent Rosenbrock protected-particle drift sensitive to time step
   and contact tolerances? This diagnoses the solver, not collision efficacy.

These are synthetic 2D mechanism experiments, not a new online policy, general
harm guarantee, high-dimensional ranking or neural-network training. A single
event effect need not explain a full interacting population's performance.

## Source trajectories and selection

Unchanged E19 physics/init: four unit particles, K=.6, rho=.24, h=.05, g=.5,
original ramp. Elastic all-on sources only, cap6000, max500 committed steps,
pair-test cap100000. Development seeds1200–1209; confirmation1240–1259; all
three original terrains. Preflight1190, source cap1800/max120steps.

Take the first up to three COMMITTED pair events of every source, in saved order,
without selecting on outcome, impulse direction or transfer. Sources with no
contact, failed sources and unavailable contexts are reported, not replaced.
All source trajectories are paid once and archived in full, including unused
later states and unsuccessful partial work. Per-seed contexts are correlated.

Reconstruct the full incoming population from the preceding committed state,
saved drag beta, ballistic propagation and earlier same-step event maps. Check
all event inputs and the reconstructed step endpoint against the saved source
to1e-9. This is floating-tolerance reconstruction, not claimed bitwise recovery
of the old source; all counterfactual branches share exactly the same saved new
context. Only already committed events are used. Save source event ordinal,
step, absolute time, incoming state and reconstruction error.

## Six branches per contact

One elastic map, four independent isotropic outgoing relative-velocity draws
in the outward hemisphere, and one zero-impulse control. Fixed RNG seeds derive
from batch seed, case and event ordinal; never reuse future source RNG or state.
Random maps conserve total linear P/K but generally not angular momentum; they
are diagnostic ablations, not complete rigid-body laws. Save all individual maps.

After this instantaneous map all six branches disable ALL binary interactions.
Keep terrain contacts, gravity and original absolute ramp clock. Turning off
hard-core interactions is an explicit external model ablation with no stored
pair potential; particles can pass each other. No velocity reset, extra supply
or artificial wall reflection is introduced.

The context is already damped within its source step. Finish the remaining
fraction WITHOUT another drag kick, then resume drag at the next original grid
boundary. If contact is at a boundary, start the next full step. Follow at most
100 subsequent observation boundaries, roughly5 physical time units. Save
10/50/100-boundary outcomes; within each context those are common-time views.
Each branch cap2000 charged queries, including6 shared context feature probes
(4values/2gradients),4 actual initialization/revalidation values, all path,
gradient and value calls. Shared6 are actually executed ONCE per context; branch
counters reserve them as an explicitly derived charge. Source full-run costs
are separately included in total actual cost, not claimed free prefixes.

Primary effect at additional cap1000: last complete observed pair-minimum loss
under that charged cap, normalized by original source initial-best minus known
minimum. V_geometry = mean(random pair loss) - elastic pair loss;
V_collision = zero-impulse pair loss - elastic pair loss. Also report full-group
current minimum, archived best, pair/group maximum worsening, final/historical
target and basin crossings separately. Known answers are scoring only, never
features or intervention choices. Each branch records its actual elapsed time;
common-time views separate geometry from differing contact-query costs.

Budget termination is retained, not numerical failure. Unavailable primary
observations and numerical/contact failures remain explicit; do not impute them
as successful runs. All four random replicas must be available for the primary
mean. No target-based early stop. Four worker processes, numerical threads1;
mechanism data cap about3.8million charged queries, actual shared cost reported.
No hardware speed superiority claim.

## Past/current features and fixed prediction

Shared probes observe the incoming four losses and two colliding gradients.
Magnitude/clock baseline: source_time, scaled current pair loss, pair kinetic,
relative speed, center speed, elastic impulse magnitude, absolute single-body
energy transfer and both gradient norms. Direction augmentation: cosine of the
elastic impulse with each gradient and each incoming velocity, and contact
normal with each gradient. Near-zero denominators produce0 plus two explicit
gradient-zero flags. Scale loss by max(1,abs(source initial-best)); no minima,
basin identities, random outcomes or later losses in features. Elastic impulse
is computable from the present incoming state, not a future trajectory label.

Fixed ridge lambda1, intercept unpenalized, development-only standardization.
Fit only V_geometry at cap1000. Archive model, training data and source hashes
before reading confirmation outcomes. Confirmation is generated after freeze;
no re-fitting/reselection there. Report MAE, non-tie sign accuracy(1e-6), training
majority baseline, per-case and per-seed errors; never treat contact windows or
random replicas as independent seeds. Promising predictive criterion: at least
5% held-out MAE reduction AND improvement in at least14/20 seed groups. This
is a screening rule, not a statistical theorem. Geometry evidence is reported
as paired better/worse/tie and effect sizes by terrain, without pooled universal
superiority claims. Original E19 seeds are not fresh confirmation.

## Numerical sensitivity, separate batch

E19 stress Rosenbrock seeds880–882, independent particles only, same original
good-parameter/height-reserve initialization. Full fixed physical time10, no
target stop; h=.05/.025/.0125; original tolerances(gap1e-9,time/normal1e-11)
versus tight(gap1e-11,time/normal1e-13), six settings×3seeds=18 executions.
Ramp gamma(t) is identical in physical time, beta=exp(-gamma_mid*h), not a
fixed per-step .995 when h changes. Cap100000queries/run; max800steps. Record
actual time, target-particle current/max absolute loss increase, initial
clearance, heat/energy/P residuals, terrain impulses and all failures. Compare
at common saved physical times, no invented interpolation. Finite-time dynamics
can amplify tiny perturbations; sensitivity alone does not establish the cause
or a converged resting-contact law. No default parameters changed from results.

## Preservation and verification

Save protocol and implementation plan first. Archive exact sources/tests/spec,
commands/environment and initial states before each fresh batch. Raw source and
replay traces/events/maps/paid unsuccessful work remain available. Unit tests
cover source reconstruction, impulse P/K, no second kick on partial resume,
absolute ramp clock, overlap-legal ghost control, shared-vs-actual charge and
past-only features. Independent saved-state audit verifies physics and labels.
Record preflights and failed engineering runs. Append findings and limits to
docs/experiment_log.md; update registry/inventory with source executions,
counterfactual branches and derived comparisons separate. No git repo: source
archives provide restoration. New protection algorithm and higher-dimensional
tests are staged follow-ups, conditional on this diagnostic evidence.
