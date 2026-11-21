# E22 — curvature-aware initialization: protection versus exploration

User approved testing curvature-matched initial energy with both good-solution
retention and escape from a bad well. E21 found a concrete curvature/height
mechanism but did not establish optimizer advantage. Fix this protocol before
E22 outcomes; no proofs, trained classifier or online energy deletion.

## System and cases

Four unit point particles, independent binary interactions OFF, original
gravity.5, h=.05 and absolute-time ramp, max500 committed steps. Terrain
reflections stay natural and elastic. Nearby independent starts may overlap:
the new independent-only subclass bypasses initial pair spacing, never terrain
validation; it rejects binary interactions. This is a terrain-contact energy
allocation study, not a result for interacting rigid balls or NanoGPT.

New rotated2D terrain f(r)=.25(r0²-1)²+.05r0+.5*k*r1², r=Qx, transverse
k=1/8/32. Q is a seed-fixed rotation shared by all policies and stiffnesses.
Two conditions: bad_only, all four near the right local minimum; good_present,
particle0 near the left global minimum and the other three near the right.
Every start receives a seed-fixed random-direction displacement of norm.001,
zero velocity. Same x, rotations and random direction draws across policies;
same clouds across stiffnesses. This larger perturbation/four-start design is
new versus E21, not a paired performance revision of its1e-8 single-point test.
Known roots/minima/saddle define setups/scoring only, never allocation choices.

## Five fixed initialization policies

H is clearance above f(x); all initial velocities0. No changes after t0.

1. fixed_high: H=.8 for every particle, sumH3.2.
2. fixed_low: H=.02 for every particle, sumH.08.
3. cap_all: H_i=min(.8,.2/max(lambda_i,1e-6)). lambda_i is the largest eigenvalue
   of a symmetrized central-FD Hessian at that current x, step1e-5, four paid
   gradients/particle. Signed eigenvalues and raw probes saved. Constant.2 is
   selected from E21's conservative empirical regime, not tuned on E22.
4. reserve_best: choose current lowest paid value j, H_j as cap_all; each of the
   other three gets .8+(.8-H_j)/3. Same sumH3.2 and total initial E/P as high.
5. reserve_random: identical redistribution, but a seed-preassigned uniform
   index j independent of values/curvature/outcomes. Same seed choice across
   conditions/stiffnesses; no balancing/reselection after seeing results.

Policies compare energy reduction and equal-total-energy allocation separately.
Only reserve_best/random vs fixed_high have the same initial E. cap_all/fixed_low
use less energy and cannot be advertised as a free same-energy improvement.
Protecting an initial good archive is trivial here; active-particle/current
state and query opportunity cost remain the questions.

## Query and data accounting

Each policy's real cap6000 from initialization, no target stopping. All need
four initial values to place z=f(x)+H (charge4). Curvature policies additionally
pay16 gradients (charge20total). Raw context values/curvatures execute once per
case/seed and are shared for efficiency; fixed policies get no free curvature
decision or artificial16-call charge. Each branch records its own derived
prefix charge plus actual initialization revalidation4 and all later calls.
Total actual cost counts shared20once, branch actual calls excluding reused
prefix, and separately three reference/setup values per context. Failed partial
steps retain paid counts and last committed state; unavailable internals marked.

Formal seeds1400–1419, six conditions, five policies=600 actual trajectories,
120 shared contexts,20 independent seed groups, nominal3.6million charged
queries. No tuning or model fit. Preflight1390, cap1200/max80steps=30 trajectories
in a fresh directory, excluded from formal efficacy. Four worker processes,
numeric threads1; exact sources/config/commands/runtime/raw input hashes saved.

## Outcomes fixed before results

Primary at actual query termination: current group minimum/final target,
archived best/historical target, any-left-basin and first crossing separately.
Bad_only target is left basin and f<=f_global+.1*(initial_group_min-f_global).
Good_present tracks particle0's current/final loss rise and maximum rise above
its actual initial value, harm threshold1e-4, plus absolute distance from the
known optimum. Save reference-to-minimum rises too; initial perturbed points
are not exactly stationary optima. Report group and individual outcomes, and
never use a preserved historical best as proof of active protection or benefit.

Same-time secondary uses the minimum last committed physical time among the
five policies of a context on the common.05 grid, without interpolation or new
execution. Query/contact cost can change elapsed time; report both views.
Initial per-particle H, E/P, curvature*H, actual heat/impulses and final per-body
states are retained, no hidden external supply or reset during trajectories.

Promising joint screen (not a statistical theorem): reserve_best vs high reduces
good-particle ever-harm by at least25 percentage points in at least2/3 stiffness
settings AND bad_only final successes are not worse by more than2/20 seeds in
any stiffness. Also verify equal E/P and correct query charges. Random-reserve
control tests selection of the current best versus a generic low-energy member.
All failures retained in denominators, paired effects by seed, no treating600
paths/shared contexts as600 independent samples. Whole-algorithm/AdamW ranking,
binary-collision protection, general stability and higher-dimensional claims
are explicitly outside this batch.

## Checks and records

Tests cover rotated terrain gradients/path geometry, paid curvature probes,
height caps, same-energy redistribution and zero initial P, independent overlaps
without permitting terrain penetration, seed-only random index and prefix
accounting. Full suite and raw preflight audit before formal launch. Independently
audit saved initial allocation, E/P/heat/wall/gravity ledgers and raw metrics
before inference. Append all outcomes/failures/reasons to experiment log, registry
and inventory. Preserve legacy files; verified unused new filenames before edits.
