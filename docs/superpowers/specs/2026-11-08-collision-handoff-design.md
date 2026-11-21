# E18 — fixed exploration-to-refinement handoff

User approved the three suffix alternatives, fixed handoff and fresh-seed
replication proposed in chat. Preserve E16/E17 and all frozen code. This design
and self-review precede new formal outcomes. No further approval is needed for
these authorized comparisons. No git repository; preserve exact source archives.

## Questions and alternatives

Does continued contact hinder refinement? Does a better exploration state help
ordinary local optimization? Does E17's ramp effect replicate in new seeds?
Keep the existing drag/impact law, rho=.24, h=.05, g=.5, four unit particles,
initial K=.6, and initialization/scoring. No new heat or reset. Primary suffixes:

1. continue: same population, contacts and E17 ramp.
2. independent: same population and velocities, only disable binary interactions.
3. gd: select the currently lowest observed f particle (stable lowest-index tie),
   refine its current x with deterministic full-gradient descent, fixed eta=.02.

The second arm is an externally changed model/diagnostic ablation, not a natural
cessation of contact. No position/velocity jump or immediate mechanical work is
applied when disabling interactions. The third is a hybrid algorithm, not a
continuous physical trajectory: archive the selected full state unchanged and
record other particles' exported E/P plus the retained E/P. The physical ledger
ends at that interface; GD uses x only. Do not silently describe unused velocity
as a velocity reset, nor claim a physical conservation law for later GD updates.
No best-from-history rewind and no loss-triggered handoff.

## Formal protocol and counting

Corrected formal restart: fresh seeds720--739, three terrains
quadratic/double_well/rosenbrock. The initial700--719 attempt is interrupted and
excluded, as detailed in the engineering amendment below.
Three prefix rules independent/elastic/random, all with unchanged E17 ramp.
For each case/seed/rule, execute a prefix ONCE. At entry to the next observation
step, stop if cumulative real queries >=3000, so the checkpoint is a committed
state and crossing3000 costs are retained. Prefix oracle cap remains6000; no
partial step is knowingly discarded at the handoff. A prefix that fails before
handoff remains a failure for all suffix comparisons; do not resume its last state.
If it fails to reach the threshold by the global500-step horizon, do not invent a
handoff; retain unavailable suffix outcomes. No such sample is silently deleted.

Deep-copy committed method state, random-generator state, oracle counters and
population at handoff. Fork each suffix with total cumulative query cap6000.
The physical suffix initializes with the original runner's paid four-value
validation; log this overhead. Drag step index continues from the prefix, while
local row times/events receive the global handoff offset when comparing paths.
Only500 total physical macrosteps allowed. Pair-test cap100000 includes prefix.
GD commits an update only after its gradient AND new value are paid and finite.
No iteration limit other than remaining queries; an unpaid final candidate is
not an observed result. All selection uses already saved current values.

180 unique prefixes +540 unique suffix executions define540 paired full-pipeline
views. A prefix is not executed again for each view; charge its queries to EACH
conceptual full pipeline, but count actual compute only once. Observations and
branches sharing a seed/prefix are not independent samples.

Separately execute true elastic E17 weak/constant/ramp WITHOUT a handoff on the
same20 seeds/three cases:180 full trajectories. These are exact-protocol new-seed
replications (same parameters), not the split continue branch, whose extra
validation costs can shift its final budget-limited observation.
Formal unique execution segments=180prefix+540suffix+180uninterrupted=900.
Conceptual compared pipelines=540split+180uninterrupted=720, not720 independent
seed samples. Nominal actual query bound3,240,000; actual cost is logged.

Preflight seed690, all cases/rules/suffixes, global80 physical steps,
total cap1200, threshold600;9prefixes+27suffixes+9uninterrupted=45segments.
No tuning or new drag selection. GD eta=.02 is a fixed, untuned local comparator,
not a tuned best GD/SGD/AdamW claim. Known minima/basins are setup/scoring only.
Run four independent worker tasks with numeric threads1. Preserve failures and
unexpected task errors with their saved prefixes/artifacts; never pool preflight.

## Metrics and fair comparisons

Primary: final relative gap (original initialization denominator), final target
hit (E17 target and left-basin condition). Historical best/hit reported separately.
Secondary: prefix quality, which prefix reached the left basin, suffix changes,
physical motion times, K/clearance, Q, exported E/P, event counts and costs.
Failure-inclusive hit denominators and valid-only gap counts are separate.
Target scoring never selects an intervention or adjusts a step size.

Paired suffix contrasts per case/prefix rule, tolerance1e-6; compare prefix rules
under common suffix, especially GD to remove continued-contact confounding.
Report all pairs and unavailable cases. For physical continue/independent, also
compare the exact saved observation at the shorter committed global time, using
the common .05 grid without interpolation, no extra execution or unpaid queries.
GD has iterations, NOT physical time; do not insert it in that time comparison.

Improvement evidence means better final metrics at the same6000 cap with clear
exploration/retention costs, replicated across fresh paired seeds; do not infer
universal superiority or choose a new positive threshold after seeing results.
Report mixed signs and per-terrain effects rather than a pooled p-value.

## Verification and archival

Tests: prefix stopping occurs before a new step and consumes no extra oracle;
unchanged initial state on fork; all branches share original RNG state but then
evolve independently; ramp index and pair-test counts carry over; split continue
matches uninterrupted dynamics on common physical times; explicit validation
overhead; current-best selection and E/P partition; GD update/cost/uncommitted
handling; global-time comparisons. Full suite before formal launch.
Independent raw audit recomputes drag heat, momentum/energy ledgers, source archive,
cost partition, prefix/suffix boundary, final/best scoring, and every GD gradient
step from saved before/gradient/after data. No solver calls needed for the audit.
Archive source/tests/spec/launch/environment, all prefix/suffix traces/events,
uncommitted mapping attempts, retained/removed states and comparison data.
Update experiment log, inventory and registry after completion.

Self-review: seeds, data, budgets, current selection, state continuity, physical
vs hybrid boundary, shared-prefix counting and failure handling are explicit.
The fixed handoff mechanism is already user-authorized; no new adaptive router.

## Engineering amendment before corrected formal outcomes

The initial formal attempt found a resume-validation bug: the base runner calls
initial=True on a continuation and imposed setup separation even on independent
particles that legitimately overlapped during their preceding motion. Stop and
retain that entire attempt under collision_handoff_interrupted; do not pool it.
For resumed states only, disable that setup-only check. Colliding populations
still enforce nonoverlap, and all terrain checks/paid four-value validation stay.
Add forced-overlap independent/colliding regression tests and a full-budget
development preflight seed700 (9prefixes+27suffixes+9replicas=45segments).
The earlier short seed690 preflight stays intact. Restart the formal comparison
on previously unused720--739, without changing drag, GD rate, handoff or metrics.
This is a data-separation amendment, not a performance-tuned algorithm change.
