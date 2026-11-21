# E23 — frozen allocation under moving incumbents and changed barriers

User approved the proposed next empirical validation on 2026-11-10. Parent E22
kinetic follow-up found matched-initial-E reserve_best harm0/20 vs high19/20,
20/20,20/20, with bad-only final16/10/7 vs17/10/8. That incumbent was stationary;
the synthetic terrain family and five initialization rules now remain fixed
except for the explicitly varied longitudinal scale and initial momentum.
No theorem, online intervention, binary collision, neural training or tuning.

## Question and alternatives

Does the existing height-only rule protect an already-moving good point, and
does the observed protection/escape tradeoff extend to changed barriers?
Continue with a paired factorial mechanism/robustness experiment. Increasing
seeds on the original stationary terrain would not address either limitation;
adding binary contacts now would mix them with a third cause. Defer binary
contacts and a redesigned total-energy controller until these results exist.

## Fixed terrain and motion

Terrain in rotated coordinates: f(r)=a*(.25*(r0²-1)²+.05*r0)+.5*8*r1²,
a=.5/1/2, transverse stiffness8. Longitudinal scaling changes barrier HEIGHT
and longitudinal curvature together; do not claim they are causally isolated.
Use the same E22 seed-based rotations, roots and norm.001 clouds, two conditions
good_present and bad_only, four unit independent point particles. Roots do not
move under this positive scaling; their known values are setup/scoring only.

Three prespecified particle0 horizontal kinetic energies e=0,1e-5,1e-3. Each
seed has an isotropic world-coordinate unit direction d, independent of the
landscape. Let u=sqrt(2e)*d and w be the original E22 three-explorer velocities
(sum w=0, kinetic.6). Set v0=u and v_i=sqrt(1-4e/(3*.6))*w_i-u/3 for i=1..3.
This keeps total horizontal K=.6, total P=0 and the specified particle0 K=e.
Vertical velocities0. Share directions/velocities/clouds across all policies,
conditions and scales within each seed; e=0 reproduces E22 moving-explorer
construction. K values bracket the existing loss-harm threshold's energy
scale, not a search for a successful velocity. No outcome-based selection.

Moving particles may have physically normal uphill excursions. Preserve the
original ever-rise>1e-4 outcome rather than silently relaxing it. Also report
continuous max/final rise and the NEW prespecified particle0 final target
(f<=global minimum+1e-4 and left of saddle) to distinguish temporary motion,
later recovery and amplification. A high harm rate alone is not proof of
numerical instability. No claim that height-cap bounds total moving energy.

## Rules, budget and sample units

Reuse E22 allocate, measured_curvatures, independent-only solver and score
unchanged: fixed_high, fixed_low, cap_all, reserve_best, reserve_random. The
same .8/.02/.2 constants, four paid gradients per point, no control after t0.
Preserve g=.5, h=.05, ramp, max500 steps, per-policy total query cap6000 from
initialization; no target early stopping. High/reserve initial total E/P match;
low/cap use less energy. Initial K/P match for ALL policies. No guarantee of
matched dissipation or physical runtime. Keep E22 raw outputs immutable.

Fresh formal seeds1500–1519, 3 scales ×3 kinetic levels ×2 conditions ×5
policies ×20 =1800 actual trajectories,360 shared contexts,20 independent seed
groups,10.8million nominal policy queries. Engineer preflight1490 across the
same90 trajectories/18contexts, query1200/max80steps, excluded from efficacy.
Four worker processes, numeric threads1. No fitting, no tuning allowance.
Stop at this factorial batch; no follow-up energy amplitude search in E23.

Charge shared20 probes once/context, each fixed policy pays its necessary4
prefix and each curvature policy20, plus its actual initialization/solver
calls. Reference values3/context separately. Failed partial steps retain all
paid cost and last committed states. Computational environment/commands, input
states, probes, numerical failures, heat/wall/gravity impulses, numerical
residuals, sources and data hashes saved. No runtime external supply/impulses
other than frozen gravity, wall contacts and original damping.

## Inference fixed before outcomes

Primary at equal query termination: per-body good-point max/final rise and
ever-harm, group final/best value, bad-only final/historical targets and left
crossings. Also particle0 final target. Failures stay in denominators. For each
e separately report reserve_best vs high and other baselines across a, using
paired seeds, not independent policies/conditions. Do not pool e to hide failure.
Original descriptive screen per e: harm reduction>=.25 in>=2/3 scales, no
scale loses>2/20 bad-only final hits, no solver failures. Interpretation further
requires high baseline>=5/20 final hits in EVERY scale for that e. Describe
each gate separately; this is not statistical noninferiority or a proof.

Secondary common-time view uses shortest last committed time among five
policies in each context on .05 grid, no interpolation/new trajectories.
Report seed-paired success switches and continuous max-rise contrasts.
Archive/history and final active solutions remain separate. Larger-e harm can
reflect ordinary motion; report it even if it causes the original screen to
fail. No posthoc harm definition or favorable-window substitution.

## Verification and recording

New unique names verified absent; legacy source files unchanged. Test scaled
terrain gradient/path, a=1 equivalence, cloud invariance, exact total K/P and
particle0 K, e=0 equivalence, paid probes and raw scoring. Full tests and
independent raw preflight audit before formal launch, without inspecting
preflight efficacy. Formal audit must reconstruct analytic values/gradients,
heights/E/P, motion recipe, initial references, wall maps, energy/heat and
external momentum, query costs and metrics before inferential summaries.
Preserve actual launch source.zip and final postprocessing source archives.
Maintain experiment_log, registry and inventory including engineering failure.

Design self-review: no placeholders; scale is not barrier-only intervention;
total K is fixed across moving-incumbent conditions; the original harm screen
is retained and recovery separately measured; independent seed count20, not1800.
No git repository is present; restorable source archives replace commits.
