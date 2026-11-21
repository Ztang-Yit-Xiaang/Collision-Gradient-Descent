# E21 — empirical stability evidence before proofs

User explicitly asked to continue and obtain concrete findings before math
proofs. Parent E20 found late Rosenbrock protected-particle drift even without
binary collisions; shrinking h and tightening tested tolerances did not remove
it. This protocol fixes empirical tests before new outcomes. No theorem or new
optimization controller is being claimed or implemented.

## Questions

1. Does the good particle drift alone, or only under a four-particle shared event
   scheduler? Do explicit, reproducible tiny perturbations overwhelm scheduling
   roundoff and reproduce the phenomenon earlier?
2. Does lowering its available vertical potential stabilize the current good
   position, without resetting a moving state or changing the collision law?
3. Can a simple parabola reproduce a transition from bounded to amplified tiny
   deviations, across curvature/height scales and two finite-difference sizes?

## R: original Rosenbrock, fixed-time controlled trajectories

Frozen CoupledRosenbrock(2), rotation5239, unit masses, g=.5, original ramp,
default contact tolerances, no binary interactions. Full time25, h=.05/.0125,
cap100000solver queries/run, max2000steps, no target stopping. Setup and scoring
calls separate. Initial good point and spectator clouds are taken from exact
E20 time25 diagnostic initial-state files880–882; save parent file hashes.

Estimate the2×2 Hessian at the known synthetic optimum with four central
gradient probes at step1e-5, plus one baseline gradient and value, symmetrize,
save raw gradients/eigenvectors/eigenvalues.
This is diagnostic setup, not online access to an unknown optimum. Fix signs of
eigenvectors for reproducibility. No new directions selected from outcomes.

Single-particle grid: clearance H=.002/.02/.08/.32/H_original(about.83676),
both h, perturbations zero,+1e-8 and-1e-8 along the larger-curvature direction,
and+1e-8 along the smaller-curvature direction:40 trajectories. Start all at
zero velocity and z=f(perturbed x)+H. Initial energy changes across H are
intentional setup conditions, not unrecorded work injected midtrajectory.

Four-particle controls: original H only, same both h, zero/+high/-high
perturbations, original three spectator clouds:18 trajectories. Only the good
particle's x/z are varied; spectators retained exactly, collisions=False.
Single original-H runs execute once, reused for pairing, not three fake
independent seeds. Total58 trajectories, all development/mechanism checks,
not fresh efficacy confirmation. Seeds change spectators, not the good point.

Metrics: current/max absolute loss rise, current/max displacement from optimum,
first time loss rise>1e-4, current/best loss separately, heat/E/P budgets. Derived
comparisons on common.05 saved times: single-vs-joint position/velocity
discrepancy; perturbed-vs-unperturbed deviation relative to1e-8, by height and
direction. Raw finer-grid maxima retained separately. Do not equate few/no
threshold violations with a proof of stability or a general harm guarantee.

## P: simple bowl, directly measured bounce return map

One parameter plus height, f(x)=kappa*x²/2. Gravity.5, mass1, fixed stationary
elastic terrain, beta=1/no drag/no binary interaction. This separate model
isolates terrain geometry; it does not pretend to replicate E20's damped clock.

Fix kappa=1/4/16 and dimensionless products r=kappa*H in
.02/.08/.2/.35/.45/.49/.51/.55/.7/1. H=r/kappa. The proposed product is a
dimensional modeling hypothesis, not an already established stability theorem.
No fitted or asserted threshold. Evaluate the same prescribed products across
scales; kappa16 is a cross-scale check, not a statistical held-out sample.

Start immediately above an outgoing contact at z=f(x), with horizontal velocity
vx, and positive vz chosen so total mechanical energy is exactly g*H. The ideal
vertical reference has x=vx=0. Follow the existing analytic ballistic contact
solver to the next incoming hit and apply its original specular reflection.
No observation-step kicks, arbitrary reflection triggers or approximate Euler
flights. Bound each flight by max(1,8sqrt(H/g)); no-hit/failures retained.

For each of30 cells execute one zero-reference return and central +/− probes
in x and vx at eps1e-6 and1e-7:9 one-return executions/cell=270. Construct a
numerical2×2 return sensitivity matrix from measured outputs(x,vx). Save every
input/output, finite differences, eigenvalues and spectral radius. Compare two
probe sizes: matrix-relative difference<1e-3 and radius difference<1e-3 is the
predefined numerical consistency screen. Radius>1.01 is a local amplification
signal, <=1.0001 is a non-expanding numerical screen; intermediate cases remain
ambiguous. Neither screen is a proof or a global nonlinear guarantee.

Independent finite-perturbation rollouts: kappa1/4/16, r=.2/.49/.51/.7, initial
x=0/+1e-7/-1e-7,vx0, same fixed energy.36 trajectories,30 actual incoming
impacts each. Record current/max loss, x/vx and phase norm
sqrt(x²+(tau0*vx)²), tau0=2sqrt(2H/g), amplification relative to initial|x| when
nonzero. Zero references are not divided by zero. Report actual bounded/growing
behavior, symmetry, energy and external wall/gravity impulses; no fitted
controller or threshold picked from these same outcomes.

## Budgets, preflight, preservation

Preflight in a fresh directory: R original-H,h.05,time2, single/joint880,
zero/+high=4 trajectories; P kappa1,r=.2/.7, same two probe sizes=18 one-return
executions and zero/+/- rollouts5impacts=6 trajectories. Keep separate from
formal data. Main R cap5.8million queries, P cap2000queries per return probe or
rollout (conservative614000combined); actual setup and solver calls recorded.
Four worker processes, numerical threads1, no hardware speed comparison.

Freeze protocol/source/initials/commands/environment before each fresh batch.
Use original models unchanged; add only diagnostic helpers and tests. Test
fixed-energy launch, exact vertical reference, reflection energy/P balance,
mirror symmetry, valid height/perturbation initialization, shared spectator
identity and finite-difference input accounting. Run full suite and audit saved
physics before analyzing results. Preserve failures and unreturned partial work
explicitly. No tuning, no high-dimensional/NanoGPT ranking and no new online
protection method. Update experiment log, registry and inventory; one-return
probes, trajectories, derived matrices and reused pairing views count separately.
