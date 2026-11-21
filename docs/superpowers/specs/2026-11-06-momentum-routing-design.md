# Cheap momentum-state routing: frozen smoke experiment

Approved direction: retain Momentum SGD, select continue / dissipate / supply a
bounded impulse using already observed training quantities. This is an optimizer
control rule inspired by mechanics, not a simulation of a rigid-body collision.
It does not claim original novelty or exact Hamiltonian conservation.

## Rules fixed before running the new experiments

- Five Momentum arms: `none`, `restart`, `damp`, `random`, `adaptive`.
  Restart and damp only handle overshoot; random only supplies pulses; adaptive
  combines damping and bounded feedback-driven pulses. Plain Adam is a synthetic
  reference; plain AdamW is the NanoGPT reference. All Momentum arms use the same
  previously selected learning rate and momentum coefficient 0.9.
- Decision occurs after the current gradient is available, before the update.
  The router has scalar inputs only; it cannot query an objective or model.
- Overshoot warning: inherited velocity and current effective gradient cosine
  exceeds 0.1, velocity norm exceeds twice the fresh gradient step norm, and
  velocity norm / (1 + parameter norm) exceeds 0.001. Require two warnings among
  the last three updates. This is a warning, not proof that a direction is bad.
  Restart sets inherited velocity to zero; damping multiplies it by 0.5.
  Wait 10 updates between dissipative actions, at most 20 actions per run.
- Stagnation: two consecutive quiet windows, each with mean relative movement
  <= 0.001 and insufficient observed progress. Windows contain 10 synthetic
  updates or 20 training updates. Synthetic progress is improvement of running
  best loss / (1 + abs(previous best)); tolerance 0.0001. NanoGPT compares adjacent
  training-loss window means: tolerance 0.001 + twice the estimated standard
  error / (1 + abs(previous mean)). The latter is a heuristic under correlated
  minibatch noise, not a statistical confidence guarantee.
- At most two pulses. Synthetic radii are (0.05, 0.15)*(1+initial parameter norm);
  NanoGPT radii are (0.001, 0.003)*(1+initial parameter norm). Energy supplied is
  E = radius^2/2 and the cap is twice the larger pulse energy. Damping does not
  replenish the cap. Direction is normalized Gaussian in parameter space.
- Random uses the larger radius each time. Adaptive first uses the smaller
  radius. After two real windows, if observed progress remains insufficient,
  the next pulse uses the larger radius and opposite previous direction. There
  are no counterfactual rollouts; observed improvement is not a causal estimate.
- For two windows after a pulse, active damping/restart is suspended to allow
  short uphill travel. Pulse cooldown is four windows. Natural gradient force
  and momentum decay continue. Momentum and its external component decay by
  the same coefficient, so their sum obeys the ordinary Momentum recurrence.
- Known optimum, basin identity, validation and test losses are scoring-only.
  Good-minimum interference is measured; the controller cannot certify that
  a stationary point is a bad minimum.

## Physics and accounting

Mass is one, velocity is parameter displacement per update, K=||v||^2/2.
Pulse amplitude along unit d is -v.d + sqrt((v.d)^2+2E), making delta K=E.
Scaling velocity by a gives exact delta K=(a^2-1)K and impulse (a-1)v.
Modify both the native momentum buffer and the inherited external component.

For deterministic, zero-weight-decay Momentum, v_next=mu*v-eta*g. Record
gradient impulse, natural damping impulse, actuator impulse separately.
H=K+eta*f. Natural damping is (mu^2-1)K. Record conservative gradient work
-eta*(f_next-f), and Euler discretization residual rather than pretending this
integrator conserves H. Check the algebraic energy and momentum balances.
Stochastic training loss changes are not conservative work: NanoGPT records
actuator K changes, motion and timing, without claiming global energy balance.

## Fixed experiments and validation

- Preserve previous source and output files; implement separate modules.
- Reuse `output/optimizer_comparison/selected.json` and
  `output/nanogpt_comparison/selected.json`, no new tuning.
- Synthetic: the six previous cases, seeds 300--309, six methods, 360 runs,
  exactly 1,500 objective/gradient calls per run; no planning calls. Report
  current and best loss, basin/target attainment, supply, dissipation and time.
- NanoGPT: licensed local official model, unchanged data split, seeds 300--302,
  six methods, 400 actual updates each, 1,200 training forward-equivalent units,
  final held-out test evaluation 32 batches each. Router never calls a probe.
- Verify mechanism fixtures before formal runs: delayed warning, uphill grace,
  failed-pulse escalation, energy cap, momentum scaling, no-action equality,
  exact oracle accounting, no extra model forward/backward, and ledger closure.
- Run the existing regression suite plus new mechanism tests. Save frozen
  configs, source hashes, complete actions, per-update traces and checkpoints.
  Report all results, including no activation or harm. Smoke results alone do
  not establish broad optimizer superiority.
