# Damping replication and a state-independent timing control

The user approved continuing the damping branch after the routing smoke test.
This follow-up tests reproducibility and whether state-based timing adds value.
It is an experiment, not a new claim of optimizer superiority.

## Frozen protocol

- NanoGPT model, data, clipping, batch construction and old selected learning
  rates remain as in the previous comparison. Momentum=0.9, lr=0.3; AdamW lr=0.001;
  both weight decay=0. No new parameter search.
- Independent initialization/training seeds 400--409. Same fixed 32 final test
  batches as before. These are new seeds, not a newly unseen dataset: previous
  results have informed the research direction.
- Five arms: plain Momentum, existing state-based restart, existing state-based
  damping, fixed-time damping, plain AdamW. No impulses or energy supply.
- Existing state-based thresholds, cooldown and caps remain unchanged. The
  fixed-time arm multiplies inherited momentum by 0.5 after 80, 160, 240 and 320
  completed updates, before the next update. It changes the native buffer and
  external component, with exactly four events and no per-step state features.
- This is a simple state-independent control, not an event-count-matched causal
  experiment: actual state-based event counts can differ and must be reported.
- Every arm completes 400 real updates (1,200 forward-equivalent training units),
  with zero diagnostic or speculative model forwards. Final evaluation is 32
  forwards outside the training cap, identical across arms.
- Method order is independently shuffled using a fixed seeded permutation per
  training seed. CPU threads=1. Record wall time and feature time; no claim of
  precise throughput without a dedicated timing study.

## Endpoints and limits

Primary descriptive comparisons: state-based damping minus plain Momentum, and
state-based damping minus fixed-time damping, using paired final test loss.
Report each seed, mean, median, better/worse counts, and a paired percentile
bootstrap interval (20,000 draws, fixed seed 5239400). Resample seed pairs, not
individual arms. This is conditional on one dataset and frozen hyperparameters;
the interval is not a universal superiority claim or a multiple-testing result.
Report restart and AdamW baselines, actual event counts and diagnostics overhead.
Do not add seeds or change thresholds after inspecting outcomes.

## Implementation and verification plan

1. Add a separate experiment module, preserving the old implementation and raw
   results. Reuse the existing routed training for plain/restart/damp/AdamW.
2. Add the fixed-time trainer with native Momentum scaling and exact actuator
   energy accounting. Before changing momentum, check that the update is payable.
3. Verify that the periodic arm equals manually scheduled native SGD, adds no
   forwards, is exactly the baseline before its first scheduled event, and does
   not actuate on an exhausted compute cap. Verify paired statistical summaries.
4. Run the full regression suite, freeze configs/source hashes, then run all
   50 formal trainings without other experimental workloads in parallel.
5. Audit queries, paired initial states, checkpoints, action schedules and energy
   accounting. Save all runs, including failures. Build plots and a Chinese
   report from saved results. Do not turn a negative result into favorable tuning.
