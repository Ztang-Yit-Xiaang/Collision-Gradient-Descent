# E20 numerical diagnostic extension, before long-window outcomes

The preregistered18 trajectories at physical time10 completed without failures.
The protected Rosenbrock particle's maximum loss rise is only around1e-20 in all
six settings; default and tightened contact tolerances give identical paths.
This short window does not reproduce E19's later1e-4-scale drift, so it cannot
explain that observation or establish stability through the original horizon.

Add a SEPARATE18-run diagnostic with the same seeds880–882, h grid, tolerance
profiles, initial states and absolute ramp, extending physical time to25. No
change to main event sources, features, labels or train/confirmation splits.
Maximum steps2000, oracle cap100000/run, no target stopping or parameter tuning.
Write output/collision_event_sensitivity_long; preserve the original time10
batch unchanged. This is an outcome-motivated follow-up, not preregistered from
the original E20 start and not new independent seeds or efficacy confirmation.
Compare common.05 saved times and max/current loss, energy/P ledger as before.
Record all costs, sources and failures. No further window extensions planned.
