# E21 implementation plan

1. Add empirical return-map and controlled-initialization helpers, keeping
   original dynamics/objectives untouched.
2. Add source/parent archives and a fresh-directory runner for preflight,
   Rosenbrock isolation and parabola map/rollout batches.
3. Test key physical and accounting contracts, run full tests, preflight and
   independent raw-state audit.
4. Run58 Rosenbrock trajectories and prescribed270 one-return/36 multi-return
   experiments; do not tune after inspecting results.
5. Analyze matched clocks, height/direction effects, probe-size consistency and
   actual finite-perturbation growth, with raw-backed plots.
6. Record outcomes/failures/limits and exact version changes; archive analysis
   and update registry/inventory. Draw empirical conclusions before any proof.

The user already approved continuing this empirical stability direction.
The referenced writing-plans skill is unavailable in the current skill catalogue;
this concrete plan is recorded without requesting repeated approval.
