# E20 implementation plan

Approved design: ../specs/2026-11-09-collision-event-design.md.

1. Add isolated replay/reconstruction helpers in new modules; use frozen
   CoolingPopulation/ScatteringPopulation without editing historical dynamics.
2. Add source/replay/sensitivity runner with fresh directories, incremental
   tables, exact source archives and actual query accounting.
3. Test reconstruction and partial-step clock/drag, physical maps, costs and
   feature boundaries. Run full suite, then a separate preflight seed1190.
4. Independently audit saved states and ledger. Run18 sensitivity trajectories
   and development1200–1209, fit fixed models and freeze their inputs.
5. Run confirmation1240–1259 only after freeze. Audit before inference, evaluate
   held-out effects and predictors, produce raw-data-backed plots.
6. Write report/log, keep failed runs, archive analysis sources and update
   registry/inventory with execution-versus-view counts explicit.

No writing-plans skill exists in the available skill locations; this concrete
plan supplies the workflow without introducing an approval already given by the
user. No new thread, delegate, external publication or model policy is required.
