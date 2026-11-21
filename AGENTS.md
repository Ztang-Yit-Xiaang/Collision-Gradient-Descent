# Project working instructions

## Commands and code discovery

- Prefix shell commands with `rtk`. If the binary is unavailable, use the existing
  session's transparent shell function rather than installing tools unnecessarily.
- Use codebase-memory-mcp first for code discovery: search_graph, trace_path,
  get_code_snippet, query_graph or search_code. Index an unindexed project first.
- Read existing files before editing. Do not overwrite historical experiment output.

## Experiment records — requested by the user on 2026-11-08

- Maintain `docs/experiment_log.md` as the human-readable chronological record of
  experiments and the evidence behind changes in research direction.
- Before any new batch, save its question/hypothesis, parent experiment, exact
  changes, baselines, seeds/data splits, tuning allowance, stopping conditions,
  query/compute/energy budgets, metrics and success criteria. No approval question
  is required just to record a protocol already within the user's authorized task.
- Use a fresh output directory. Save complete configurations, source/data hashes,
  per-run outcomes, actual query counts, actual interventions/supply/dissipation,
  failure reasons, and the raw artifacts needed to reconstruct reported figures.
- Preserve the exact source version with a commit or source archive and record
  the actual launch command/environment. A hash alone cannot restore old code.
- After the batch, append results, limitations and the reason for continuing,
  changing or abandoning a direction. Update `docs/experiment_registry.json` and
  `docs/experiment_inventory.csv`. Record development preflights and failed runs.
- Separate real training from previews, formal comparisons from mechanism checks,
  and new runs from derived budget prefixes/plots. JSON and CSV representations
  of the same run are not independent experiments. Count paired replays separately
  from the number of comparisons they support.
- Report current/final loss and historical best separately. Saving a best checkpoint
  is not evidence that an intervention is beneficial or free of harm.
- Do not tune on held-out results while describing those same results as untouched
  confirmation. Keep seed, data and tuning provenance explicit.
- Unknown historical dates, missing artifacts and overwritten transient runs must
  be labeled as unknown/unrecoverable; do not reconstruct them as invented facts.
- Changes motivated by physics must state which subsystem conserves what and
  record external work, impulses, damping and numerical residuals separately.

## Language

Converse with the user in Traditional Chinese regardless of their input language.

## Metric terminology — approved by the user on 2026-11-10

- Use 「好粒子途中曾明顯偏離初始解的次數」 in current explanations and tables.
  Count one run if the good particle's loss at any saved observation exceeds
  its own initial loss by more than 1e-4. This measures loss increase, not
  parameter distance, physical damage or final failure. Report final recovery
  separately and state the denominator (e.g. 20/20 or 0/20).
- Keep historical archives and machine-readable metric keys unchanged; a label
  update is not a new experiment or a change to the metric/threshold.
