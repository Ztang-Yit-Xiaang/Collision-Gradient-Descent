"""Audit the extended model forecast and compare both alerts at the same event."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from audit_contact_warning_replication import LONG, SHORT, analyze, audit, classify, summarize
from run_contact_long_forecast import SEEDS
from run_contact_warning_replication import BOUNDARY_STEP, END_STEP, H, LIMIT, MODES


def audit_extension(c, continuation, peaks):
    queries = c["actual_forecast_calls"]
    if c["already_deviated"]:
        assert queries == 0 and "extended_forecast" not in c
        return {"extended_warning": False, "extended_forecast_complete": None, "forecast_queries": 0}
    forecast = c["extended_forecast"]
    points = forecast["observations"]
    assert queries == (1 if forecast["next_contact"] is None else 3)
    f = continuation[2][:, 0] - c["initial_loss"]
    assert [p["step"] for p in points] == list(range(BOUNDARY_STEP, BOUNDARY_STEP + len(points)))
    assert (not points and c["predicted_boundary_rise"] is None) or (
        points and points[0]["loss_rise"] == c["predicted_boundary_rise"])
    for p in points:
        idx = p["step"] - BOUNDARY_STEP
        if idx >= len(f):
            break
        error = abs(p["loss_rise"] - f[idx])
        assert error < 1e-12
        peaks["extended_forecast_loss"] = max(peaks.get("extended_forecast_loss", 0.), error)
    if forecast["next_contact"] is not None:
        observed = next((e["time"] + BOUNDARY_STEP*H for e in continuation[4] if e["i"] == 0), None)
        if observed is not None:
            error = abs(observed - forecast["next_contact"])
            assert error < 1e-12
            peaks["extended_reflection_time"] = max(peaks.get("extended_reflection_time", 0.), error)
    hits = [p for p in points if p["loss_rise"] > LIMIT]
    assert not hits or (len(hits) == 1 and hits[0] is points[-1])
    if forecast["complete"] and not hits:
        assert points[-1]["step"] == END_STEP
    actual = np.flatnonzero(f > LIMIT)
    return {
        "extended_warning": bool(hits), "extended_forecast_complete": forecast["complete"],
        "forecast_queries": queries,
        "predicted_crossing_step": hits[0]["step"] if hits else None,
        "predicted_crossing_after_next_reflection": hits[0]["after_next_contact"] if hits else None,
        "keep_first_future_deviation": float((actual[0]+BOUNDARY_STEP)*H) if actual.size else None,
    }


def report(rows):
    baseline = analyze(rows)
    valid = [r for r in rows if r["eligible"] and all(b["complete"] for b in r["branches"].values())]
    active = [r for r in valid if not r["already_deviated"]]
    metrics = {}
    for name, key in (("single_step", "warning"), ("extended", "extended_warning")):
        values = [{**r, "warning": r[key]} for r in active]
        lead = [r["keep_first_future_deviation"]-r["contact_time"] for r in active
                if r[key] and r["keep_first_future_deviation"] is not None]
        costs = [r["forecast_queries"] if name == "extended" else 1 for r in active]
        metrics[name] = {
            "point35": classify(values, SHORT), "through_end": classify(values, LONG),
            "forecast_queries": sum(costs), "mean_forecast_queries": sum(costs)/len(costs) if costs else None,
            "forecast_query_distribution": dict(sorted(Counter(costs).items())),
            "unavailable_or_incomplete": sum(not r["extended_forecast_complete"] for r in active)
                if name == "extended" else sum(r["predicted_boundary_rise"] is None for r in active),
            "lead_min_median_max": [min(lead), median(lead), max(lead)] if lead else None,
            "pair_groups": {group: {mode: summarize([
                r["branches"][mode] for r in active if r[key] == warning
            ]) for mode in MODES} for group, warning in (("warning", True), ("no_warning", False))},
            "cohort_warning_only": summarize([
                r["branches"]["transfer" if r[key] else "keep"] for r in valid]),
            "cohort_warning_and_recovery": summarize([
                r["branches"]["transfer" if r[key] or r["already_deviated"] else "keep"] for r in valid]),
        }
    short = metrics["single_step"]["through_end"]
    long = metrics["extended"]["through_end"]
    recovered = [r for r in active if not r["warning"] and r["extended_warning"] and r["branches"]["keep"][LONG]]
    checks = {
        "all_pairs_complete_and_forecasts_available": len(valid) == baseline["eligible"]
            and not metrics["extended"]["unavailable_or_incomplete"],
        "at_least_twenty_positive_controls": long["positive"] >= 20,
        "recall_at_least_90_percent": bool(long["positive"] and long["true_warnings"]/long["positive"] >= .90),
        "false_warning_rate_at_most_10_percent": bool(long["negative"] and long["false_warnings"]/long["negative"] <= .10),
        "recovered_75_percent_of_single_step_misses": bool(short["misses"] and len(recovered)/short["misses"] >= .75),
        "mean_at_most_three_forecast_queries": metrics["extended"]["mean_forecast_queries"] <= 3,
        "prevent_at_least_90_percent_of_true_warnings":
            bool(long["true_warnings"] and long["prevented_true_warnings"]/long["true_warnings"] >= .90),
    }
    return {
        "eligible": baseline["eligible"], "complete_pairs": len(valid), "ineligible": baseline["ineligible"],
        "unique_seed_groups": baseline["unique_seed_groups"], "predictions": metrics,
        "single_step_misses_recovered": len(recovered),
        "recovered_crossings_after_second_reflection": sum(r["predicted_crossing_after_next_reflection"] for r in recovered),
        "already_deviated": baseline["groups"]["already_deviated"],
        "always_keep": baseline["cohort_always_keep"], "checks": checks, "passed_screen": all(checks.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output, seed_sets=SEEDS, evaluator=report, extension_audit=audit_extension)
