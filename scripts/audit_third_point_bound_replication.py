"""Verify fresh-seed E51/E52 forecasts, bounds, and matched physical continuations."""

import argparse
from collections import Counter
from pathlib import Path

from audit_later_reflection_warning import update_peak
from audit_mixed_post_contact_curvature import METRIC, audit, summarize
from audit_third_point_bound import POLICIES, check_forecast
from run_post_contact_positive_stress import BOUNDARY_STEP, H, LIMIT
from run_third_point_bound_replication import SEEDS


def inspect(row, trace, events, peaks):
    baseline, screened = (row[p] for p in POLICIES)
    points = check_forecast(row,screened,baseline,trace,events,peaks)
    first = next(e for e in events if e["i"]==0)
    contact, loss = BOUNDARY_STEP*H + first["time"], first["point_i"][-1]
    update_peak(peaks,"constructed_contact_time",abs(contact-row["contact_time_target"]))
    update_peak(peaks,"constructed_contact_loss",abs(loss-row["initial_loss"]-row["contact_ratio_target"]*LIMIT))
    bounds = screened.get("saved_loss_upper_bounds")
    threshold = row["initial_loss"] + LIMIT
    statuses = (None if bounds is None else
                ["unavailable" if b is None else "possible_crossing" if b>=threshold-1e-12 else "below" for b in bounds])
    return {"time":contact,"rise":loss-row["initial_loss"],"following_saved_points":points,
            "saved_loss_upper_bounds":bounds,"bound_status":statuses,
            "stops":{p:row[p].get("window_stop","prior_observation") for p in POLICIES}}


def evaluate(rows, result):
    third = [r for r in rows if r["crossing_timing"]=="third_post_contact_saved"]
    skipped = [r for r in rows if r["geometry"]["stops"]["e52"]=="loss_bounds"]
    protected = [r for r in rows if r["keep"][METRIC] is not None and r["e51"]["outcome"][METRIC] is None]
    kept = [r for r in protected if r["e52"]["outcome"][METRIC] is None]
    return {
        "e51_preventions_retained":len(kept),
        "third_saved_point_cases":summarize(third,policies=POLICIES),
        "loss_bound_skips":summarize(skipped,policies=POLICIES),
        "third_point_bound_patterns":dict(Counter(
            "/".join(r["geometry"]["bound_status"] or ["no_bound"]) for r in third)),
        "forecast_stops":{p:dict(Counter(r["geometry"]["stops"][p] for r in rows)) for p in POLICIES},
        "matching_decisions":sum(r["e51"]["warning"]==r["e52"]["warning"] for r in rows),
        "available_bounds_checked":sum(b is not None for r in rows for b in (r["geometry"]["saved_loss_upper_bounds"] or [])),
        "checks":{"mixed":bool(result["safe_keep"] and third),
                  "e51_preventions_retained":len(kept)==len(protected),
                  "no_new_unnecessary_transfers":result["new_unnecessary_transfers"]==0,
                  "forecast_queries_reduced":result["e52"]["forecast_queries"]<result["e51"]["forecast_queries"],
                  "total_queries_reduced":result["e52"]["total_queries"]<result["e51"]["total_queries"]},
    }


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.output,seeds=SEEDS,policies=POLICIES,inspect=inspect,evaluate=evaluate)
