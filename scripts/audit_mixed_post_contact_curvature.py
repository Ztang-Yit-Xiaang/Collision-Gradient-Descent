"""Rebuild E49 outcomes and independently verify both forecast windows."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, check_fraction, update_peak
from audit_post_contact_positive_stress import score
from finalize_initial_budget import create
from run_mixed_post_contact_curvature import POLICIES, SEEDS, STRATA
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, H, LIMIT, START, STOP_STEP, read, sha
from run_later_reflection_warning import decode_rows

METRIC = "first_saved_crossing"


def audit_forecasts(row, trace, events, peaks):
    event, *later = [e for e in events if e["i"] == 0]
    contact = BOUNDARY_STEP*H + event["time"]
    next_contact = min((BOUNDARY_STEP*H + e["time"] for e in later), default=float("inf"))
    contact_loss = event["point_i"][-1]
    initial = row["initial_loss"]
    update_peak(peaks, "constructed_contact_time", abs(contact - row["contact_time_target"]))
    update_peak(peaks, "constructed_contact_loss", abs(contact_loss - initial - row["contact_ratio_target"]*LIMIT))
    boundary = int(np.ceil(contact/H))
    before = trace[:boundary-BOUNDARY_STEP]
    points = trace[boundary-BOUNDARY_STEP:boundary+2-BOUNDARY_STEP]
    rotation = np.asarray(row["rotation"])
    x, y = rotation @ event["point_i"][:2]
    velocity = np.asarray(event["after_i"])
    remaining = boundary*H - contact
    beta = float(points[1]["drag_beta"])
    bounds, available = [], []
    for offset, (duration, point) in enumerate(zip((remaining, remaining + beta*H), points, strict=True)):
        dx, dy = rotation @ (duration * velocity[:2])
        curvature = max(0., 3*max(x*x, (x+dx)**2)-1.) * dx**2 + 8.*dy**2
        slope = (x**3-x+.05)*dx + 8.*y*dy
        upper = contact_loss + max(0., slope + .5*curvature)
        time = (boundary+offset)*H
        free_flight = next_contact > time
        if free_flight:
            assert point["values"][0] <= upper + 1e-12
        available.append({"time": time, "loss": point["values"][0], "free_flight": free_flight})
        bounds.append(upper)

    prior_warning = any(r["values"][0] - initial > LIMIT for r in before)
    for policy, length in zip(POLICIES, (1, 2), strict=True):
        forecast = row[policy]
        for i, (actual, predicted) in enumerate(zip(before, forecast["observations"], strict=True)):
            assert predicted["step"] == BOUNDARY_STEP+i
            update_peak(peaks, "forecast_loss", abs(predicted["loss"] - actual["values"][0]))
        hit = forecast["next_contact"]
        update_peak(peaks, "forecast_time", abs(hit["time"] - contact))
        update_peak(peaks, "forecast_loss", abs(hit["loss"] - contact_loss))
        paid = not prior_warning and max(bounds[:length]) >= initial+LIMIT-1e-12
        assert forecast["counts"] == {"value_calls": 0, "gradient_calls": int(not prior_warning), "path_calls": 1+int(paid)}
        if not prior_warning:
            bound = forecast["post_contact_bound"]
            update_peak(peaks, "bound_loss", abs(bound["upper_loss"] - bounds[0]))
            update_peak(peaks, "outgoing_velocity", np.max(abs(velocity - bound["outgoing_velocity"])))
            if length == 2:
                update_peak(peaks, "bound_loss", abs(forecast["second_saved_upper_loss"] - bounds[1]))
        warning = prior_warning
        for key, point in zip(("after_contact_saved", "second_after_contact_saved")[:length], available[:length], strict=True):
            pred = forecast.get(key)
            supported = paid and point["free_flight"] and not warning
            assert (pred is not None) == supported
            if supported:
                update_peak(peaks, "forecast_loss", abs(pred["loss"] - point["loss"]))
                update_peak(peaks, "forecast_time", abs(pred["time"] - point["time"]))
                warning |= point["loss"] - initial > LIMIT
        assert forecast["warning"] == warning
    return {"time": contact, "rise": contact_loss-initial, "following_saved_points": available,
            "next_contact": next_contact if later else None, "upper_losses": bounds}


def summarize(rows, *, policies=POLICIES):
    safe = [x for x in rows if x["keep"][METRIC] is None]
    positive = [x for x in rows if x["keep"][METRIC] is not None]
    summary = {"contexts": len(rows), "safe_keep": len(safe), "deviated_keep": len(positive),
               "positive_timing": dict(Counter(r["crossing_timing"] for r in positive))}
    for mode in policies:
        outcomes = [r[mode]["outcome"] for r in rows]
        summary[mode] = {
            "transfers": sum(r[mode]["warning"] for r in rows),
            "unnecessary_transfers": sum(r[mode]["warning"] for r in safe),
            "prevented": sum(r[mode]["outcome"][METRIC] is None for r in positive),
            "deviated": sum(r[METRIC] is not None for r in outcomes),
            "deviated_including_contacts": sum(r["first_crossing_including_contacts"] is not None for r in outcomes),
            "final_safe": sum(r["within_limit_at_end"] for r in outcomes),
            "recovered": sum(r["first_recovery"] is not None for r in outcomes),
            "recrossed": sum(r["first_recross"] is not None for r in outcomes),
            "median_final_loss": median(r["final_loss"] for r in outcomes) if outcomes else None,
            "median_best_saved_loss": median(r["best_saved_loss"] for r in outcomes) if outcomes else None,
            "forecast_queries": sum(r[mode]["forecast_queries"] for r in rows),
            "total_queries": sum(r[mode]["total_queries"] for r in rows),
        }
    base, extended = policies
    first, second = ([r[p]["outcome"][METRIC] for r in rows] for p in policies)
    summary.update(
        additional_preventions=sum(a is not None and b is None and r["keep"][METRIC] is not None
                                   for r, a, b in zip(rows, first, second)),
        lost_preventions=sum(a is None and b is not None and r["keep"][METRIC] is not None
                             for r, a, b in zip(rows, first, second)),
        added_deviations_in_safe_cases=sum(r[base]["outcome"][METRIC] is None and r[extended]["outcome"][METRIC] is not None for r in safe),
        earlier_saved_crossings=sum(a is not None and b is not None and b<a for a,b in zip(first,second)),
        new_unnecessary_transfers=sum(r[extended]["warning"] and not r[base]["warning"] for r in safe),
    )
    return summary


def audit(root, *, seeds=SEEDS, policies=POLICIES, inspect=audit_forecasts, evaluate=None):
    cfg, contexts, runs, costs = [read(root / (n + ".json")) for n in ("config", "contexts", "runs", "execution_counts")]
    assert sha(root / "source.zip") == cfg["source_archive_sha256"]
    assert all(sha(Path(p)) == h for p, h in cfg["dependency_data_sha256"].items())
    expected = [(seed, i) for seed in seeds[cfg["stage"]] for i in range(len(STRATA))]
    assert cfg["jobs"]==[list(job) for job in expected]
    assert [(c["seed"], c["case"]-1) for c in contexts] == expected and len(contexts) == costs["contexts"]
    by_branch = {(r["context_id"], r["policy"]): r for r in runs}
    assert len(by_branch) == len(runs) == costs["continuations"]
    rows, ledgers = [], []
    counts, peaks = Counter(), dict(energy=0., momentum=0., reflection=0.)
    for c in contexts:
        key, initial = c["context_id"], c["initial_loss"]
        checkpoint = {"time": START, "boundary_step": BOUNDARY_STEP, "remaining_duration": BOUNDARY_STEP*H-START,
                      "post_reflection": c["start"], "initial_loss": initial, "already_deviated": False,
                      "above_at_contact": False}
        outcomes = {}
        assert ((key, "capped") in by_branch) == any(c[p]["warning"] for p in policies)
        for mode, check in (("keep", check_fraction), ("capped", capped_fraction)):
            if (key, mode) not in by_branch:
                continue
            run = by_branch[key, mode]
            _, _, _, loss, contacts, ledger = audit_branch(root, run, checkpoint, c, counts, peaks,
                                                           fraction_audit=check, end_step=STOP_STEP, query_cap=ADDED_BUDGET)
            saved = [((BOUNDARY_STEP+i)*H, float(f)) for i, f in enumerate(loss[:,0])]
            outcomes[mode] = {**score(saved, contacts, initial), "final_loss": saved[-1][1],
                              "best_saved_loss": min(f for _, f in saved)}
            assert ledger["fraction"]["external_energy_supply"] == 0.
            ledgers.append(ledger)
        keep = outcomes["keep"]
        trace = decode_rows(root, key + "__keep")
        events = read(root / "events" / (key + "__keep.json"))
        geometry = inspect(c, trace, events, peaks)
        post = geometry["following_saved_points"]
        timing_labels = {p["time"]: name for p,name in zip(
            post, ("first_post_contact_saved","second_post_contact_saved","third_post_contact_saved"))}
        timing = timing_labels.get(keep[METRIC], "later")
        if keep[METRIC] is None:
            timing = "safe"
        elif keep[METRIC]<post[0]["time"]:
            timing = "before_contact"
        setup = sum(c["setup_counts"].values())
        assert setup == 4
        row = {"context_id": key, "seed": c["seed"], "stratum": c["stratum"], "keep": keep,
               "crossing_timing": timing, "geometry": geometry}
        for policy in policies:
            warning = c[policy]["warning"]
            branch = "capped" if warning else "keep"
            forecast = sum(c[policy]["counts"].values())
            assert forecast<=cfg["max_policy_forecast_queries_each"]
            selected = by_branch[key, branch]
            row[policy] = {"warning": warning, "run_id": selected["run_id"], "outcome": outcomes[branch],
                           "forecast_queries": forecast,
                           "total_queries": setup + forecast + selected["counts"]["total_oracle_calls"]}
        rows.append(row)

    assert costs["setup_queries"] == 4*len(contexts)
    assert costs["continuation_queries"] == sum(r["counts"]["total_oracle_calls"] for r in runs)
    for policy in policies:
        assert costs[policy+"_forecast_queries"] == sum(r[policy]["forecast_queries"] for r in rows)
    assert costs["actual_new_queries"] == sum(v for k,v in costs.items()
        if k.endswith("_queries") and k!="actual_new_queries") <= cfg["max_new_queries"]
    assert len(ledgers) == len(runs)
    result = summarize(rows, policies=policies)
    result["by_keep_outcome"] = {label: summarize([r for r in rows if (r["keep"][METRIC] is None)==safe],policies=policies)
                                 for label,safe in (("safe",True),("deviated",False))}
    result["by_stratum"] = {name: summarize([r for r in rows if r["stratum"]==name],policies=policies) for name,_,_ in STRATA}
    result["checks"] = {"mixed": bool(result["safe_keep"] and result["deviated_keep"]),
                        "additional_prevention": result["additional_preventions"]>0,
                        "no_lost_preventions": result["lost_preventions"]==0,
                        "no_new_unnecessary_transfers": result["new_unnecessary_transfers"]==0}
    if evaluate is not None:
        result.update(evaluate(rows,result))
    audit_result = {"passed": True, "contexts": len(rows), "continuations": len(runs), "solver_failures": [],
                    "max_residuals": peaks, **counts}
    create(root / "paired_outcomes.json", rows)
    create(root / "physical_ledgers.json", ledgers)
    create(root / "results.json", result)
    create(root / "audit.json", audit_result)
    print({k: v for k, v in result.items() if k not in ("by_stratum", "by_keep_outcome")})
    print(audit_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
