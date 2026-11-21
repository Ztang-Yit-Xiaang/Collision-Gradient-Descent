"""Audit and compare the paired E37 and second-reflection capped policies."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import absolute_points, audit_branch, audit_prediction, first_cross, update_peak
from finalize_initial_budget import create
from run_later_reflection_warning import BOUNDARY_STEP, H, LIMIT, decode_rows, late_checkpoint, read, sha
from run_second_reflection_branches import selected_source


def recovery(points, initial, time, already_within=False):
    t0 = time if already_within else next((t for t,f in sorted(points) if t>time and f-initial<=LIMIT),None)
    recross = None if t0 is None else first_cross([(t,f) for t,f in points if t>t0+1e-10],initial)
    return {"first_recovery":t0,"first_recross":recross}


def policy_outcome(saved, contacts, initial):
    first = first_cross(saved,initial)
    any_contact = first_cross(saved+contacts,initial)
    return {"first_crossing":first,"first_crossing_including_contacts":any_contact,
            "by_t1":first is not None and first<=1.+1e-10,"by_t2":first is not None,
            "within_limit_at_end":saved[-1][1]-initial<=LIMIT,"final_rise":saved[-1][1]-initial,
            "best_saved_loss":min(f for _,f in saved)}


def compare(root,row,context,case,runs,baseline_runs,counts,peaks):
    target = root/"second_reflection"
    choice,source,family,trace,events,files = selected_source(root,row,context)
    assert source["run_id"]==case["source"] and all(sha(p)==case["source_hashes"][str(p)] for p in files)
    initial = context["initial_loss"]
    saved,contacts = absolute_points(trace,events,BOUNDARY_STEP)
    first_contact = (family["contact_time"],initial+family["contact_rise"])
    prefix = [(i*H,initial+rise) for i,rise in enumerate(family["history_rises"])]+[first_contact]
    saved = prefix+saved
    baseline = policy_outcome(saved,contacts,initial)
    assert baseline["by_t1"]==source["primary"]["deviated"] and baseline["by_t2"]==source["extended"]["deviated"]
    assert baseline["within_limit_at_end"]==source["extended"]["within_limit"]
    omitted_guard = int(context["contact_guard_queries"] and not context["contact_guard_alert"])
    baseline_charge = baseline_runs[source["run_id"]]["counts"]["total_oracle_calls"]+omitted_guard
    baseline_predictions = choice["forecast_queries"]+context["contact_guard_queries"]
    base_transfers = sum(choice[k] for k in ("early_transfer","second_transfer","bounce_transfer"))
    cp = case["checkpoint"]
    policy = baseline
    new_saved,new_contacts = saved,contacts
    delta = case["forecast_queries"]
    ledger = None
    skip_recovery = None
    if cp:
        index = next(i for i,e in enumerate(events) if e["i"]==0)
        _,rebuilt = late_checkpoint(trace,events,index,2,context,family)
        assert cp==rebuilt
        assert case["forecast_queries"]==int(not cp["already_deviated"])
        audit_prediction({**cp,"forecast_counts":{"value_calls":0,"gradient_calls":0,"path_calls":case["forecast_queries"]},
                          "warning":case["warning"],"prediction":case.get("prediction"),
                          "first_predicted_crossing":case.get("first_predicted_crossing")},trace,contacts,peaks)
        if cp["already_deviated"]:
            assert not case["warning"] and not case["transferred"]
            skip_recovery = {"above_at_contact":cp["above_at_contact"],"time":cp["time"],
                             **recovery(saved+contacts,initial,cp["time"],not cp["above_at_contact"]),
                             "within_limit_at_end":baseline["within_limit_at_end"]}
    else:
        assert not contacts and not case["forecast_queries"] and not case["transferred"]
    if case["transferred"]:
        assert case["warning"] and not cp["already_deviated"]
        r = runs[case["context_id"]]
        assert r["charged_prefix_queries"]==sum(case["prefix_queries"].values())+case["forecast_queries"]
        assert r["checkpoint_id"]==cp["checkpoint_id"]
        _,q,v,f,tail_contacts,ledger = audit_branch(target,r,cp,context,counts,peaks,fraction_audit=capped_fraction)
        new_saved = [point for point in saved if point[0]<cp["time"]]
        new_saved += [((cp["boundary_step"]+i)*H,float(y)) for i,y in enumerate(f[:,0])]
        new_contacts = [p for p in contacts if p[0]<=cp["time"]+1e-10]+tail_contacts
        policy = policy_outcome(new_saved,new_contacts,initial)
        delta = r["counts"]["total_oracle_calls"]-baseline_charge
        # The retained replay pays four extra boundary validations.
        step = cp["event"]["step"]
        inherited = int(trace[step-1]["total_oracle_calls"])+omitted_guard
        paid_events = [e for e in events[:cp["source_event_index"]+1] if e["step"]==step]
        iterations = 1+sum(b["time"]-a["time"]>1e-11 for a,b in zip(paid_events,paid_events[1:]))
        assert r["charged_prefix_queries"]==inherited+8*iterations+2*len(paid_events)+case["forecast_queries"]
    item = {**{k:case[k] for k in ("context_id","seed","incumbent_energy","warning","transferred","forecast_queries")},
            "baseline":baseline,"policy":policy,"baseline_query_count":baseline_charge,
            "baseline_forecasts":baseline_predictions,"baseline_transfers":base_transfers,"net_query_change":delta,
            "checkpoint_time":cp["time"] if cp else None,"skip_recovery":skip_recovery}
    for name,result,points in (("baseline",baseline,saved+contacts),("policy",policy,new_saved+new_contacts)):
        first = result["first_crossing_including_contacts"]
        item[name+"_late_recovery"] = None
        if first is not None and (cp is None or first>cp["time"]+1e-10):
            item[name+"_late_recovery"] = {**recovery(points,initial,first),"first_crossing":first,
                                           "within_limit_at_end":result["within_limit_at_end"]}
    return item,ledger


def recovery_totals(rows):
    return {"states":len(rows),"recovered_or_already_within":sum(r["first_recovery"] is not None for r in rows),
            "recrossed":sum(r["first_recross"] is not None for r in rows),
            "within_limit_at_end":sum(r["within_limit_at_end"] for r in rows)}


def summary(rows):
    old = [r["baseline"] for r in rows]
    new = [r["policy"] for r in rows]
    metrics = {}
    for metric in ("first_crossing","first_crossing_including_contacts"):
        both = list(zip((r[metric] for r in old),(r[metric] for r in new)))
        metrics[metric] = {"e37_deviated":sum(a is not None for a,b in both),
                          "capped_deviated":sum(b is not None for a,b in both),
                          "prevented":sum(a is not None and b is None for a,b in both),
                          "induced":sum(a is None and b is not None for a,b in both),
                          "earlier":sum(a is not None and b is not None and b<a-1e-10 for a,b in both)}
    primary = metrics["first_crossing"]
    late = [r["skip_recovery"] for r in rows if r["skip_recovery"] is not None]
    warnings = sum(r["warning"] for r in rows)
    transfers = sum(r["transferred"] for r in rows)
    unnecessary = sum(r["transferred"] and not r["baseline"]["by_t2"] for r in rows)
    checks = {"at_least_five_e37_deviations":primary["e37_deviated"]>=5,
              "prevents_at_least_75_percent":bool(primary["e37_deviated"] and primary["prevented"]>=.75*primary["e37_deviated"]),
              "no_new_or_earlier":all(m["induced"]==0 and m["earlier"]==0 for m in metrics.values()),
              "unnecessary_at_most_10_percent":bool(transfers and unnecessary<=.10*transfers)}
    return {"contexts":len(rows),"seed_groups":len({r["seed"] for r in rows}),**metrics,
            "by_t1":[sum(r["by_t1"] for r in g) for g in (old,new)],
            "within_limit_at_end":[sum(r["within_limit_at_end"] for r in g) for g in (old,new)],
            "median_final_rise":[median(r["final_rise"] for r in g) for g in (old,new)],
            "median_historical_best_loss":[median(r["best_saved_loss"] for r in g) for g in (old,new)],
            "second_reflections":sum(r["checkpoint_time"] is not None for r in rows),
            "new_forecast_queries":sum(r["forecast_queries"] for r in rows),
            "e37_forecast_queries":sum(r["baseline_forecasts"] for r in rows),
            "e37_total_policy_queries":sum(r["baseline_query_count"] for r in rows),
            "net_total_policy_queries":sum(r["net_query_change"] for r in rows),
            "e37_transfers":sum(r["baseline_transfers"] for r in rows),"new_transfers":transfers,"warnings":warnings,
            "unnecessary_transfers":unnecessary,"missed_eligible":sum(r["forecast_queries"] and not r["warning"] and r["baseline"]["by_t2"] for r in rows),
            "warning_false_positives":sum(r["warning"] and not r["baseline"]["by_t2"] for r in rows),
            "already_deviated_at_second_reflection":{
                "all":recovery_totals(late),"still_above":recovery_totals([r for r in late if r["above_at_contact"]]),
                "already_within":recovery_totals([r for r in late if not r["above_at_contact"]])},
            "deviated_after_second_reflection":{mode:recovery_totals([r[mode+"_late_recovery"] for r in rows if r[mode+"_late_recovery"] is not None]) for mode in ("baseline","policy")},
            "checks":checks,"passed_screen":all(checks.values())}


def audit(root):
    target = root/"second_reflection"
    cfg = read(target/"config.json")
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    contexts = {r["context_id"]:r for r in read(root/"contexts.json")}
    baseline = {r["run_id"]:r for r in read(root/"runs.json")}
    runs = {r["context_id"]:r for r in read(target/"runs.json")}
    cases,rows = read(target/"contexts.json"),read(root/"paired_outcomes.json")
    assert len(cases)==len(rows)==len(contexts)
    results,ledgers = [],[]
    counts,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    for row,case in zip(rows,cases,strict=True):
        assert row["context_id"]==case["context_id"]
        outcome,ledger = compare(root,row,contexts[row["context_id"]],case,runs,baseline,counts,peaks)
        results.append(outcome)
        if ledger:
            ledgers.append(ledger)
    costs = read(target/"execution_counts.json")
    assert len(runs)==costs["continuations"]==sum(c["transferred"] for c in cases)
    assert costs["actual_forecast_queries"]==sum(c["forecast_queries"] for c in cases)
    for kind in ("fraction","tail"):
        assert costs[f"actual_{kind}_queries"]==sum(r[f"actual_{kind}_calls"] for r in runs.values())
    assert costs["actual_new_solver_queries"]==sum(costs[k] for k in (
        "actual_forecast_queries","actual_fraction_queries","actual_tail_queries"))
    create(target/"paired_outcomes.json",results)
    create(target/"already_deviated_cases.json",[r for r in results if r["skip_recovery"] is not None])
    create(target/"physical_ledgers.json",ledgers)
    create(target/"results.json",summary(results))
    audit_result = {"passed":True,"contexts":len(cases),"continuations":len(runs),"solver_failures":[],
                    "max_residuals":peaks,"value_checks":counts["value_checks"],"gradient_checks":counts["gradient_checks"]}
    create(target/"audit.json",audit_result)
    print(audit_result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
