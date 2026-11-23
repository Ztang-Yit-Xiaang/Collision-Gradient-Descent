"""Audit E46 timing, charged predictions, and the paired continuations."""

import argparse
from collections import Counter
from pathlib import Path

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, check_fraction, first_cross, update_peak
from audit_second_reflection_capped_policy import recovery
from finalize_initial_budget import create
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, FRACTION, H, LIMIT, START, STOP_STEP, read, sha


def score(saved,contacts,initial):
    first = first_cross(saved,initial)
    any_cross = first_cross(saved+contacts,initial)
    return {"first_saved_crossing":first,"first_crossing_including_contacts":any_cross,
            "final_rise":saved[-1][1]-initial,"within_limit_at_end":saved[-1][1]-initial<=LIMIT,
            **(recovery(saved+contacts,initial,any_cross) if any_cross is not None
               else {"first_recovery":None,"first_recross":None})}


def audit_forecasts(c,saved,contacts,peaks):
    contact_time,contact_loss = contacts[0]
    initial = c["initial_loss"]
    update_peak(peaks,"constructed_contact_time",abs(contact_time-c["contact_time_target"]))
    update_peak(peaks,"constructed_contact_loss",abs(contact_loss-initial-c["contact_ratio_target"]*LIMIT))
    previous = [p for p in saved if p[0]<contact_time]
    next_saved = next(p for p in saved if p[0]>contact_time)
    assert previous and all(f-initial<=LIMIT for _,f in previous)
    for mode in ("full","gated"):
        forecast = c[mode]
        post = forecast.get("after_contact_saved")
        expected_extension = mode=="full" or contact_loss-initial>=FRACTION*LIMIT
        assert (post is not None)==expected_extension
        assert forecast["counts"]=={"value_calls":0,"gradient_calls":int(expected_extension),"path_calls":1+int(expected_extension)}
        assert len(previous)==len(forecast["observations"])
        for actual,pred in zip(previous,forecast["observations"],strict=True):
            update_peak(peaks,"forecast_time",abs(actual[0]-pred["time"]))
            update_peak(peaks,"forecast_loss",abs(actual[1]-pred["loss"]))
        update_peak(peaks,"forecast_time",abs(contact_time-forecast["next_contact"]["time"]))
        update_peak(peaks,"forecast_loss",abs(contact_loss-forecast["next_contact"]["loss"]))
        if post:
            update_peak(peaks,"forecast_time",abs(next_saved[0]-post["time"]))
            update_peak(peaks,"forecast_loss",abs(next_saved[1]-post["loss"]))
        assert forecast["warning"]==(expected_extension and next_saved[1]-initial>LIMIT)
    return contact_time,contact_loss-initial,next_saved[0],next_saved[1]-initial


def summarize(rows, *, modes=("full","gated")):
    summary = {"contexts":len(rows),"target_positives":sum(r["target_positive"] for r in rows),
               "contact_safe_positives":sum(r["target_positive"] and r["contact_rise"]<=LIMIT for r in rows)}
    for mode in modes:
        active = [r for r in rows if r[mode]["warning"]]
        positive = [r for r in rows if r["target_positive"]]
        selected = [r[mode]["outcome"] for r in rows]
        summary[mode] = {"queries":sum(r[mode]["queries"] for r in rows),"warnings":len(active),
                         "true_positives":sum(r[mode]["warning"] for r in positive),
                         "missed_positives":sum(not r[mode]["warning"] for r in positive),
                         "false_positives":sum(not r["target_positive"] for r in active),
                         "prevented_saved":sum(r["keep"]["first_saved_crossing"] is not None and
                                               r[mode]["outcome"]["first_saved_crossing"] is None for r in rows),
                         "unnecessary_transfers":sum(r["keep"]["first_saved_crossing"] is None for r in active),
                         "saved_deviated":sum(r["first_saved_crossing"] is not None for r in selected),
                         "contact_inclusive_deviated":sum(r["first_crossing_including_contacts"] is not None for r in selected),
                         "final_safe":sum(r["within_limit_at_end"] for r in selected),
                         "recovered":sum(r["first_recovery"] is not None for r in selected),
                         "recrossed":sum(r["first_recross"] is not None for r in selected)}
    summary["missed_by_gate"] = [r["context_id"] for r in rows if r["target_positive"] and not r["gated"]["warning"]]
    return summary


def audit(root):
    cfg,contexts,run_rows,cost = [read(root/(n+".json")) for n in ("config","contexts","runs","execution_counts")]
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    assert len(contexts)==len(cfg["jobs"])==cost["contexts"] and len(run_rows)==cost["continuations"]
    assert cost["actual_new_queries"]==sum(v for k,v in cost.items() if k.endswith("_queries") and k!="actual_new_queries")
    assert cost["actual_new_queries"]<=cfg["max_new_queries"]
    runs = {(r["context_id"],r["policy"]):r for r in run_rows}
    rows,ledgers = [],[]
    counts,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    for c in contexts:
        key = c["context_id"]
        initial = c["initial_loss"]
        cp = {"time":START,"boundary_step":BOUNDARY_STEP,"remaining_duration":BOUNDARY_STEP*H-START,
              "post_reflection":c["start"],"initial_loss":initial,"already_deviated":False,"above_at_contact":False}
        branches = {}
        for mode,checker in (("keep",check_fraction),("capped",capped_fraction)):
            if (key,mode) not in runs:
                continue
            run = runs[key,mode]
            _,_,_,f,contacts,ledger = audit_branch(root,run,cp,c,counts,peaks,fraction_audit=checker,
                                                  end_step=STOP_STEP,query_cap=ADDED_BUDGET)
            saved = [((BOUNDARY_STEP+i)*H,float(v)) for i,v in enumerate(f[:,0])]
            branches[mode] = score(saved,contacts,initial)
            ledgers.append(ledger)
            if mode=="keep":
                keep_saved,keep_contacts = saved,contacts
        t,rise,ts,rs = audit_forecasts(c,keep_saved,keep_contacts,peaks)
        target_positive = branches["keep"]["first_saved_crossing"]==ts
        assert target_positive==(c["case"]<=6)
        assert (rise<=LIMIT)==(c["case"]!=6)
        row = {"context_id":key,"case":c["case"],"seed":c["seed"],"target_positive":target_positive,
               "contact_time":t,"contact_rise":rise,"first_post_contact_saved_time":ts,"first_post_contact_saved_rise":rs,
               "previous_saved_rises":[[u,f-initial] for u,f in keep_saved if u<t],"keep":branches["keep"]}
        for mode in ("full","gated"):
            warning = c[mode]["warning"]
            row[mode] = {"warning":warning,"queries":sum(c[mode]["counts"].values()),
                         "outcome":branches["capped" if warning else "keep"]}
        rows.append(row)
    assert cost["full_forecast_queries"]==sum(r["full"]["queries"] for r in rows)
    assert cost["gated_forecast_queries"]==sum(r["gated"]["queries"] for r in rows)
    assert cost["setup_queries"]==4*len(rows)
    assert cost["continuation_queries"]==sum(r["counts"]["total_oracle_calls"] for r in run_rows)
    result = {"passed":True,"contexts":len(rows),"continuations":len(runs),"solver_failures":[],"max_residuals":peaks,**counts}
    create(root/"paired_outcomes.json",rows)
    create(root/"physical_ledgers.json",ledgers)
    create(root/"results.json",summarize(rows))
    create(root/"audit.json",result)
    print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
