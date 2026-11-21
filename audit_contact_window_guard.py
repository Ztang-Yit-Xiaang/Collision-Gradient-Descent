"""Audit E37 contact-specific prevention, overhead, and recovery."""

import argparse
from pathlib import Path

from audit_second_step_protection import metrics, recovery_cases, recovery_summary, select
from audit_two_stage_protection import WINDOWS, audit, lookup
from finalize_initial_budget import create, sha
from run_conditional_transfer import read
from run_contact_window_guard import SEEDS


def guarded(row):
    return {**row,"second_warning":row["second_warning"] or row["contact_guard_alert"]}


def contact_cases(rows):
    output=[]
    for r in rows:
        old=select(r,True)
        source=r["families"][old["family"]]
        before=lookup(r,old)
        if before["first_deviation"]!=source["contact_time"]:
            continue
        updated=guarded(r)
        new=select(updated,True)
        target=r["families"][new["family"]]
        after=lookup(updated,new)
        output.append({
            "context_id":r["context_id"],"seed":r["seed"],"incumbent_energy":r["incumbent_energy"],
            "second_fixed_score":r["second_score"],"guard_queries":r["contact_guard_queries"],
            "guard_alert":r["contact_guard_alert"],"guard_prediction":r["contact_guard_prediction"],
            "old_contact_rise":source["contact_rise"],"new_contact_rise":target["contact_rise"],
            "old_first_deviation":before["first_deviation"],"new_first_deviation":after["first_deviation"],
            "prevented_through_first_contact":not target["already_deviated"],
            "prevented_through_t1":not after["primary"]["deviated"],
            "prevented_through_t2":not after["extended"]["deviated"],
        })
    return output


def report(rows):
    valid=[r for r in rows if all(f["complete"] and all(b["complete"] for b in f["branches"].values())
                                 for f in r["families"].values())]
    upgraded=[guarded(r) for r in valid]
    checked=sum(r["contact_guard_queries"] for r in valid)
    alerts=sum(r["contact_guard_alert"] for r in valid)
    summary={}
    for window in WINDOWS:
        original=metrics(valid,True,window)
        new=metrics(upgraded,True,window)
        before=[lookup(r,select(r,True))[window]["deviated"] for r in valid]
        after=[lookup(r,select(r,True))[window]["deviated"] for r in upgraded]
        always=[r["families"]["base"]["branches"]["keep"][window]["deviated"] for r in valid]
        summary[window]={
            "fixed_window":original,"contact_guard":new,
            "prevented_vs_fixed":sum(b and not a for b,a in zip(before,after)),
            "induced_vs_fixed":sum(not b and a for b,a in zip(before,after)),
            "net_extra_transfers":new["total_transfers"]-original["total_transfers"],
            "guard_alerts_where_always_keep_safe":sum(r["contact_guard_alert"] and not b for r,b in zip(valid,always)),
            "guard_alerts_where_fixed_policy_safe":sum(r["contact_guard_alert"] and not b for r,b in zip(valid,before)),
            "extra_reflection_forecast_queries":new["forecast_queries"]-original["forecast_queries"],
            "extra_total_forecast_queries":checked+new["forecast_queries"]-original["forecast_queries"],
        }
    contact=contact_cases(valid)
    saved=sum(r["prevented_through_t1"] for r in contact)
    main=summary["primary"]
    checks={
        "all_policy_contexts_complete":len(valid)==len(rows),
        "at_least_three_contact_crossings":len(contact)>=3,
        "prevents_at_least_75_percent_to_t1":bool(contact and saved/len(contact)>=.75),
        "no_new_deviations":main["induced_vs_fixed"]==0,
        "unnecessary_at_most_10_percent":bool(alerts and main["guard_alerts_where_always_keep_safe"]/alerts<=.10),
    }
    return {
        "contexts":len(rows),"complete_contexts":len(valid),"seed_groups":len({r["seed"] for r in valid}),
        "policy_metrics":summary,"contact_guard_queries":checked,"contact_guard_alerts":alerts,
        "unavailable_contact_forecasts":sum(r["contact_guard_queries"] and r["contact_guard_prediction"] is None for r in valid),
        "original_first_crossings_at_contact":len(contact),
        "prevented_through_contact":sum(r["prevented_through_first_contact"] for r in contact),
        "prevented_through_t1":saved,"prevented_through_t2":sum(r["prevented_through_t2"] for r in contact),
        "recovery":{"fixed_window":recovery_summary(recovery_cases(valid,True)),
                    "contact_guard":recovery_summary(recovery_cases(upgraded,True))},
        "checks":checks,"passed_screen":all(checks.values()),
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    root=parser.parse_args().output
    audit(root,seed_sets=SEEDS,evaluator=report,second_stage=True)
    if read(root/"config.json")["stage"]=="formal":
        source=root/"paired_outcomes.json"
        rows=[r for r in read(source) if all(f["complete"] and all(b["complete"] for b in f["branches"].values())
                                            for f in r["families"].values())]
        create(root/"first_crossings_at_contact.json",{"rows":contact_cases(rows),"source_sha256":sha(source)})
        create(root/"policy_recovery_cases.json",{
            "fixed_window":recovery_cases(rows,True),"contact_guard":recovery_cases([guarded(r) for r in rows],True),
            "source_sha256":sha(source),"new_oracle_calls":0,
        })
