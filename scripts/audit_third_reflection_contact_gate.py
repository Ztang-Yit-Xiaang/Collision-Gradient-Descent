"""Audit the full and gated saved-point decisions on the same paid branches."""

import argparse
from pathlib import Path

from audit_third_reflection_replication import audit_replication
from finalize_initial_budget import create
from run_later_reflection_warning import LIMIT
from run_third_reflection_contact_gate import CONTACT_FRACTION
from run_third_reflection_guard import EPS, read, sha


def gate(row,decision,fraction):
    extended = row["extended_forecast"]
    prediction = decision["prediction"]
    contact_rise = (prediction["next_contact"]["loss"]-decision["checkpoint"]["initial_loss"]
                    if extended else None)
    skip = extended and contact_rise < fraction*LIMIT
    assert decision["gated_forecast_queries"]==row["saved_window_forecast_queries"]-2*int(skip)
    assert decision["gated_warning"]==(row["saved_window_warning"] and not skip)
    return {**row,"contact_rise_for_extension":contact_rise,"extension_skipped":skip,
            "gated_queries":row["saved_window_forecast_queries"]-2*int(skip),
            "gated_transfer":row["saved_window_transfer"] and not skip,
            "gated_policy":row["baseline"] if skip else row["saved_window_policy"],
            "missed_warning":skip and row["saved_window_warning"]}


def compare(rows):
    modes = ("saved_window_policy","gated_policy")
    result = {"contexts":len(rows),"contact_fraction":CONTACT_FRACTION}
    for key in ("saved","including_contacts","after_t2_saved","after_t2_with_contacts"):
        base,full,gated = ([r[m][key] for r in rows] for m in ("baseline",*modes))
        result[key] = {"e40":sum(t is not None for t in base),"full":sum(t is not None for t in full),
                       "gated":sum(t is not None for t in gated),
                       "full_prevented":sum(b is not None and f is None for b,f in zip(base,full)),
                       "gated_prevented":sum(b is not None and g is None for b,g in zip(base,gated)),
                       "lost_preventions":sum(b is not None and f is None and g is not None for b,f,g in zip(base,full,gated)),
                       "new_crossings":sum(f is None and g is not None for f,g in zip(full,gated)),
                       "earlier":sum(f is not None and g is not None and g<f-EPS for f,g in zip(full,gated)),
                       "unnecessary":[sum(r[m] and b is None for r,b in zip(rows,base))
                                      for m in ("saved_window_transfer","gated_transfer")]}
    forecast = [sum(r[k] for r in rows) for k in ("saved_window_forecast_queries","gated_queries")]
    suffix = [sum(r["transfer_suffix_query_change"] for r in rows if r[k])
              for k in ("saved_window_transfer","gated_transfer")]
    base = sum(r["e40_policy_queries"] for r in rows)
    result.update(forecast_queries=forecast,forecast_queries_saved=forecast[0]-forecast[1],
                  policy_queries=[base+f+s for f,s in zip(forecast,suffix)],net_suffix_queries_vs_e40=suffix,
                  extensions=[sum(r["extended_forecast"] for r in rows),
                              sum(r["extended_forecast"] and not r["extension_skipped"] for r in rows)],
                  transfers=[sum(r[k] for r in rows) for k in ("saved_window_transfer","gated_transfer")],
                  within_limit_at_end=[sum(r[m]["within_limit_at_end"] for r in rows) for m in modes],
                  missed_warnings=[r["context_id"] for r in rows if r["missed_warning"]])
    main = result["saved"]
    checks = {"at_least_three_full_preventions":main["full_prevented"]>=3,
              "all_full_preventions_kept":main["lost_preventions"]==0,
              "no_earlier_or_new_saved_crossings":main["new_crossings"]==main["earlier"]==0,
              "at_least_half_forecasts_saved":forecast[1]*2<=forecast[0],
              "unnecessary_transfers_not_increased":main["unnecessary"][1]<=main["unnecessary"][0]}
    return {**result,"checks":checks,"passed_screen":all(checks.values())}


def audit(root):
    audit_replication(root,phase="E45")
    target = root/"third_reflection"
    pairs,decisions = [read(target/(n+".json")) for n in ("paired_outcomes","decisions")]
    assert all(r["context_id"]==d["context_id"] for r,d in zip(pairs,decisions,strict=True))
    rows = [gate(r,d,CONTACT_FRACTION) for r,d in zip(pairs,decisions)]
    for r in rows:
        assert r["gated_queries"]<=r["saved_window_forecast_queries"]
        assert not r["gated_transfer"] or r["saved_window_transfer"]
        if r["retained_recovery"] is not None:
            assert r["gated_queries"]==0 and r["gated_policy"]==r["saved_window_policy"]==r["baseline"]
    result = compare(rows)
    create(target/"contact_gate_outcomes.json",rows)
    create(target/"contact_gate_results.json",{**result,"retained_recovery":read(target/"results.json")["retained_recovery"],
        "dependency_data_sha256":{str(target/n):sha(target/n) for n in ("paired_outcomes.json","decisions.json")}})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
