"""Compare the original contact window and saved-observation window in E44."""

import argparse
from pathlib import Path

from audit_third_reflection_replication import audit_replication
from finalize_initial_budget import create
from run_third_reflection_guard import EPS, read


def compare(rows):
    modes = ("original_policy","saved_window_policy")
    result = {"contexts":len(rows)}
    for metric in ("saved","including_contacts","after_t2_saved","after_t2_with_contacts"):
        base = [r["baseline"][metric] for r in rows]
        original,new = ([r[m][metric] for r in rows] for m in modes)
        result[metric] = {"e40":sum(x is not None for x in base),"original":sum(x is not None for x in original),
            "saved_window":sum(x is not None for x in new),
            "original_prevented":sum(b is not None and a is None for b,a in zip(base,original)),
            "saved_window_prevented":sum(b is not None and a is None for b,a in zip(base,new)),
            "lost_preventions":sum(b is not None and a is None and c is not None for b,a,c in zip(base,original,new)),
            "new_vs_original":sum(a is None and b is not None for a,b in zip(original,new)),
            "prevented_vs_original":sum(a is not None and b is None for a,b in zip(original,new)),
            "earlier_vs_original":sum(a is not None and b is not None and b<a-EPS for a,b in zip(original,new)),
            "original_unnecessary_transfers":sum(r["original_transfer"] and b is None for r,b in zip(rows,base)),
            "saved_window_unnecessary_transfers":sum(r["saved_window_transfer"] and b is None for r,b in zip(rows,base))}
    old_transfers,new_transfers = ([r[k] for r in rows] for k in ("original_transfer","saved_window_transfer"))
    base = sum(r["e40_policy_queries"] for r in rows)
    forecast = [sum(r[k] for r in rows) for k in ("forecast_queries","saved_window_forecast_queries")]
    suffix = [sum(r["transfer_suffix_query_change"] for r in rows if r[k]) for k in ("original_transfer","saved_window_transfer")]
    result.update(original_transfers=sum(old_transfers),saved_window_transfers=sum(new_transfers),
        original_only_transfers=sum(a and not b for a,b in zip(old_transfers,new_transfers)),
        saved_window_only_transfers=sum(b and not a for a,b in zip(old_transfers,new_transfers)),
        forecast_queries=forecast,extra_forecast_queries=forecast[1]-forecast[0],
        extended_forecasts=sum(r["extended_forecast"] for r in rows),
        incomplete_saved_forecasts=sum(r["incomplete_saved_forecast"] for r in rows),
        net_suffix_changes_vs_e40=suffix,total_policy_queries=[base+f+s for f,s in zip(forecast,suffix)],
        within_limit_at_end=[sum(r[m]["within_limit_at_end"] for r in rows) for m in modes])
    main = result["saved"]
    checks = {"three_or_more_original_preventions":main["original_prevented"]>=3,
              "keeps_all_original_preventions":main["lost_preventions"]==0,
              "no_new_saved_crossings":main["new_vs_original"]==0,
              "fewer_transfers":sum(new_transfers)<sum(old_transfers)}
    return {**result,"checks":checks,"passed_screen":all(checks.values())}


def audit(root):
    audit_replication(root,phase="E44")
    target = root/"third_reflection"
    rows = read(target/"paired_outcomes.json")
    result = compare(rows)
    result["retained_recovery"] = read(target/"results.json")["retained_recovery"]
    create(target/"saved_window_results.json",result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
