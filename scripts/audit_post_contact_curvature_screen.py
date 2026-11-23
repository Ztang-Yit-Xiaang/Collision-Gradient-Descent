"""Verify E47 decisions and compare the selected, physically audited E46 paths."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit_later_reflection_warning import update_peak
from audit_post_contact_positive_stress import summarize
from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import decode_rows
from run_post_contact_curvature_screen import BOUNDARY_STEP, H, LIMIT, STOP_STEP, read, sha

MODES = ("full","gated","curvature")


def audit_forecast(c,decision,row,events,peaks):
    forecast = decision["forecast"]
    original = c["full"]
    initial = c["initial_loss"]
    assert forecast["observations"]==original["observations"]
    assert forecast["next_contact"]==original["next_contact"]
    bound = forecast["post_contact_bound"]
    post = original["after_contact_saved"]
    first = next(e for e in events if e["i"]==0)
    x = np.asarray(first["point_i"][:2])
    v = np.asarray(first["after_i"])
    update_peak(peaks,"outgoing_velocity",float(np.max(abs(v-bound["outgoing_velocity"]))))
    dt = post["time"]-row["contact_time"]
    update_peak(peaks,"remaining_time",abs(bound["remaining_time"]-dt))
    rot = np.asarray(c["rotation"])
    x0,y0 = rot@x
    dx,dy = rot@(dt*v[:2])
    k = max(0.,3.*max(x0**2,(x0+dx)**2)-1.)*dx**2+8.*dy**2
    linear = (x0**3-x0+.05)*dx+8.*y0*dy
    upper = initial+row["contact_rise"]+max(0.,linear+.5*k)
    update_peak(peaks,"loss_upper_bound",abs(bound["upper_loss"]-upper))
    assert post["loss"]<=upper+1e-12
    assert "after_contact_saved" in forecast or upper < initial+LIMIT-1e-12
    extended = "after_contact_saved" in forecast
    if extended:
        assert forecast["after_contact_saved"]==post
    assert forecast["counts"]=={"value_calls":0,"gradient_calls":1,"path_calls":1+int(extended)}
    assert forecast["warning"]==(extended and post["loss"]-initial>LIMIT)
    return bound,extended


def audit(root):
    cfg,decisions,cost = [read(root/(name+".json")) for name in ("config","decisions","execution_counts")]
    parent = Path(cfg["parent"])
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    contexts,previous,run_rows = [read(parent/(n+".json")) for n in ("contexts","paired_outcomes","runs")]
    archived = {r["run_id"]:r for r in run_rows}
    assert len(contexts)==len(decisions)==len(previous)==cost["contexts"]
    assert cost["actual_new_forecast_queries"]<=cfg["forecast_query_cap"]
    assert cost["actual_new_continuation_queries"]==cost["continuations"]==0
    assert cost["reused_selected_continuations"]==len(contexts)
    branch_files = [parent/d/(key+suffix) for key in archived for d,suffix in (
        ("traces",".csv.gz"),("events",".json"),("fractions",".json"))]
    branch_hashes = checked_hashes(parent,branch_files)
    rows,late = [],[]
    peaks = {}
    policy_queries = Counter()
    for c,d,r in zip(contexts,decisions,previous,strict=True):
        key = c["context_id"]
        assert key==d["context_id"]==r["context_id"]
        keep = key+"__keep"
        capped = key+"__capped"
        events = read(parent/"events"/(keep+".json"))
        bound,extended = audit_forecast(c,d,r,events,peaks)
        forecast = d["forecast"]
        assert not forecast["warning"] or r["full"]["warning"]
        chosen = capped if forecast["warning"] else keep
        assert d["selected_run"]==chosen and d["reused"]
        run = archived[chosen]
        assert run["stop_reason"]=="completed" and run["committed_steps"]==STOP_STEP-BOUNDARY_STEP
        outcome = r["full"]["outcome"] if forecast["warning"] else r["keep"]
        row = {**r,"curvature":{"warning":forecast["warning"],"queries":sum(forecast["counts"].values()),"outcome":outcome},
               "bound_rise":bound["upper_loss"]-c["initial_loss"],"remaining_time":bound["remaining_time"],
               "extended":extended,"selected_run":chosen}
        for m in MODES:
            k = capped if row[m]["warning"] else keep
            policy_queries[m] += 4+row[m]["queries"]+archived[k]["counts"]["total_oracle_calls"]
        rows.append(row)
        if not r["target_positive"]:
            first = r["first_post_contact_saved_time"]
            second = (round(first/H)+1)*H
            assert abs(r["keep"]["first_saved_crossing"]-second)<1e-10
            trace = decode_rows(parent,keep)
            second_loss = trace[round(second/H)-BOUNDARY_STEP]["values"][0]
            late.append({"context_id":key,"case":r["case"],"seed":r["seed"],
                         "first_saved_time":first,"first_saved_rise":r["first_post_contact_saved_rise"],
                         "second_saved_time":second,"second_saved_rise":second_loss-c["initial_loss"],
                         "outcomes":{m:row[m]["outcome"] for m in MODES}})
    result = summarize(rows,modes=MODES)
    assert cost["actual_new_forecast_queries"]==result["curvature"]["queries"]
    recovered = [r["context_id"] for r in rows if r["target_positive"] and not r["gated"]["warning"]
                 and r["curvature"]["outcome"]["first_saved_crossing"] is None]
    lost = sum(r["full"]["outcome"]["first_saved_crossing"] is None and
               r["curvature"]["outcome"]["first_saved_crossing"] is not None for r in rows)
    if cfg["stage"]=="formal":
        assert len(late)==12 and len(rows)==48
    result.update(recovered_gate_misses=recovered,lost_full_preventions=lost,second_saved_cases=len(late),
                  total_policy_queries=dict(policy_queries),curvature_extensions=sum(r["extended"] for r in rows),
                  forecast_queries_saved_vs_full=result["full"]["queries"]-result["curvature"]["queries"],
                  reused_e46_ledgers=str(parent/"physical_ledgers.json"))
    create(root/"paired_outcomes.json",rows)
    create(root/"second_saved_cases.json",late)
    create(root/"reused_branch_hashes.json",branch_hashes)
    create(root/"results.json",result)
    record = {"passed":True,"contexts":len(rows),"continuations":0,"solver_failures":[],
              "max_residuals":peaks,"value_checks":0,"gradient_checks":0,"archived_branch_files":len(branch_hashes)}
    create(root/"audit.json",record)
    print(record)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
