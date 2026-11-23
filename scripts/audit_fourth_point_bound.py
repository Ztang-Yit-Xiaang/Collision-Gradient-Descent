"""Audit E54's fourth-point screen and paired E52 continuation outcomes."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, update_peak
from audit_mixed_post_contact_curvature import METRIC, summarize
from audit_post_contact_positive_stress import score
from audit_third_point_bound import check_forecast
from audit_third_saved_point import inspect_forecast
from collision_optim.core import State
from collision_optim.curvature_energy import RotatedEnergyWells
from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import decode_rows
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, CFG, H, LIMIT, START, STOP_STEP, read, sha
from run_third_point_bound import contact_safe_bounds

POLICIES = ("e52","e54")


def verify_window(context, forecast, trace, events, peaks):
    first_three = check_forecast(context,context["e52"],context["e51"],trace,events,peaks)
    step = first_three[-1]["step"]+1
    fourth = {"step":step,"time":step*H,"loss":trace[step-BOUNDARY_STEP]["values"][0]}
    points = [*first_three,fourth]
    bounds = forecast.get("saved_loss_upper_bounds")
    added_risk = False
    if bounds is None:
        assert forecast==context["e52"]
    else:
        first = next(e for e in events if e["i"]==0)
        t = BOUNDARY_STEP*H+first["time"]
        point = np.asarray(first["point_i"])
        v = np.asarray(first["after_i"])
        reflected = State(point[:2],v[:2],point[2],v[2])
        expected = contact_safe_bounds(RotatedEnergyWells(8,context["seed"]),reflected,CFG.physics,t,step-3,step)
        for bound,reference,observed in zip(bounds,expected,points,strict=True):
            assert (bound is None)==(reference is None)
            if bound is not None:
                update_peak(peaks,"four_point_bound",abs(bound-reference))
                assert observed["loss"]<=bound+1e-12
        threshold = context["initial_loss"]+LIMIT
        safe = [v is not None and v<threshold-1e-12 for v in bounds]
        added_risk = not safe[-1]
        assert (forecast["window_stop"]=="loss_bounds")==all(safe)
        if all(safe):
            assert not forecast["warning"] and not forecast["extended_observations"] and not forecast["extended_contacts"]
            assert forecast["counts"]=={"value_calls":0,"gradient_calls":1,"path_calls":1}
        else:
            inspect_forecast(context,forecast,trace,events,peaks,saved_count=4 if added_risk else 3)

    initial = context["initial_loss"]
    times = [BOUNDARY_STEP*H+e["time"] for e in events if e["i"]==0]
    third_state = trace[step-1-BOUNDARY_STEP]
    rotation = np.asarray(context["rotation"])
    y = (rotation@third_state["positions"][0][:2])[1]
    vy = (rotation@third_state["velocities"][0][:2])[1]
    return {"following_saved_points":points,"saved_loss_upper_bounds":bounds,"fourth_point_risk":added_risk,
            "fourth_point_evaluated":any(p["step"]==step for p in forecast.get("extended_observations",[])),
            "third_point_transverse_yvy":float(y*vy),
            "contact_times_through_fourth":[t for t in times if t<=fourth["time"]],
            "contacts_between_third_and_fourth":[t for t in times if first_three[-1]["time"]<t<=fourth["time"]],
            "saved_loss_rises":[p["loss"]-initial for p in points]}


def audit(root):
    cfg,decisions,new_runs,cost = [read(root/(n+".json")) for n in ("config","decisions","runs","execution_counts")]
    parent = Path(cfg["parent"])
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==h for p,h in cfg["dependency_data_sha256"].items())
    contexts = {r["context_id"]:r for r in read(parent/"contexts.json")}
    previous = {r["context_id"]:r for r in read(parent/"paired_outcomes.json")}
    archived = {r["run_id"]:r for r in read(parent/"runs.json")}
    executed = {r["run_id"]:r for r in new_runs}
    assert [r["context_id"] for r in decisions]==cfg["context_ids"]
    assert len(decisions)==cost["contexts"] and len(executed)==len(new_runs)==cost["continuations"]
    assert not archived.keys() & executed.keys()
    keys = {k+"__keep" for k in cfg["context_ids"]}
    keys.update(previous[k]["e52"]["run_id"] for k in cfg["context_ids"])
    keys.update(d["selected_run"] for d in decisions if d["reused"])
    hashes = checked_hashes(parent,[parent/d/(key+suffix) for key in sorted(keys)
                                   for d,suffix in (("traces",".csv.gz"),("events",".json"),("fractions",".json"))])
    rows,ledgers,late = [],[],[]
    peaks,counts = dict(energy=0.,momentum=0.,reflection=0.),Counter()
    for d in decisions:
        key = d["context_id"]
        c,old = contexts[key],previous[key]
        initial = c["initial_loss"]
        keep = key+"__keep"
        forecast = d["forecast"]
        geometry = verify_window(c,forecast,decode_rows(parent,keep),read(parent/"events"/(keep+".json")),peaks)
        warning = forecast["warning"]
        chosen = d["selected_run"]
        assert chosen==key+("__capped" if warning else "__keep")
        if d["reused"]:
            branch = archived[chosen]
            assert chosen in (old["e52"]["run_id"],keep)
            assert branch["stop_reason"]=="completed" and branch["committed_steps"]==STOP_STEP-BOUNDARY_STEP
            outcome = old["e52"]["outcome"] if chosen==old["e52"]["run_id"] else old["keep"]
        else:
            assert warning and not old["e52"]["warning"]
            branch = executed[chosen]
            checkpoint = {"time":START,"boundary_step":BOUNDARY_STEP,"remaining_duration":BOUNDARY_STEP*H-START,
                          "post_reflection":c["start"],"initial_loss":initial,
                          "already_deviated":False,"above_at_contact":False}
            _,_,_,loss,contacts,ledger = audit_branch(root,branch,checkpoint,c,counts,peaks,
                                                      fraction_audit=capped_fraction,end_step=STOP_STEP,query_cap=ADDED_BUDGET)
            saved = [((BOUNDARY_STEP+i)*H,float(f)) for i,f in enumerate(loss[:,0])]
            outcome = {**score(saved,contacts,initial),"final_loss":saved[-1][1],
                       "best_saved_loss":min(f for _,f in saved)}
            assert ledger["fraction"]["external_energy_supply"]==0.
            ledgers.append(ledger)
        queries = sum(forecast["counts"].values())
        assert queries<=cfg["forecast_query_cap_each"]
        row = {**old,"e54":{"warning":warning,"run_id":chosen,"outcome":outcome,"forecast_queries":queries,
                           "total_queries":sum(c["setup_counts"].values())+queries+branch["counts"]["total_oracle_calls"]},
               "run_reused_in_e54":d["reused"],"four_point_geometry":geometry}
        if old["e52"]["outcome"][METRIC] is not None:
            late.append(row)
        rows.append(row)

    assert len(ledgers)==len(new_runs)
    assert cost["reused_selected_continuations"]==sum(r["run_reused_in_e54"] for r in rows)
    assert cost["actual_new_forecast_queries"]==sum(r["e54"]["forecast_queries"] for r in rows)
    assert cost["actual_new_continuation_queries"]==sum(r["counts"]["total_oracle_calls"] for r in new_runs)
    assert cost["actual_new_queries"]==cost["actual_new_forecast_queries"]+cost["actual_new_continuation_queries"]<=cfg["max_new_queries"]
    result = summarize(rows,policies=POLICIES)
    fourth = [r for r in late if r["e52"]["outcome"][METRIC]==r["four_point_geometry"]["following_saved_points"][3]["time"]]
    later = [r for r in late if r["e52"]["outcome"][METRIC]>r["four_point_geometry"]["following_saved_points"][3]["time"]]
    if cfg["stage"]=="formal":
        assert (len(late),len(fourth),len(later))==(21,7,14)
    result["previous_unprotected"] = {"fourth_point":summarize(fourth,policies=POLICIES),"later":summarize(later,policies=POLICIES)}
    result["by_keep_outcome"] = {label:summarize([r for r in rows if (r["keep"][METRIC] is None)==safe],policies=POLICIES)
                                 for label,safe in (("safe",True),("deviated",False))}
    result["fourth_point_evaluations"] = sum(r["four_point_geometry"]["fourth_point_evaluated"] for r in rows)
    result["checks"] = {"additional_preventions":result["additional_preventions"]>0,
                        "no_lost_preventions":result["lost_preventions"]==0,
                        "no_new_unnecessary_transfers":result["new_unnecessary_transfers"]==0}
    record = {"passed":True,"contexts":len(rows),"continuations":len(new_runs),"solver_failures":[],
              "max_residuals":peaks,"archived_branch_files":len(hashes),**counts}
    create(root/"paired_outcomes.json",rows)
    create(root/"previous_unprotected_cases.json",late)
    create(root/"reused_branch_hashes.json",hashes)
    create(root/"physical_ledgers.json",{"new_continuations":ledgers,"archived_ledgers":str(parent/"physical_ledgers.json")})
    create(root/"results.json",result)
    create(root/"audit.json",record)
    print({k:v for k,v in result.items() if k not in ("by_keep_outcome","previous_unprotected")})
    print(record)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
