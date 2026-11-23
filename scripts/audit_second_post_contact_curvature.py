"""Compare E48 against E47 and independently check the longer flight bound."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, update_peak
from audit_post_contact_positive_stress import score
from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import decode_rows
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, H, LIMIT, START, STOP_STEP, read, sha


def check_prediction(c,d,row,trace,events,peaks):
    forecast = d["forecast"]
    initial = c["initial_loss"]
    original = c["full"]
    assert forecast["observations"]==original["observations"]
    assert forecast["next_contact"]==original["next_contact"]
    first = original["after_contact_saved"]
    step = first["step"]
    t = row["contact_time"]
    event,*later = [e for e in events if e["i"]==0]
    assert abs(BOUNDARY_STEP*H+event["time"]-t)<1e-12
    next_contact = min((BOUNDARY_STEP*H+e["time"] for e in later),default=float("inf"))
    assert next_contact>step*H
    rotation = np.asarray(c["rotation"])
    x,y = rotation@event["point_i"][:2]
    v = np.asarray(event["after_i"])
    remaining = first["time"]-t
    bound = forecast["post_contact_bound"]
    update_peak(peaks,"remaining_time",abs(bound["remaining_time"]-remaining))
    update_peak(peaks,"outgoing_velocity",float(np.max(abs(v-bound["outgoing_velocity"]))))
    beta = float(trace[step+1-BOUNDARY_STEP]["drag_beta"])
    observations = []
    for idx,amount,upper in ((step,remaining,bound["upper_loss"]),
                              (step+1,remaining+beta*H,forecast["second_saved_upper_loss"])):
        dx,dy = rotation@(amount*v[:2])
        curvature = max(0.,3.*max(x*x,(x+dx)**2)-1.)*dx**2+8.*dy**2
        slope = (x**3-x+.05)*dx+8.*y*dy
        independently_bounded = initial+row["contact_rise"]+max(0.,slope+.5*curvature)
        update_peak(peaks,"upper_loss",abs(upper-independently_bounded))
        loss = trace[idx-BOUNDARY_STEP]["values"][0]
        free_flight = next_contact>idx*H
        if free_flight:
            assert loss<=upper+1e-12
        observations.append({"step":idx,"time":idx*H,"loss":loss,"free_flight":free_flight})
    paid = max(bound["upper_loss"],forecast["second_saved_upper_loss"])>=initial+LIMIT-1e-12
    assert forecast["counts"]=={"value_calls":0,"gradient_calls":1,"path_calls":1+int(paid)}
    expected = observations[:1]
    if observations[0]["loss"]<=initial+LIMIT and observations[1]["free_flight"]:
        expected.append(observations[1])
    predictions = [forecast[k] for k in ("after_contact_saved","second_after_contact_saved") if forecast.get(k) is not None]
    if paid:
        for pred,actual in zip(predictions,expected,strict=True):
            assert pred["step"]==actual["step"] and pred["time"]==actual["time"]
            update_peak(peaks,"saved_loss",abs(pred["loss"]-actual["loss"]))
    else:
        assert not predictions
    assert forecast["warning"]==any(o["loss"]-initial>LIMIT for o in expected)
    return observations


def audit(root):
    cfg,decisions,cost,runs = [read(root/(n+".json")) for n in ("config","decisions","execution_counts","runs")]
    parent,comparison = Path(cfg["parent"]),Path(cfg["comparison"])
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    contexts,archived = [read(parent/(n+".json")) for n in ("contexts","runs")]
    previous = read(comparison/"paired_outcomes.json")
    ledger_count,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    archives = {r["run_id"]:r for r in archived}
    executed = {r["run_id"]:r for r in runs}
    keys = [parent/d/(r["run_id"]+ext) for r in archived for d,ext in (
        ("traces",".csv.gz"),("fractions",".json"),("events",".json"))]
    hashes = checked_hashes(parent,keys)
    paired,ledgers = [],[]
    totals = Counter()
    for c,d,r in zip(contexts,decisions,previous,strict=True):
        key = c["context_id"]
        assert key==d["context_id"]==r["context_id"]
        keep = key+"__keep"
        observations = check_prediction(c,d,r,decode_rows(parent,keep),read(parent/"events"/(keep+".json")),peaks)
        selected = d["selected_run"]
        forecast = d["forecast"]
        assert selected==key+("__capped" if forecast["warning"] else "__keep")
        if d["reused"]:
            branch = archives[selected]
            assert branch["stop_reason"]=="completed" and branch["committed_steps"]==STOP_STEP-BOUNDARY_STEP
            outcome = r["curvature"]["outcome"] if r["curvature"]["warning"]==forecast["warning"] else r["keep"]
            assert not forecast["warning"] or r["curvature"]["warning"]
        else:
            cp = {"time":START,"boundary_step":BOUNDARY_STEP,"remaining_duration":BOUNDARY_STEP*H-START,
                  "post_reflection":c["start"],"initial_loss":c["initial_loss"],"already_deviated":False,"above_at_contact":False}
            branch = executed[selected]
            _,_,_,f,contacts,ledger = audit_branch(root,branch,cp,c,ledger_count,peaks,fraction_audit=capped_fraction,
                                                   end_step=STOP_STEP,query_cap=ADDED_BUDGET)
            saved = [((BOUNDARY_STEP+i)*H,float(v)) for i,v in enumerate(f[:,0])]
            outcome = score(saved,contacts,c["initial_loss"])
            ledgers.append(ledger)
        forecast_queries = sum(forecast["counts"].values())
        e47_queries = r["curvature"]["queries"]
        e47_branch = archives[r["selected_run"]]
        totals.update(e47_forecasts=e47_queries,e48_forecasts=forecast_queries,e47_total=4+e47_queries+e47_branch["counts"]["total_oracle_calls"],
                      e48_total=4+forecast_queries+branch["counts"]["total_oracle_calls"])
        paired.append({**r,"extended_window":{"warning":forecast["warning"],"queries":forecast_queries,"outcome":outcome},
                       "first_two_saved_rises":[[o["time"],o["loss"]-c["initial_loss"]] for o in observations],
                       "free_flight_through_second_saved_point":observations[-1]["free_flight"],
                       "selected_e48_run":selected,"e48_run_reused":d["reused"],
                       "e47_selected_solver_queries":e47_branch["counts"]["total_oracle_calls"],
                       "e48_selected_solver_queries":branch["counts"]["total_oracle_calls"]})
    assert len(paired)==len(decisions)==cost["contexts"]
    assert len(ledgers)==len(runs)==cost["continuations"]
    assert cost["actual_new_forecast_queries"]==totals["e48_forecasts"]<=cfg["forecast_query_cap"]
    assert cost["actual_new_continuation_queries"]==sum(r["counts"]["total_oracle_calls"] for r in runs)
    assert cost["actual_new_queries"]==cost["actual_new_forecast_queries"]+cost["actual_new_continuation_queries"]
    late = [r for r in paired if not r["target_positive"]]
    early = [r for r in paired if r["target_positive"]]
    summaries = {}
    for label,mode in (("e47","curvature"),("e48","extended_window")):
        summaries[label] = {"transfers":sum(r[mode]["warning"] for r in paired),
            "unnecessary_transfers":sum(r[mode]["warning"] and r["keep"]["first_saved_crossing"] is None for r in paired),
            "deviated_saved":sum(r[mode]["outcome"]["first_saved_crossing"] is not None for r in paired),
            "deviated_including_contact":sum(r[mode]["outcome"]["first_crossing_including_contacts"] is not None for r in paired),
            "final_safe":sum(r[mode]["outcome"]["within_limit_at_end"] for r in paired)}
    result = {"contexts":len(paired),"first_saved_cases":len(early),"second_saved_cases":len(late),
              "first_saved_preventions_retained":sum(r["extended_window"]["outcome"]["first_saved_crossing"] is None for r in early),
              "second_saved_preventions_added":sum(r["extended_window"]["outcome"]["first_saved_crossing"] is None for r in late),
              **totals,**summaries}
    create(root/"paired_outcomes.json",paired)
    create(root/"second_saved_cases.json",late)
    create(root/"reused_branch_hashes.json",hashes)
    create(root/"physical_ledgers.json",{"new_continuations":ledgers,"archived_ledgers":str(parent/"physical_ledgers.json")})
    create(root/"results.json",result)
    record = {"passed":True,"contexts":len(paired),"continuations":len(runs),"solver_failures":[],
              "max_residuals":peaks,"archived_branch_files":len(hashes),**ledger_count}
    create(root/"audit.json",record)
    print(result)
    print(record)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
