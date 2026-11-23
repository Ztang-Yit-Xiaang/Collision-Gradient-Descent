"""Verify E51's paid reflection forecasts and the paired continuation outcomes."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, update_peak
from audit_mixed_post_contact_curvature import METRIC, summarize
from audit_post_contact_positive_stress import score
from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import decode_rows
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, CFG, H, LIMIT, START, STOP_STEP, read, sha

POLICIES = ("e50","e51")


def inspect_forecast(context, pred, trace, events, peaks, *, saved_count=3):
    initial = context["initial_loss"]
    contacts = [e for e in events if e["i"]==0]
    times = [BOUNDARY_STEP*H+e["time"] for e in contacts]
    first = pred["next_contact"]
    update_peak(peaks,"first_time",abs(first["time"]-times[0]))
    update_peak(peaks,"first_loss",abs(first["loss"]-contacts[0]["point_i"][-1]))
    step = int(np.ceil(times[0]/H))
    saved = [{"step":i,"time":i*H,"loss":trace[i-BOUNDARY_STEP]["values"][0]}
             for i in range(step,step+saved_count)]
    for point,raw in zip(pred["observations"],trace[:step-BOUNDARY_STEP],strict=True):
        update_peak(peaks,"prior_observation_loss",abs(point["loss"]-raw["values"][0]))
    prior_crossing = any(p["loss"]-initial>LIMIT for p in pred["observations"])
    stop = pred.get("window_stop")
    if prior_crossing:
        assert stop is None and pred["warning"]
        assert pred["counts"]=={"value_calls":0,"gradient_calls":0,"path_calls":1}
        return saved,times

    v = np.asarray(contacts[0]["after_i"])
    g, height_scale = CFG.physics.gravity, CFG.physics.height_scale
    upper = (contacts[0]["point_i"][-1]+.5*(v@v)/g)/height_scale
    update_peak(peaks,"energy_bound",abs(upper-pred["energy_upper_loss"]))
    assert all(p["loss"]<=upper+1e-10 for p in saved)
    if upper<initial+LIMIT-1e-12:
        assert stop=="energy_bound" and not pred["warning"]
        assert not pred["extended_contacts"] and not pred["extended_observations"]
        assert pred["counts"]=={"value_calls":0,"gradient_calls":1,"path_calls":1}
        return saved,times

    expected_stop = "completed"
    points = []
    for ordinal,point in enumerate(saved,1):
        reflection_cap = 4 if ordinal==4 else 3
        if len(times)>reflection_cap and times[reflection_cap]<=point["time"]:
            expected_stop = "reflection_limit"
            update_peak(peaks,"unresolved_contact_time",abs(times[reflection_cap]-pred["unresolved_contact_time"]))
            break
        points.append(point)
        if point["loss"]-initial>LIMIT:
            expected_stop = "saved_crossing"
            break
    assert stop==expected_stop and pred["warning"]==(stop=="saved_crossing")
    for predicted, actual in zip(pred["extended_observations"],points,strict=True):
        assert predicted["step"]==actual["step"] and predicted["time"]==actual["time"]
        update_peak(peaks,"saved_loss",abs(predicted["loss"]-actual["loss"]))
    end = times[reflection_cap] if stop=="reflection_limit" else points[-1]["time"]
    resolved = [(t,e) for t,e in zip(times[1:reflection_cap],contacts[1:reflection_cap],strict=True) if t<=end]
    assert pred["counts"]=={"value_calls":0,"gradient_calls":1+len(resolved),"path_calls":2+len(resolved)}
    rotation = np.asarray(context["rotation"])
    for forecast,(t,event) in zip(pred["extended_contacts"],resolved,strict=True):
        incoming = np.asarray(event["before_i"])
        x,y = rotation@event["point_i"][:2]
        normal = np.r_[-height_scale*rotation.T@[x**3-x+.05,8*y],1.]
        normal /= np.linalg.norm(normal)
        outgoing = incoming-2*(incoming@normal)*normal
        update_peak(peaks,"contact_time",abs(t-forecast["time"]))
        update_peak(peaks,"contact_loss",abs(event["point_i"][-1]-forecast["loss"]))
        update_peak(peaks,"incoming_velocity",np.max(abs(incoming-forecast["incoming_velocity"])))
        update_peak(peaks,"outgoing_velocity",np.max(abs(outgoing-forecast["outgoing_velocity"])))
        update_peak(peaks,"reflection_kinetic_change",abs(.5*(outgoing@outgoing-incoming@incoming)))
    return saved,times


def audit(root):
    cfg,decisions,new_runs,cost = [read(root/(n+".json")) for n in ("config","decisions","runs","execution_counts")]
    parent,baseline = Path(cfg["parent"]),Path(cfg["comparison"])
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(path))==sha256 for path,sha256 in cfg["dependency_data_sha256"].items())
    contexts = {c["context_id"]:c for c in read(parent/"contexts.json")}
    previous = {r["context_id"]:r for r in read(baseline/"paired_outcomes.json")}
    archived = {r["run_id"]:(p,r) for p in (parent,baseline) for r in read(p/"runs.json")}
    executed = {r["run_id"]:r for r in new_runs}
    assert len(decisions)==len(cfg["context_ids"])==cost["contexts"]
    assert [d["context_id"] for d in decisions]==cfg["context_ids"]
    assert len(executed)==len(new_runs)==cost["continuations"] and not archived.keys() & executed.keys()

    keys = {k+"__keep" for k in cfg["context_ids"]}
    keys.update(previous[k]["e50"]["run_id"] for k in cfg["context_ids"])
    keys.update(d["selected_run"] for d in decisions if d["reused"])
    hashes = {}
    for p in (parent,baseline):
        hashes.update(checked_hashes(p,[p/folder/(k+suffix) for k in sorted(keys) if archived[k][0]==p
                                        for folder,suffix in (("traces",".csv.gz"),("events",".json"),("fractions",".json"))]))
    rows,ledgers,late = [],[],[]
    counts,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    for d in decisions:
        key = d["context_id"]
        c,old = contexts[key],previous[key]
        initial = c["initial_loss"]
        keep = key+"__keep"
        forecast = d["forecast"]
        points,times = inspect_forecast(c,forecast,decode_rows(parent,keep),read(parent/"events"/(keep+".json")),peaks)
        warning = forecast["warning"]
        chosen = d["selected_run"]
        assert chosen==key+("__capped" if warning else "__keep")
        if d["reused"]:
            branch = archived[chosen][1]
            assert branch["stop_reason"]=="completed" and branch["committed_steps"]==STOP_STEP-BOUNDARY_STEP
            outcome = old["e50"]["outcome"] if chosen==old["e50"]["run_id"] else old["keep"]
            assert chosen==old["e50"]["run_id"] or chosen==keep
        else:
            assert chosen not in archived and warning and not old["e50"]["warning"]
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
        row = {**old,"e51":{"warning":warning,"run_id":chosen,"outcome":outcome,"forecast_queries":queries,
                           "total_queries":sum(c["setup_counts"].values())+queries+branch["counts"]["total_oracle_calls"]},
               "run_reused_in_e51":d["reused"],"third_window_stop":forecast.get("window_stop"),
               "first_three_saved_rises":[[p["time"],p["loss"]-initial] for p in points],
               "contacts_through_third_point":[t for t in times if t<=points[2]["time"]],
               "contacts_between_second_and_third_points":[t for t in times if points[1]["time"]<t<=points[2]["time"]]}
        if old["e50"]["outcome"][METRIC] is not None:
            late.append(row)
        rows.append(row)

    assert len(ledgers)==len(new_runs)
    assert cost["reused_selected_continuations"]==sum(r["run_reused_in_e51"] for r in rows)
    assert cost["actual_new_forecast_queries"]==sum(r["e51"]["forecast_queries"] for r in rows)
    assert cost["actual_new_continuation_queries"]==sum(r["counts"]["total_oracle_calls"] for r in new_runs)
    assert cost["actual_new_queries"]==cost["actual_new_forecast_queries"]+cost["actual_new_continuation_queries"]<=cfg["max_new_queries"]
    result = summarize(rows,policies=POLICIES)
    third = [r for r in late if r["e50"]["outcome"][METRIC]==r["first_three_saved_rises"][2][0]]
    subsequent = [r for r in late if r["e50"]["outcome"][METRIC]>r["first_three_saved_rises"][2][0]]
    if cfg["stage"]=="formal":
        assert len(late)==37 and len(third)==16 and len(subsequent)==21
    result["previous_unprotected"] = {"third_point":summarize(third,policies=POLICIES),
                                       "later":summarize(subsequent,policies=POLICIES)}
    result["by_keep_outcome"] = {label:summarize([r for r in rows if (r["keep"][METRIC] is None)==safe],policies=POLICIES)
                                 for label,safe in (("safe",True),("deviated",False))}
    result["forecast_stops"] = dict(Counter(d["forecast"].get("window_stop","prior_observation") for d in decisions))
    result["checks"] = {"additional_preventions":result["additional_preventions"]>0,
                        "no_lost_preventions":result["lost_preventions"]==0,
                        "no_new_unnecessary_transfers":result["new_unnecessary_transfers"]==0}
    audit_record = {"passed":True,"contexts":len(rows),"continuations":len(new_runs),"solver_failures":[],
                    "max_residuals":peaks,"archived_branch_files":len(hashes),**counts}
    create(root/"paired_outcomes.json",rows)
    create(root/"previous_unprotected_cases.json",late)
    create(root/"reused_branch_hashes.json",hashes)
    create(root/"physical_ledgers.json",{"new_continuations":ledgers,
        "archived_ledgers":[str(p/"physical_ledgers.json") for p in (parent,baseline)]})
    create(root/"results.json",result)
    create(root/"audit.json",audit_record)
    print({k:v for k,v in result.items() if k not in ("by_keep_outcome","previous_unprotected")})
    print(audit_record)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
