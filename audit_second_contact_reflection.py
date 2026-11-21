"""Compare E50 to the archived E48 policy and verify the second bounce."""

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
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, H, LIMIT, START, STOP_STEP, read, sha
from run_second_contact_reflection import FORECAST_CAP

POLICIES = ("e48", "e50")


def check_forecast(parent, context, decision, peaks):
    baseline = context["e48"]
    forecast = decision["forecast"]
    truncated = "second_after_contact_saved" in baseline and baseline["second_after_contact_saved"] is None
    if not truncated:
        assert forecast == baseline
        return None

    changed = {"second_contact", "second_after_contact_saved", "warning", "counts"}
    assert {k:v for k,v in forecast.items() if k not in changed} == {k:v for k,v in baseline.items() if k not in changed}
    assert forecast["counts"] == {"value_calls":0, "gradient_calls":2, "path_calls":3}
    key = context["context_id"]+"__keep"
    contacts = [e for e in read(parent/"events"/(key+".json")) if e["i"]==0]
    second = contacts[1]
    t = BOUNDARY_STEP*H + second["time"]
    assert baseline["after_contact_saved"]["time"] < t < baseline["after_contact_saved"]["time"] + H
    checkpoint = forecast["second_contact"]
    rotation = np.asarray(context["rotation"])
    position = np.asarray(second["point_i"])
    x,y = rotation@position[:2]
    normal = np.r_[-rotation.T@[x**3-x+.05,8*y],1.]
    normal /= np.linalg.norm(normal)
    incoming = np.asarray(second["before_i"])
    independent_outgoing = incoming-2*(incoming@normal)*normal
    outgoing = np.asarray(checkpoint["outgoing_velocity"])
    update_peak(peaks,"second_normal",np.max(abs(normal-second["normal"])))
    update_peak(peaks,"second_time",abs(t-checkpoint["time"]))
    update_peak(peaks,"second_contact_loss",abs(position[-1]-checkpoint["loss"]))
    update_peak(peaks,"second_incoming_velocity",np.max(abs(incoming-checkpoint["incoming_velocity"])))
    update_peak(peaks,"second_outgoing_velocity",np.max(abs(independent_outgoing-outgoing)))
    update_peak(peaks,"second_archive_outgoing",np.max(abs(second["after_i"]-outgoing)))
    update_peak(peaks,"second_kinetic_change",abs(.5*(outgoing@outgoing-incoming@incoming)))
    step = baseline["after_contact_saved"]["step"]+1
    saved_time = step*H
    third = BOUNDARY_STEP*H+contacts[2]["time"] if len(contacts)>2 else None
    supported = third is None or third>saved_time
    prediction = forecast["second_after_contact_saved"]
    assert (prediction is not None) == supported
    recorded_loss = decode_rows(parent,key)[step-BOUNDARY_STEP]["values"][0]
    if supported:
        assert prediction["step"]==step and prediction["time"]==saved_time
        xx,yy = rotation@(position[:2]+(saved_time-t)*independent_outgoing[:2])
        expected = .25*(xx*xx-1.)**2+.05*xx+4.*yy**2
        update_peak(peaks,"second_independent_saved_loss",abs(expected-recorded_loss))
        update_peak(peaks,"second_predicted_saved_loss",abs(prediction["loss"]-recorded_loss))
    assert forecast["warning"] == (supported and recorded_loss-context["initial_loss"]>LIMIT)
    return {"second_contact_time":t,"third_contact_time":third,"second_saved_time":saved_time,
            "second_saved_loss":recorded_loss,"second_saved_rise":recorded_loss-context["initial_loss"],
            "second_contact_rise":position[-1]-context["initial_loss"],
            "incoming_velocity":incoming.tolist(),"outgoing_velocity":outgoing.tolist(),
            "archived_terrain_impulse":(independent_outgoing-incoming).tolist(),"supported":supported}


def audit(root):
    cfg, decisions, new_runs, costs = (read(root/(n+".json")) for n in ("config","decisions","runs","execution_counts"))
    parent = Path(cfg["parent"])
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==h for p,h in cfg["dependency_data_sha256"].items())
    contexts = {c["context_id"]:c for c in read(parent/"contexts.json")}
    previous = {r["context_id"]:r for r in read(parent/"paired_outcomes.json")}
    archived = {r["run_id"]:r for r in read(parent/"runs.json")}
    executed = {r["run_id"]:r for r in new_runs}
    assert len(decisions)==len(cfg["context_ids"])==costs["contexts"]
    assert [d["context_id"] for d in decisions]==cfg["context_ids"]
    assert len(executed)==len(new_runs)==costs["continuations"] and not archived.keys() & executed.keys()
    keys = {r["e48"]["run_id"] for r in previous.values() if r["context_id"] in cfg["context_ids"]}
    keys.update(k+"__keep" for k in cfg["context_ids"])
    hashes = checked_hashes(parent, [parent/d/(key+suffix) for key in sorted(keys) for d,suffix in
                                      (("traces",".csv.gz"),("events",".json"),("fractions",".json"))])
    counts, peaks = Counter(), dict(energy=0.,momentum=0.,reflection=0.)
    rows, ledgers, targeted = [], [], []
    for decision in decisions:
        key = decision["context_id"]
        c, old = contexts[key], previous[key]
        initial = c["initial_loss"]
        geometry = check_forecast(parent,c,decision,peaks)
        warning = decision["forecast"]["warning"]
        chosen = decision["selected_run"]
        assert chosen == key+("__capped" if warning else "__keep")
        reused = decision["reused"]
        if reused:
            assert chosen in archived and chosen==old["e48"]["run_id"]
            branch = archived[chosen]
            assert branch["stop_reason"]=="completed" and branch["committed_steps"]==STOP_STEP-BOUNDARY_STEP
            outcome = old["e48"]["outcome"]
        else:
            assert chosen not in archived and warning and not old["e48"]["warning"]
            branch = executed[chosen]
            checkpoint = {"time":START, "boundary_step":BOUNDARY_STEP, "remaining_duration":BOUNDARY_STEP*H-START,
                          "post_reflection":c["start"], "initial_loss":initial,
                          "already_deviated":False, "above_at_contact":False}
            _,_,_,loss,contacts,ledger = audit_branch(root,branch,checkpoint,c,counts,peaks,
                                                     fraction_audit=capped_fraction,end_step=STOP_STEP,query_cap=ADDED_BUDGET)
            saved = [((BOUNDARY_STEP+i)*H,float(f)) for i,f in enumerate(loss[:,0])]
            outcome = {**score(saved,contacts,initial), "final_loss":saved[-1][1],
                       "best_saved_loss":min(f for _,f in saved)}
            assert ledger["fraction"]["external_energy_supply"]==0.
            ledgers.append(ledger)
        forecast_queries = sum(decision["forecast"]["counts"].values())
        assert forecast_queries <= FORECAST_CAP
        row = {**old, "e50":{"warning":warning, "run_id":chosen, "outcome":outcome,
                             "forecast_queries":forecast_queries,
                             "total_queries":sum(c["setup_counts"].values())+forecast_queries+branch["counts"]["total_oracle_calls"]},
               "run_reused":reused, "resumed_geometry":geometry}
        if old["crossing_timing"]=="second_post_contact_saved" and not old["e48"]["warning"]:
            assert geometry is not None
            targeted.append(row)
        rows.append(row)

    expected_targets = 14 if cfg["stage"]=="formal" else 1
    assert len(targeted)==expected_targets and len(ledgers)==len(new_runs)
    assert costs["reused_selected_continuations"]==sum(r["run_reused"] for r in rows)
    assert costs["actual_new_forecast_queries"]==sum(r["e50"]["forecast_queries"] for r in rows)
    assert costs["actual_new_continuation_queries"]==sum(r["counts"]["total_oracle_calls"] for r in new_runs)
    assert costs["actual_new_queries"]==costs["actual_new_forecast_queries"]+costs["actual_new_continuation_queries"]<=cfg["max_new_queries"]
    result = summarize(rows,policies=POLICIES)
    result["target_cases"] = summarize(targeted,policies=POLICIES)
    result["by_keep_outcome"] = {label:summarize([r for r in rows if (r["keep"][METRIC] is None)==safe],policies=POLICIES)
                                 for label,safe in (("safe",True),("deviated",False))}
    result["second_contacts_resumed"] = sum(r["resumed_geometry"] is not None for r in rows)
    result["checks"] = {"target_forecasts_all_warned":result["target_cases"]["e50"]["transfers"]==expected_targets,
                        "target_cases_all_prevented":result["target_cases"]["e50"]["prevented"]==expected_targets,
                        "no_lost_preventions":result["lost_preventions"]==0,
                        "no_new_unnecessary_transfers":result["new_unnecessary_transfers"]==0}
    audit_result = {"passed":True, "contexts":len(rows), "continuations":len(new_runs), "solver_failures":[],
                   "max_residuals":peaks, "archived_branch_files":len(hashes), **counts}
    create(root/"paired_outcomes.json",rows)
    create(root/"second_contact_cases.json",targeted)
    create(root/"reused_branch_hashes.json",hashes)
    create(root/"physical_ledgers.json",{"new_continuations":ledgers,"archived_ledgers":str(parent/"physical_ledgers.json")})
    create(root/"results.json",result)
    create(root/"audit.json",audit_result)
    print({k:v for k,v in result.items() if k not in ("by_keep_outcome","target_cases")})
    print(audit_result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
