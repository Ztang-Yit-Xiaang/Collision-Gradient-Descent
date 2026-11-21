"""Verify E52's geometric bound and matched decisions against archived trajectories."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, update_peak
from audit_mixed_post_contact_curvature import METRIC, summarize
from audit_post_contact_positive_stress import score
from audit_third_saved_point import inspect_forecast
from collision_optim.core import State
from collision_optim.curvature_energy import RotatedEnergyWells
from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import decode_rows
from run_post_contact_positive_stress import ADDED_BUDGET, BOUNDARY_STEP, CFG, H, LIMIT, START, STOP_STEP, read, sha
from run_third_point_bound import contact_safe_bounds

POLICIES = ("e51","e52")


def check_forecast(context, pred, baseline, trace, events, peaks):
    saved,times = inspect_forecast(context,baseline,trace,events,peaks)
    bounds = pred.get("saved_loss_upper_bounds")
    if bounds is None:
        assert pred==baseline
        return saved

    first = next(e for e in events if e["i"]==0)
    v = np.asarray(first["after_i"])
    reflected = State(np.asarray(first["point_i"][:2]),v[:2],first["point_i"][-1],v[-1])
    objective = RotatedEnergyWells(8,context["seed"])
    expected = contact_safe_bounds(objective,reflected,CFG.physics,times[0],saved[0]["step"],saved[-1]["step"])
    for bound,reference,observed in zip(bounds,expected,saved,strict=True):
        assert (bound is None)==(reference is None)
        if bound is not None:
            update_peak(peaks,"recomputed_upper_loss",abs(bound-reference))
            assert observed["loss"]<=bound+1e-12

    threshold = context["initial_loss"]+LIMIT
    can_skip = all(b is not None and b<threshold-1e-12 for b in bounds)
    assert (pred["window_stop"]=="loss_bounds")==can_skip
    exact_fields = {k:v for k,v in pred.items() if k!="saved_loss_upper_bounds"}
    if can_skip:
        assert all(p["loss"]<=threshold for p in saved)
        assert pred["counts"]=={"value_calls":0,"gradient_calls":1,"path_calls":1}
        assert not pred["extended_observations"] and not pred["extended_contacts"] and not pred["warning"]
        assert pred["next_contact"]==baseline["next_contact"] and pred["observations"]==baseline["observations"]
    else:
        assert exact_fields==baseline
    return saved


def audit(root):
    cfg,decisions,new_runs,cost = [read(root/(n+".json")) for n in ("config","decisions","runs","execution_counts")]
    parent,baseline = Path(cfg["parent"]),Path(cfg["comparison"])
    sources = [parent,*map(Path,cfg["earlier_archives"]),baseline]
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(path))==digest for path,digest in cfg["dependency_data_sha256"].items())
    contexts = {c["context_id"]:c for c in read(parent/"contexts.json")}
    previous = {r["context_id"]:r for r in read(baseline/"paired_outcomes.json")}
    forecasts = {r["context_id"]:r["forecast"] for r in read(baseline/"decisions.json")}
    archived = {r["run_id"]:(p,r) for p in sources for r in read(p/"runs.json")}
    executed = {r["run_id"]:r for r in new_runs}
    assert [d["context_id"] for d in decisions]==cfg["context_ids"]
    assert len(decisions)==cost["contexts"] and len(executed)==len(new_runs)==cost["continuations"]
    assert not archived.keys() & executed.keys()
    keys = {k+"__keep" for k in cfg["context_ids"]}
    keys.update(previous[k]["e51"]["run_id"] for k in cfg["context_ids"])
    keys.update(d["selected_run"] for d in decisions if d["reused"])
    hashes = {}
    for p in sources:
        hashes.update(checked_hashes(p,[p/d/(key+suffix) for key in sorted(keys) if archived[key][0]==p
                                        for d,suffix in (("traces",".csv.gz"),("events",".json"),("fractions",".json"))]))
    peaks,counts = dict(energy=0.,momentum=0.,reflection=0.),Counter()
    rows,ledgers = [],[]
    for d in decisions:
        key = d["context_id"]
        c,old = contexts[key],previous[key]
        pred = d["forecast"]
        keep = key+"__keep"
        saved = check_forecast(c,pred,forecasts[key],decode_rows(parent,keep),read(parent/"events"/(keep+".json")),peaks)
        warning = pred["warning"]
        chosen = d["selected_run"]
        assert chosen==key+("__capped" if warning else "__keep")
        if d["reused"]:
            branch = archived[chosen][1]
            assert chosen in (old["e51"]["run_id"],keep)
            assert branch["stop_reason"]=="completed" and branch["committed_steps"]==STOP_STEP-BOUNDARY_STEP
            outcome = old["e51"]["outcome"] if chosen==old["e51"]["run_id"] else old["keep"]
        else:
            assert warning and not old["e51"]["warning"]
            branch = executed[chosen]
            initial = c["initial_loss"]
            checkpoint = {"time":START,"boundary_step":BOUNDARY_STEP,"remaining_duration":BOUNDARY_STEP*H-START,
                          "post_reflection":c["start"],"initial_loss":initial,
                          "already_deviated":False,"above_at_contact":False}
            _,_,_,loss,contacts,ledger = audit_branch(root,branch,checkpoint,c,counts,peaks,
                                                      fraction_audit=capped_fraction,end_step=STOP_STEP,query_cap=ADDED_BUDGET)
            points = [((BOUNDARY_STEP+i)*H,float(f)) for i,f in enumerate(loss[:,0])]
            outcome = {**score(points,contacts,initial),"final_loss":points[-1][1],
                       "best_saved_loss":min(f for _,f in points)}
            assert ledger["fraction"]["external_energy_supply"]==0.
            ledgers.append(ledger)
        queries = sum(pred["counts"].values())
        assert queries<=cfg["forecast_query_cap_each"]
        row = {**old,"e52":{"warning":warning,"run_id":chosen,"outcome":outcome,"forecast_queries":queries,
                           "total_queries":sum(c["setup_counts"].values())+queries+branch["counts"]["total_oracle_calls"]},
               "run_reused_in_e52":d["reused"],"e52_window_stop":pred.get("window_stop"),
               "saved_loss_upper_bounds":pred.get("saved_loss_upper_bounds"),
               "audited_saved_losses":[p["loss"] for p in saved]}
        rows.append(row)
    assert len(ledgers)==len(new_runs)
    assert cost["reused_selected_continuations"]==sum(r["run_reused_in_e52"] for r in rows)
    assert cost["actual_new_forecast_queries"]==sum(r["e52"]["forecast_queries"] for r in rows)
    assert cost["actual_new_continuation_queries"]==sum(r["counts"]["total_oracle_calls"] for r in new_runs)
    assert cost["actual_new_queries"]==cost["actual_new_forecast_queries"]+cost["actual_new_continuation_queries"]<=cfg["max_new_queries"]
    protected = [r for r in rows if r["e51"]["warning"] and not r["e50"]["warning"]]
    skipped = [r for r in rows if r["e52_window_stop"]=="loss_bounds"]
    if cfg["stage"]=="formal":
        assert len(protected)==16
    result = summarize(rows,policies=POLICIES)
    result["e51_third_point_preventions"] = summarize(protected,policies=POLICIES)
    result["loss_bound_skips"] = summarize(skipped,policies=POLICIES)
    result["by_keep_outcome"] = {label:summarize([r for r in rows if (r["keep"][METRIC] is None)==safe],policies=POLICIES)
                                 for label,safe in (("safe",True),("deviated",False))}
    result["checks"] = {"all_e51_third_preventions_retained":all(r["e52"]["outcome"][METRIC] is None for r in protected),
                        "no_lost_preventions":result["lost_preventions"]==0,
                        "no_new_unnecessary_transfers":result["new_unnecessary_transfers"]==0,
                        "forecast_queries_reduced":result["e52"]["forecast_queries"]<result["e51"]["forecast_queries"]}
    record = {"passed":True,"contexts":len(rows),"continuations":len(new_runs),"solver_failures":[],
              "max_residuals":peaks,"archived_branch_files":len(hashes),"loss_bound_skips":len(skipped),**counts}
    create(root/"paired_outcomes.json",rows)
    create(root/"e51_third_point_preventions.json",protected)
    create(root/"loss_bound_skips.json",skipped)
    create(root/"reused_branch_hashes.json",hashes)
    create(root/"physical_ledgers.json",{"new_continuations":ledgers,
        "archived_ledgers":[str(p/"physical_ledgers.json") for p in sources]})
    create(root/"results.json",result)
    create(root/"audit.json",record)
    print({k:v for k,v in result.items() if k not in ("by_keep_outcome","e51_third_point_preventions","loss_bound_skips")})
    print(record)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
