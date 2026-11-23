"""Audit E42 forecasts, conservation, timing, and paired outcomes."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch, check_fraction, first_cross, update_peak
from audit_second_reflection_capped_policy import recovery, recovery_totals
from finalize_initial_budget import create
from run_third_reflection_guard import ADDED_BUDGET, Archive, E40, E41, EPS, H, LIMIT, STOP_STEP, checkpoint, history, read, sha


def audit_forecast(d, saved, hits, peaks):
    cp = d["checkpoint"]
    initial = cp["initial_loss"]
    if cp["already_deviated"]:
        assert not d["forecast_queries"] and not d["transferred"] and not d["warning"] and d["prediction"] is None
        return None
    assert d["forecast_queries"]==1
    predicted = d["prediction"]
    expected = next(((t,f) for t,f in hits if t>cp["time"]+EPS),None)
    actual = predicted["next_contact"]
    assert (actual is None)==(expected is None)
    if actual is not None:
        update_peak(peaks,"forecast_contact_time",abs(actual["time"]-expected[0]))
        update_peak(peaks,"forecast_loss",abs(actual["loss"]-expected[1]))
    expected_steps = [i for i,(t,f) in enumerate(saved) if t>cp["time"] and (expected is None or t<expected[0])]
    assert [p["step"] for p in predicted["observations"]]==expected_steps
    for p in predicted["observations"]:
        update_peak(peaks,"forecast_loss",abs(p["loss"]-saved[p["step"]][1]))
    pred_points = [(p["time"],p["loss"]) for p in predicted["observations"]+([actual] if actual else [])]
    first = first_cross(pred_points,initial)
    assert d["warning"]==(first is not None)
    return first


def outcomes(saved,hits,initial):
    return {"saved":first_cross(saved,initial),"including_contacts":first_cross(saved+hits,initial),
            "after_t2_saved":first_cross([(t,f) for t,f in saved if t>2.+EPS],initial),
            "after_t2_with_contacts":first_cross([(t,f) for t,f in saved+hits if t>2.+EPS],initial),
            "within_limit_at_end":saved[-1][1]-initial<=LIMIT,"final_rise":saved[-1][1]-initial}


def summary(rows):
    selected = [r for r in rows if r["late_target"]]
    alerts = [r for r in rows if r["warning"]]
    transfers = [r for r in rows if r["transferred"]]
    leads = [r["baseline"]["after_t2_with_contacts"]-r["checkpoint_time"] for r in selected if r["warning"]]
    skipped = [r["retained_recovery"] for r in rows if r["retained_recovery"] is not None]
    result = {"contexts":len(rows),"late_targets":len(selected),"late_targets_alerted":sum(r["warning"] for r in selected),
              "late_targets_prevented":sum(r["policy"]["after_t2_with_contacts"] is None for r in selected),
              "late_targets_already_deviated":sum(r["retained_recovery"] is not None for r in selected),
              "median_lead":median(leads) if leads else None,"alerts":len(alerts),"new_transfers":len(transfers),
              "new_forecast_queries":sum(r["forecast_queries"] for r in rows),
              "retained_recovery":{label:recovery_totals(group) for label,group in (
                  ("all",skipped),("still_above",[r for r in skipped if r["above_at_contact"]]),
                  ("already_within",[r for r in skipped if not r["above_at_contact"]]))}}
    for k in ("saved","including_contacts","after_t2_saved","after_t2_with_contacts"):
        pairs = [(r["baseline"][k],r["policy"][k]) for r in rows]
        result[k] = {"baseline":sum(a is not None for a,b in pairs),"new_policy":sum(b is not None for a,b in pairs),
                     "prevented":sum(a is not None and b is None for a,b in pairs),
                     "induced":sum(a is None and b is not None for a,b in pairs),
                     "earlier":sum(a is not None and b is not None and b<a-EPS for a,b in pairs),
                     "unnecessary_transfers":sum(r["baseline"][k] is None for r in transfers)}
    result["within_limit_at_end"] = [sum(r[m]["within_limit_at_end"] for r in rows) for m in ("baseline","policy")]
    return result


def audit(root,*,data=None,parent_root=E40,extension=E41,case_root=None,extra=None):
    cfg = read(root/"config.json")
    if case_root is None:
        case_root = root
        assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    data = Archive() if data is None else data
    orig = {c["context_id"]:c for c in data.read(parent_root/"contexts.json")}
    parent = {c["context_id"]:c["capped_policy"] for c in data.read(extension/"paired_outcomes.json")} if extra is None else None
    cases = read(case_root/"contexts.json")
    decisions,run_rows,costs = [read(root/(name+".json")) for name in ("decisions","runs","execution_counts")]
    runs = {(r["context_id"],r["policy"]):r for r in run_rows}
    assert len(cases)==len(decisions)==costs["contexts"]==cfg["contexts"] and len(runs)==costs["continuations"]
    counts,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    rows,ledgers,diagnosis = [],[],[]
    for case,d in zip(cases,decisions,strict=True):
        key = case["context_id"]
        assert key==d["context_id"]
        ctx = orig[key]
        saved,hits,raw,events,start = history(data,case,ctx,parent=parent_root,extension=extension)
        assert all(data.dependencies[p]==digest for p,digest in d["dependencies"].items())
        _,cp = checkpoint(case,ctx,saved,hits,raw,events,start)
        assert d["checkpoint"]==cp
        update_peak(peaks,"event_reconstruction",cp["reconstruction_error"])
        initial,time = cp["initial_loss"],cp["time"]
        predicted_first = audit_forecast(d,saved,hits,peaks)
        old = outcomes(saved,hits,initial)
        if parent is not None:
            expected = parent[key]
            assert old["saved"]==expected["good_saved"]["first_crossing"]
            assert abs((old["including_contacts"] or 0)-(expected["good_with_contacts"]["first_crossing"] or 0))<EPS
            assert abs((old["after_t2_with_contacts"] or 0)-(expected["good_with_contacts"]["first_after_t2_crossing"] or 0))<EPS
        new = old
        assert d["transferred"]==((key,"capped") in runs)
        for mode,checker in (("keep",check_fraction),("capped",capped_fraction)):
            if (key,mode) not in runs:
                continue
            run = runs[key,mode]
            _,q,v,f,new_contacts,ledger = audit_branch(root,run,cp,ctx,counts,peaks,fraction_audit=checker,
                                                      end_step=STOP_STEP,query_cap=ADDED_BUDGET)
            tail = [((cp["boundary_step"]+i)*H,float(y)) for i,y in enumerate(f[:,0])]
            prefix = [(t,y) for t,y in saved if t<time]
            changed_hits = [(t,y) for t,y in hits if t<=time+EPS]+new_contacts
            evaluated = outcomes(prefix+tail,changed_hits,initial)
            if mode=="keep":
                update_peak(peaks,"keep_loss_overlap",max(abs(a[1]-b[1]) for a,b in zip(prefix+tail,saved,strict=True)))
                assert (evaluated["saved"] is None)==(old["saved"] is None)
                if old["saved"] is not None:
                    assert abs(evaluated["saved"]-old["saved"])<EPS
            else:
                new = evaluated
            ledgers.append(ledger)
        recovery_row = None
        if cp["already_deviated"]:
            recovery_row = {"above_at_contact":cp["above_at_contact"],
                            **recovery(saved+hits,initial,time,not cp["above_at_contact"]),
                            "within_limit_at_end":old["within_limit_at_end"]}
        late = old["after_t2_with_contacts"] is not None
        if late:
            e = cp["event"]
            before,after = np.array(e["before_i"][:2]),np.array(e["after_i"][:2])
            normal = np.array(e["normal"])
            slope = -normal[:2]/normal[-1]
            diagnosis.append({"context_id":key,"seed":case["seed"],"incumbent_energy":case["incumbent_energy"],
                              "third_contact_time":time,"third_contact_rise":cp["contact_rise"],
                              "t2_rise":saved[40][1]-initial,"next_contact_time":hits[3][0],"next_contact_rise":hits[3][1]-initial,
                              "first_after_t2_crossing":old["after_t2_with_contacts"],
                              "kinetic_before":.5*float(before@before),"kinetic_after":.5*float(after@after),
                              "horizontal_impulse":(after-before).tolist(),"directional_loss_rate_before":float(slope@before),
                              "directional_loss_rate_after":float(slope@after),"already_deviated":cp["already_deviated"]})
        extra_fields = extra(case,d,ctx,saved,hits,raw,events,start,runs,counts,peaks) if extra else {}
        window = {}
        if cfg.get("saved_window"):
            predicted = d["prediction"] or {}
            after = predicted.get("after_contact_saved")
            pre = any(p["loss"]-initial>LIMIT for p in predicted.get("observations",[]))
            extended = not pre and predicted.get("next_contact") is not None
            assert d["saved_forecast_queries"]==d["forecast_queries"]+2*int(extended)
            assert d["saved_warning"]==(pre or (after is not None and after["loss"]-initial>LIMIT))
            if after is not None:
                assert extended
                assert after["time"]==saved[after["step"]][0]
                update_peak(peaks,"after_contact_saved_loss",abs(after["loss"]-saved[after["step"]][1]))
            window = {"saved_window_forecast_queries":d["saved_forecast_queries"],
                      "saved_window_warning":d["saved_warning"],"saved_window_transfer":d["saved_warning"] and d["transferred"],
                      "original_transfer":d["warning"] and d["transferred"],"extended_forecast":extended,
                      "incomplete_saved_forecast":extended and after is None,
                      "original_policy":new if d["warning"] else old,"saved_window_policy":new if d["saved_warning"] else old}
        rows.append({"context_id":key,"seed":case["seed"],"incumbent_energy":case["incumbent_energy"],
                     "checkpoint_time":time,"late_target":late,"predicted_first_crossing":predicted_first,
                     **{k:d[k] for k in ("warning","transferred","forecast_queries")},"baseline":old,"policy":new,
                     "retained_recovery":recovery_row,**extra_fields,**window})
    assert costs["forecast_queries"]==sum(r.get("saved_window_forecast_queries",r["forecast_queries"]) for r in rows)
    for k in ("fraction","tail"):
        assert costs[k+"_queries"]==sum(r["actual_"+k+"_calls"] for r in run_rows)
    assert costs["actual_new_queries"]==sum(costs[k] for k in ("forecast_queries","fraction_queries","tail_queries"))<=cfg.get("new_query_cap",len(cases)*(ADDED_BUDGET+1))
    if cfg["stage"]=="replay":
        assert len(diagnosis)==15
    create(root/"paired_outcomes.json",rows)
    create(root/"already_deviated_cases.json",[r for r in rows if r["retained_recovery"] is not None])
    create(root/"late_crossing_mechanisms.json",diagnosis)
    create(root/"physical_ledgers.json",ledgers)
    create(root/"results.json",summary(rows))
    result = {"passed":True,"contexts":len(rows),"continuations":len(runs),"solver_failures":[],"max_residuals":peaks,**counts}
    create(root/"audit.json",result)
    print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
