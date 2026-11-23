"""Audit later-reflection timing, matched continuations, and actual query costs."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from audit_contact_time_transfer import physical
from audit_shared_prefix_transfer import segment_audit
from finalize_initial_budget import create
from run_later_reflection_warning import BOUNDARY_STEP, CAP, END_STEP, H, LIMIT, read, sha, decode_rows


def absolute_points(trace, events, start_step):
    saved = [(int(r["step"])*H+start_step*H,r["values"][0]) for r in trace]
    contacts = [(start_step*H+e["time"],e["point_i"][-1]) for e in events if e["i"]==0]
    return saved,contacts


def first_cross(points, initial):
    return next((t for t,f in sorted(points) if f-initial>LIMIT),None)


def score(saved, contacts, checkpoint):
    initial = checkpoint["initial_loss"]
    crossed = first_cross(saved,initial)
    any_cross = first_cross(saved+contacts,initial)
    first_recovery = first_recross = None
    if checkpoint["already_deviated"]:
        first_recovery = checkpoint["time"] if not checkpoint["above_at_contact"] else next(
            (t for t,f in sorted(saved+contacts) if f-initial<=LIMIT),None)
        if first_recovery is not None:
            first_recross = first_cross([(t,f) for t,f in saved+contacts if t>first_recovery],initial)
    return {"first_saved_crossing":crossed,"first_crossing_including_contacts":any_cross,
            "first_recovery":first_recovery,"first_recross":first_recross,
            "final_rise":float(saved[-1][1]-initial),"peak_saved_rise":float(max(f for t,f in saved)-initial),
            "best_saved_loss":float(min(f for t,f in saved)),"within_limit_at_end":bool(saved[-1][1]-initial<=LIMIT)}


def update_peak(peaks, name, error, tolerance=1e-10):
    assert error<tolerance,(name,error)
    peaks[name]=max(peaks.get(name,0.),float(error))


def check_fraction(frac, checkpoint, mode, peaks, *, expected_horizontal=None):
    q, old = (np.array(checkpoint["post_reflection"][k]) for k in ("positions","velocities"))
    v = np.array(frac["start"]["velocities"])
    assert np.array_equal(q,frac["start"]["positions"]) and np.array_equal(v[:,2],old[:,2])
    if expected_horizontal is not None:
        assert np.allclose(v[:,:2],expected_horizontal,atol=1e-12,rtol=0)
    elif mode=="keep":
        assert np.array_equal(v,old)
    else:
        center = old[:,:2].mean(axis=0)
        relative = old[:,:2]-center
        rest = relative[1:]-relative[1:].mean(axis=0)
        assert np.allclose(v[0,:2],center,atol=1e-12,rtol=0)
        assert np.allclose(v[1:,:2],center+rest*np.linalg.norm(relative)/np.linalg.norm(rest),atol=1e-12,rtol=0)
    k0,e0,p0 = physical(q,old)
    k1,e1,p1 = physical(q,v)
    q2,v2 = (np.array(frac["end"][k]) for k in ("positions","velocities"))
    k2,e2,p2 = physical(q2,v2)
    diag = frac["diagnostics"]
    assert frac["duration"]==checkpoint["remaining_duration"] and frac["damping_beta"]==1.
    assert diag["damping_energy"]==0. and not np.any(diag["damping_impulse"])
    update_peak(peaks,"map_energy",abs(e1-e0))
    update_peak(peaks,"map_momentum",np.max(abs(p1-p0)))
    update_peak(peaks,"fraction_energy",abs(e2-e1-diag["impulse_energy"]))
    update_peak(peaks,"fraction_momentum",np.max(abs(p2-p1-np.array(diag["gravity_impulse"])-diag["wall_impulse"])))
    work = -.5*float(np.sum(q2[:,2]-q[:,2]))
    update_peak(peaks,"fraction_work",abs(k2-k1-work-diag["impulse_energy"]))
    return {"map_energy_change":e1-e0,"map_total_linear_impulse":(p1-p0).tolist(),
            "map_particle_impulses":(v-old).tolist(),"map_angular_momentum_change":np.cross(q,v-old).sum(axis=0).tolist(),
            "external_energy_supply":0.,"fraction_gravity_work":work,"fraction_terrain_work":diag["impulse_energy"],
            "fraction_gravity_impulse":diag["gravity_impulse"],"fraction_terrain_impulse":diag["wall_impulse"],
            "fraction_damping_heat":0.,"fraction_energy_residual":diag["energy_residual"],
            "fraction_linear_momentum_residual":diag["momentum_residual"]}


def audit_prediction(checkpoint, trace, source_contacts, peaks):
    counts = checkpoint["forecast_counts"]
    if checkpoint["already_deviated"]:
        assert sum(counts.values())==0 and not checkpoint["warning"] and checkpoint["prediction"] is None
        return
    assert counts=={"value_calls":0,"gradient_calls":0,"path_calls":1}
    pred = checkpoint["prediction"]
    after = next(((t,f) for t,f in source_contacts if t>checkpoint["time"]+1e-10),None)
    hit = pred["next_contact"]
    assert (hit is None)==(after is None)
    if hit:
        update_peak(peaks,"forecast_contact_time",abs(hit["time"]-after[0]))
        update_peak(peaks,"forecast_loss",abs(hit["loss"]-after[1]))
    start, end = checkpoint["boundary_step"], END_STEP+1
    if hit:
        end = min(end,int(np.ceil(hit["time"]/H)))
    assert [p["step"] for p in pred["observations"]]==list(range(start,end))
    for p in pred["observations"]:
        update_peak(peaks,"forecast_loss",abs(p["loss"]-trace[p["step"]-BOUNDARY_STEP]["values"][0]))
    points = pred["observations"]+([hit] if hit else [])
    first = next((p for p in points if p["loss"]-checkpoint["initial_loss"]>LIMIT),None)
    assert checkpoint["first_predicted_crossing"]==first and checkpoint["warning"]==(first is not None)


def audit_branch(root, run, checkpoint, context, counts, peaks, *, fraction_audit=check_fraction,
                 end_step=END_STEP, query_cap=CAP):
    key = run["run_id"]
    assert run["stop_reason"]=="completed" and run["committed_steps"]==end_step-checkpoint["boundary_step"],run
    frac = read(root/"fractions"/(key+".json"))
    ledger = fraction_audit(frac,checkpoint,run["policy"],peaks)
    (q,v,f,calls,events),tail = segment_audit(root,run,context,counts,peaks,start_step=checkpoint["boundary_step"])
    assert np.array_equal(q[0],frac["end"]["positions"]) and np.array_equal(v[0],frac["end"]["velocities"])
    assert np.array_equal(f[0],frac["values"])
    fraction_end = frac["total_oracle_calls_at_end"]
    assert run["actual_fraction_calls"]+run.get("charged_prefix_queries",0)==fraction_end
    assert calls[0]==fraction_end+4 and calls[-1]==run["counts"]["total_oracle_calls"]<=query_cap
    assert run["actual_tail_calls"]==calls[-1]-fraction_end
    saved = [((checkpoint["boundary_step"]+i)*H,y) for i,y in enumerate(f[:,0])]
    contacts = [(checkpoint["boundary_step"]*H+e["time"],e["point_i"][-1]) for e in events if e["i"]==0]
    contacts[:0] = [(checkpoint["time"]+e["time"],e["point_i"][-1]) for e in frac["events"] if e["i"]==0]
    return score(saved,contacts,checkpoint),q,v,f,contacts,{"run_id":key,"fraction":ledger,"tail":tail}


def mechanism(row, saved, contacts):
    c2,c3 = row["checkpoints"]
    initial = c2["initial_loss"]
    first = first_cross(saved,initial)
    if first is None or first<=1.+1e-10:
        return None
    def contact_geometry(c):
        e = c["event"]
        before,after = (np.array(e[k])[:2] for k in ("before_i","after_i"))
        gradient = -np.array(e["normal"][:2])/e["normal"][-1]
        return {"time":c["time"],"loss_rise":c["contact_rise"],
                "horizontal_kinetic_before":float(.5*(before@before)),
                "horizontal_kinetic_after":float(.5*(after@after)),
                "horizontal_impulse":(after-before).tolist(),
                "directional_loss_rate_before":float(gradient@before),"directional_loss_rate_after":float(gradient@after)}
    return {"context_id":row["context_id"],"seed":row["seed"],"incumbent_energy":row["incumbent_energy"],
            "first_saved_crossing":first,"first_crossing_including_contacts":first_cross(saved+contacts,initial),
            "first_saved_crossing_after_third":first>c3["time"],"t1_rise":min(saved,key=lambda r:abs(r[0]-1.))[1]-initial,
            "peak_second_flight_rise":max(f for t,f in [*saved,*contacts] if c2["time"]<=t<=c3["time"]+1e-10)-initial,
            "second":contact_geometry(c2),"third":contact_geometry(c3)}


def summarize(pairs, diagnostic):
    def recovery(group,mode):
        outcomes = [r["branches"][mode] for r in group]
        return {"states":len(outcomes),"recovered":sum(r["first_recovery"] is not None for r in outcomes),
                "recrossed":sum(r["first_recross"] is not None for r in outcomes),
                "within_limit_at_end":sum(r["within_limit_at_end"] for r in outcomes)}
    summary = {}
    for metric in ("saved","including_contacts"):
        k = "deviated_"+metric
        positives = [r for r in pairs if r["baseline"][k]]
        alerts = [r for r in pairs if r["selected_warning"] is not None]
        leads = [r["lead_saved"] for r in alerts if r["lead_saved"] is not None]
        summary[metric] = {
            "baseline_deviated":len(positives),"policy_deviated":sum(r["policy"][k] for r in pairs),
            "warnings":len(alerts),"timely_true_positive":sum(r["selected_warning"] is not None for r in positives),
            "false_positive":sum(not r["baseline"][k] for r in alerts),
            "missed":sum(r["selected_warning"] is None for r in positives),
            "prevented":sum(not r["policy"][k] for r in positives),
            "induced":sum(not r["baseline"][k] and r["policy"][k] for r in pairs),
            "warned_still_deviated":sum(r["baseline"][k] and r["policy"][k] for r in alerts),
            "median_lead_to_saved_crossing":median(leads) if leads else None}
    safe = summary["saved"]
    checks = {"at_least_75_percent_timely":safe["timely_true_positive"]>=.75*safe["baseline_deviated"],
              "at_least_half_prevented":safe["prevented"]>=.5*safe["baseline_deviated"],
              "no_induced_deviations":safe["induced"]==0,"at_most_10_percent_unnecessary":safe["false_positive"]<=.1*safe["warnings"]}
    return {"contexts":len(pairs),"new_independent_seed_groups":0,**summary,
            "policy_forecast_queries":sum(r["policy_forecast_queries"] for r in pairs),
            "selected_warning_ordinals":dict(Counter(r["selected_warning"]["ordinal"] for r in pairs if r["selected_warning"])),
            "recovery":{name:{mode:recovery(group,mode) for mode in ("keep","transfer")} for name,group in (
                ("all",diagnostic),("above_at_contact",[r for r in diagnostic if r["above_at_contact"]]),
                ("historical_only",[r for r in diagnostic if not r["above_at_contact"]]))},
            "checks":checks,"passed_screen":all(checks.values())}


def audit(root):
    cfg = read(root/"config.json")
    parent = Path(cfg["parent"])
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    originals = {r["context_id"]:r for r in read(parent/"contexts.json")}
    cases,runs,costs = (read(root/(name+".json")) for name in ("contexts","runs","execution_counts"))
    assert len(cases)==costs["contexts"]==(6 if cfg["stage"]=="preflight" else 480)
    assert len(runs)==costs["continuations"]
    by_checkpoint = defaultdict(dict)
    for r in runs:
        assert r["policy"] not in by_checkpoint[r["checkpoint_id"]]
        by_checkpoint[r["checkpoint_id"]][r["policy"]]=r
    counts,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    pairs, diagnostic, details, ledgers, mechanisms = [],[],[],[],[]
    for row in cases:
        assert all(sha(Path(p))==digest for p,digest in row["source_dependencies"].items())
        context = originals[row["context_id"]]
        key = row["source_branch"]["run_id"]
        trace = decode_rows(parent,key)
        saved,contacts = absolute_points(trace,read(parent/"events"/(key+".json")),BOUNDARY_STEP)
        assert len(contacts)==2
        initial = context["initial_loss"]
        family = key.rsplit("__",2)[-2]
        before_tail_best = initial+min(context["families"][family]["history_rises"])
        source_cross = first_cross(saved,initial)
        source_any_cross = first_cross(saved+contacts,initial)
        assert (source_cross is not None)==row["source_branch"]["extended"]["deviated"]
        baseline = {"deviated_saved":source_cross is not None,"deviated_including_contacts":source_any_cross is not None,
                    "first_saved_crossing":source_cross,"first_crossing_including_contacts":source_any_cross,
                    "final_rise":saved[-1][1]-initial,"best_saved_loss":min(before_tail_best,*(f for _,f in saved))}
        policy,selected,queries = baseline,None,0
        for checkpoint in row["checkpoints"]:
            audit_prediction(checkpoint,trace,contacts,peaks)
            paid = sum(checkpoint["forecast_counts"].values())
            counts["forecast_queries"]+=paid
            if selected is None:
                queries+=paid
            branches = {}
            for mode,r in by_checkpoint[checkpoint["checkpoint_id"]].items():
                result,q,v,f,event_points,ledger = audit_branch(root,r,checkpoint,context,counts,peaks)
                branches[mode]=result
                ledgers.append(ledger)
                if mode=="keep":
                    start = checkpoint["boundary_step"]-BOUNDARY_STEP
                    for column,actual in (("positions",q),("velocities",v),("values",f)):
                        error = np.max(abs(actual-np.array([x[column] for x in trace[start:]])))
                        update_peak(peaks,"keep_parent_overlap",error)
                    expected = [p for p in contacts if p[0]>checkpoint["time"]+1e-10]
                    assert len(expected)==len(event_points)
                    if expected:
                        update_peak(peaks,"keep_contact_overlap",np.max(abs(np.array(expected)-event_points)))
            assert len(branches)==2*int(checkpoint["warning"] or checkpoint["already_deviated"] or checkpoint["engineering_pair"])
            d = {"context_id":row["context_id"],**checkpoint,"branches":branches}
            details.append(d)
            if checkpoint["already_deviated"]:
                diagnostic.append(d)
            if checkpoint["warning"] and selected is None:
                selected = {"checkpoint_id":checkpoint["checkpoint_id"],"ordinal":checkpoint["ordinal"],"time":checkpoint["time"]}
                moved = branches["transfer"]
                policy = {**moved,"deviated_saved":moved["first_saved_crossing"] is not None,
                          "deviated_including_contacts":moved["first_crossing_including_contacts"] is not None,
                          "best_saved_loss":min(before_tail_best,moved["best_saved_loss"],
                                                *(f for t,f in saved if t<checkpoint["time"]))}
        pairs.append({"context_id":row["context_id"],"seed":row["seed"],"incumbent_energy":row["incumbent_energy"],
                      "baseline":baseline,"policy":policy,"selected_warning":selected,"policy_forecast_queries":queries,
                      "lead_saved":source_cross-selected["time"] if source_cross is not None and selected else None})
        if result := mechanism(row,saved,contacts):
            mechanisms.append(result)
    assert counts["forecast_queries"]==costs["actual_forecast_queries"]
    assert costs["actual_fraction_queries"]==sum(r["actual_fraction_calls"] for r in runs)
    assert costs["actual_tail_queries"]==sum(r["actual_tail_calls"] for r in runs)
    assert costs["actual_new_solver_queries"]==sum(v for k,v in costs.items() if k.startswith("actual_") and k!="actual_new_solver_queries")
    assert costs["actual_new_solver_queries"]<=cfg["new_query_cap"]
    if cfg["stage"]=="replay":
        assert len(mechanisms)==19
    create(root/"paired_outcomes.json",pairs)
    create(root/"warning_checkpoints.json",details)
    create(root/"already_deviated_cases.json",diagnostic)
    create(root/"late_crossing_mechanisms.json",mechanisms)
    create(root/"physical_ledgers.json",ledgers)
    create(root/"results.json",summarize(pairs,diagnostic))
    result = {"passed":True,"contexts":len(cases),"continuations":len(runs),"solver_failures":[],
              "max_residuals":peaks,**counts}
    create(root/"audit.json",result)
    print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
