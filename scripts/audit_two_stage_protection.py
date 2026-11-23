"""Audit the shared timing tree and derive four policy outcomes from its branches."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_contact_long_forecast import audit_extension
from audit_contact_time_transfer import audit_fraction
from audit_contact_warning_replication import event_audit
from audit_high_kinetic_intervention_timing import early_map, outcome, summarize
from audit_shared_prefix_transfer import segment_audit
from collision_optim.prebounce_warning import flight_factors, forecast_rise
from finalize_initial_budget import create, sha
from run_two_stage_protection import BOUNDARY_STEP, CAP, END_STEP, ENERGIES, H, LIMIT, POLICIES, PRECONTACT_STEP, PRIMARY_STEP, SEEDS, read

WINDOWS = {"primary": PRIMARY_STEP*H, "extended": END_STEP*H}


def policy_record(row, policy):
    use_early = row["early_warning"] and policy in ("early_only", "two_stage")
    kind = "early" if use_early else "base"
    f = row["families"][kind]
    use_bounce = f["complete"] and f["warning"] and policy in ("reflection_only", "two_stage")
    mode = "transfer" if use_bounce else "keep"
    return {
        "family":kind,"mode":mode,"early_transfer":use_early,"bounce_transfer":use_bounce,
        "forecast_queries":f["forecast_queries"] if policy in ("reflection_only","two_stage") else 0,
    }


def lookup(row, selection):
    return row["families"][selection["family"]]["branches"][selection["mode"]]


def by_policy(rows, policy, horizon):
    selected = [(r, r["policies"][policy]) for r in rows]
    values = [lookup(r,s) for r,s in selected]
    baseline = [r["families"]["base"]["branches"]["keep"] for r,_ in selected]
    previous = [r["families"][s["family"]]["branches"]["keep"] for r,s in selected]
    early_n = sum(s["early_transfer"] for _,s in selected)
    bounce_n = sum(s["bounce_transfer"] for _,s in selected)
    early_unneeded = sum(s["early_transfer"] and not b[horizon]["deviated"] for (_,s),b in zip(selected,baseline))
    bounce_unneeded = sum(s["bounce_transfer"] and not p[horizon]["deviated"] for (_,s),p in zip(selected,previous))
    safe_reflection = sum(s["early_transfer"] and not lookup(r,r["policies"]["reflection_only"])[horizon]["deviated"]
                          for r,s in selected)
    return {
        **summarize(values,horizon),
        "prevented_vs_keep":sum(b[horizon]["deviated"] and not y[horizon]["deviated"] for b,y in zip(baseline,values)),
        "induced_vs_keep":sum(not b[horizon]["deviated"] and y[horizon]["deviated"] for b,y in zip(baseline,values)),
        "early_transfers":early_n,"bounce_transfers":bounce_n,"total_transfers":early_n+bounce_n,
        "unnecessary_early_transfers":early_unneeded,"unnecessary_bounce_transfers":bounce_unneeded,
        "unnecessary_transfers":early_unneeded+bounce_unneeded,
        "early_transfers_with_reflection_only_already_safe":safe_reflection,
        "forecast_queries":sum(s["forecast_queries"] for _,s in selected),
    }


def recovery_groups(rows):
    result={}
    for kind in ("base","early"):
        late=[r["families"][kind] for r in rows if kind in r["families"]
              and r["families"][kind]["complete"] and r["families"][kind]["already_deviated"]]
        result[kind]={"states":len(late)}
        for mode in ("keep","transfer"):
            branches=[r["branches"][mode] for r in late]
            result[kind][mode]={
                name:{**summarize(branches,name),
                      "recrossed":sum(b["first_recross"] is not None and b["first_recross"]<=time for b in branches)}
                for name,time in WINDOWS.items()
            }
    return result


def evaluate(rows):
    usable=[r for r in rows if all(f["complete"] and all(b["complete"] for b in f["branches"].values())
                                  for f in r["families"].values())]
    metrics={name:{p:by_policy(usable,p,name) for p in POLICIES} for name in WINDOWS}
    main=metrics["primary"]
    base=main["keep"]["deviated"]
    both=main["two_stage"]
    combined=[]
    for r in usable:
        select=r["policies"]["two_stage"]
        f=r["families"][select["family"]]
        combined.append(f["branches"]["transfer"] if f["already_deviated"] else lookup(r,select))
    checks={
        "at_least_20_control_deviations":base>=20,
        "prevents_at_least_90_percent":bool(base and both["prevented_vs_keep"]/base>=.90),
        "prevents_at_least_5_more_than_each_single_stage":both["prevented_vs_keep"]>=
            max(main[p]["prevented_vs_keep"] for p in ("early_only","reflection_only"))+5,
        "unnecessary_at_most_10_percent":bool(both["total_transfers"] and
            both["unnecessary_transfers"]/both["total_transfers"]<=.10),
        "all_complete":len(usable)==len(rows),
    }
    return {
        "contexts":len(rows),"complete_policy_contexts":len(usable),"seed_groups":len({r["seed"] for r in usable}),
        "policy_metrics":metrics,"by_initial_kinetic":{str(e):{
            p:by_policy([r for r in usable if r["incumbent_energy"]==e],p,"primary") for p in POLICIES
        } for e in ENERGIES},
        "recovery":recovery_groups(usable),
        "two_stage_with_diagnostic_recovery":{name:summarize(combined,name) for name in WINDOWS},
        "early_already_deviated":sum(r["already_deviated_at_step1"] for r in usable),
        "recomputed_early_contact_families":sum("early" in r["families"] for r in usable),
        "early_family_bounce_warnings":sum(r["families"]["early"]["warning"] for r in usable if "early" in r["families"]),
        "checks":checks,"passed_screen":all(checks.values()),
    }


def audit_family(root,c,f,kind,prefix,branches,direct,counts,peaks,ledgers,base_rises):
    record={"complete":f["complete"],"branches":{},"forecast_queries":f["actual_forecast_queries"]}
    if not f["complete"]:
        assert not branches
        return {**record,"unavailable_reason":f["unavailable_reason"]}
    q,v,loss,calls,events=prefix
    local_start={"base":0,"early":1,"second":2}[kind]
    assert len(q)==PRECONTACT_STEP-local_start+1 and not any(e["i"]==0 for e in events)
    event_audit({**c,**f},q[-1],v[-1],counts,peaks)
    assert f["actual_contact_queries"]==10 and f["charged_at_fork"]==calls[-1]+10+f["actual_forecast_queries"]
    rises=loss[:,0]-c["initial_loss"]
    rises=np.r_[base_rises[:local_start],rises]
    assert np.array_equal(rises,f["history_rises"])
    assert f["contact_rise"]==f["contact"]["contact_value"]-c["initial_loss"]
    assert f["already_deviated"]==(max(*rises,f["contact_rise"])>LIMIT)
    assert set(branches)==({"keep","transfer"} if f["warning"] or f["already_deviated"] else {"keep"})
    record.update(warning=f["warning"],already_deviated=f["already_deviated"],contact_time=f["contact_time"],
                  contact_rise=f["contact_rise"])
    for mode,r in branches.items():
        raw,tail=segment_audit(root,r,c,counts,peaks,start_step=BOUNDARY_STEP)
        frac=read(root/"fractions"/(r["run_id"]+".json"))
        map_ledger=audit_fraction(frac,f,mode,peaks)
        assert np.array_equal(raw[0][0],frac["end"]["positions"]) and np.array_equal(raw[1][0],frac["end"]["velocities"])
        assert np.array_equal(raw[2][0],frac["values"])
        assert frac["total_oracle_calls_at_end"]==f["charged_at_fork"]+r["actual_fraction_calls"]
        assert raw[3][0]==frac["total_oracle_calls_at_end"]+4
        assert r["counts"]["total_oracle_calls"]==f["charged_at_fork"]+r["actual_suffix_calls"]<=CAP
        ledgers.append({"run_id":r["run_id"],"intervention_or_fraction":map_ledger,"tail":tail})
        b=outcome({**f,"already_deviated_at_reflection":f["already_deviated"]},raw,r,mode)
        recovery=b["first_saved_recovery"]
        b["first_recross"]=next(((i+BOUNDARY_STEP)*H for i,value in enumerate(raw[2][:,0])
                                 if (i+BOUNDARY_STEP)*H>recovery and value-c["initial_loss"]>LIMIT),
                                None) if recovery is not None else None
        record["branches"][mode]=b
        if mode=="keep":
            if f["already_deviated"]:
                assert f["prediction"] is None and f["actual_forecast_queries"]==0
            else:
                pred=f["prediction"]
                obs=pred["observations"]
                p35=obs[0]["loss_rise"] if obs else None
                validation=audit_extension({
                    "already_deviated":False,"actual_forecast_calls":f["actual_forecast_queries"],
                    "extended_forecast":pred,"initial_loss":c["initial_loss"],"predicted_boundary_rise":p35,
                },raw,peaks)
                assert f["warning"]==validation["extended_warning"]
                record["forecast_complete"]=validation["extended_forecast_complete"]
            if kind=="base" and direct:
                for x,y in zip(direct[:3],raw[:3]):
                    error=float(np.max(abs(x[BOUNDARY_STEP:BOUNDARY_STEP+len(y)]-y)))
                    assert error<1e-12
                    peaks["uninterrupted_overlap"]=max(peaks["uninterrupted_overlap"],error)
                assert all(q==direct[3][BOUNDARY_STEP+i]+4+f["actual_forecast_queries"] for i,q in enumerate(raw[3]))
    return record


def audit(root, *, seed_sets=SEEDS, evaluator=evaluate, second_stage=False):
    cfg,contexts,prefixes,runs,direct,costs=(read(root/(n+".json")) for n in (
        "config","contexts","prefixes","runs","uninterrupted","execution_counts"))
    assert len(contexts)==costs["contexts"]==len(seed_sets[cfg["stage"]])*len(ENERGIES)
    assert {(c["seed"],c["incumbent_energy"]) for c in contexts}=={
        (s,e) for s in seed_sets[cfg["stage"]] for e in ENERGIES}
    assert len(prefixes)==costs["prefix_executions"]==len(contexts)+sum(
        c["early_warning"] or c.get("second_warning",False) or c.get("contact_guard_alert",False) for c in contexts)
    assert len(runs)==costs["continuations"] and len(direct)==(len(contexts) if cfg["stage"]=="preflight" else 0)
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    for p,digest in cfg["dependency_data_sha256"].items():
        assert sha(Path(p))==digest
    with zipfile.ZipFile(root/"source.zip") as archive:
        for p,digest in cfg["source_sha256"].items():
            assert hashlib.sha256(archive.read(p)).hexdigest()==digest
    families=[f for c in contexts for f in c["families"].values()]
    assert costs["actual_prefix_queries"]==sum(p["actual_prefix_calls"] for p in prefixes)
    assert costs["actual_contact_queries"]==sum(f["actual_contact_queries"] for f in families)
    assert costs["actual_forecast_queries"]==sum(f["actual_forecast_queries"] for f in families)
    assert costs.get("actual_contact_guard_queries",0)==sum(c.get("contact_guard_queries",0) for c in contexts)
    assert costs["actual_continuation_queries"]==sum(r["actual_suffix_calls"] for r in runs)
    assert costs["actual_uninterrupted_queries_excluding_shared_initial_values"]==sum(
        d["counts"]["total_oracle_calls"]-4 for d in direct)
    assert costs["actual_solver_queries"]==sum(v for k,v in costs.items()
        if k.startswith("actual_") and k not in ("actual_solver_queries","actual_queries_including_setup"))
    assert costs["actual_queries_including_setup"]==costs["actual_solver_queries"]+4*len(contexts)
    by_prefix={(r["initial_context_id"],r["family"]):r for r in prefixes}
    by_direct={r["initial_context_id"]:r for r in direct}
    by_run=defaultdict(dict)
    for r in runs:
        by_run[r["initial_context_id"],r["family"]][r["policy"]]=r
    counts=Counter()
    peaks=dict(energy=0.,momentum=0.,reflection=0.,map_energy=0.,map_momentum=0.,fraction_energy=0.,
               fraction_momentum=0.,contact_energy=0.,contact_momentum=0.,uninterrupted_overlap=0.)
    rows,ledgers=[],[]
    for c in contexts:
        name=c["context_id"]
        raw,ledger=segment_audit(root,by_prefix[name,"base"],c,counts,peaks)
        ledgers.append({"run_id":by_prefix[name,"base"]["run_id"],"prefix":ledger})
        q,v,f,calls,events=raw
        assert calls[0]==8 and calls[1]==20 and np.array_equal(q[0],c["start"]["positions"])
        assert np.array_equal(v[0],c["start"]["velocities"]) and np.array_equal(f[0],c["values"])
        assert abs(.5*np.sum(v[0]**2)-.6)<1e-12 and abs(.5*np.sum(v[0,0]**2)-c["incumbent_energy"])<1e-12
        assert np.array_equal(q[1],c["early_state"]["positions"]) and np.array_equal(v[1],c["early_state"]["velocities"])
        rises=f[:2,0]-c["initial_loss"]
        score=forecast_rise(rises[1:2],float(np.linalg.norm(v[0,0,:2])),flight_factors(),cfg["early_prior_curvature"])
        assert score==c["early_score"]
        assert c["already_deviated_at_step1"]==(max(rises)>LIMIT)
        assert c["early_warning"]==(not c["already_deviated_at_step1"] and score>LIMIT)
        expected_families={"base"}
        if c["early_warning"]:
            expected_families.add("early")
        if second_stage:
            expected_already=not c["early_warning"] and np.max(f[:3,0]-c["initial_loss"])>LIMIT
            expected_score=None if c["early_warning"] or expected_already else forecast_rise(
                f[1:3,0]-c["initial_loss"],float(np.linalg.norm(v[0,0,:2])),flight_factors())
            assert c["second_score"]==expected_score and c["already_deviated_at_step2"]==expected_already
            assert c["second_warning"]==(expected_score is not None and expected_score>LIMIT)
            assert np.array_equal(q[2],c["second_state"]["positions"]) and np.array_equal(v[2],c["second_state"]["velocities"])
            assert np.array_equal(v[0,:,:2],c["initial_horizontal_velocities"])
            if "contact_guard_fraction" in cfg:
                queried=expected_score is not None and cfg["contact_guard_fraction"]*LIMIT<=expected_score<=LIMIT
                assert c["contact_guard_queries"]==int(queried)
                pred=c["contact_guard_prediction"]
                assert pred is None or queried
                assert c["contact_guard_alert"]==(pred is not None and pred["loss"]-c["initial_loss"]>LIMIT)
                if pred is not None:
                    observed=c["families"]["base"]
                    if observed["complete"]:
                        for name_,err in (
                            ("first_contact_guard_time",abs(pred["time"]-observed["contact_time"])),
                            ("first_contact_guard_loss",abs(pred["loss"]-observed["contact"]["contact_value"])),
                        ):
                            assert err<1e-12
                            peaks[name_]=max(peaks.get(name_,0.),err)
            if c["second_warning"] or c.get("contact_guard_alert",False):
                expected_families.add("second")
        assert set(c["families"])==expected_families
        if name in by_direct:
            full,d_ledger=segment_audit(root,by_direct[name],c,counts,peaks)
            ledgers.append({"run_id":by_direct[name]["run_id"],"uninterrupted":d_ledger})
            if c["families"]["base"]["complete"]:
                first=next(e for e in full[4] if e["i"]==0)
                assert abs(first["time"]-c["families"]["base"]["contact_time"])<1e-12
        else:
            full=None
        row={k:c[k] for k in ("context_id","seed","incumbent_energy","early_warning","already_deviated_at_step1")}
        if second_stage:
            row.update(second_warning=c["second_warning"],second_score=c["second_score"],
                       already_deviated_at_step2=c["already_deviated_at_step2"])
        if "contact_guard_fraction" in cfg:
            row.update({k:c[k] for k in ("contact_guard_queries","contact_guard_prediction","contact_guard_alert")})
        row["families"]={}
        for kind,family in c["families"].items():
            path=raw
            if kind!="base":
                step,state_key=(1,"early_state") if kind=="early" else (2,"second_state")
                p=by_prefix[name,kind]
                path,ledger=segment_audit(root,p,c,counts,peaks,start_step=step)
                intervention=early_map({**c,"early_state":c[state_key]},path[0][0],path[1][0],peaks)
                charge=calls[step]+int(kind=="second" and c.get("contact_guard_alert",False))
                assert np.array_equal(path[2][0],f[step]) and path[3][0]==charge+4
                assert p["actual_prefix_calls"]==p["counts"]["total_oracle_calls"]-charge
                ledgers.append({"run_id":p["run_id"],"prefix":ledger,"early_intervention":intervention})
            row["families"][kind]=audit_family(root,c,family,kind,path,by_run[name,kind],full,counts,peaks,ledgers,
                                                f[:,0]-c["initial_loss"])
        row["policies"]={p:policy_record(row,p) for p in POLICIES}
        rows.append(row)
    if cfg["stage"]=="formal":
        create(root/"results.json",evaluator(rows))
    create(root/"paired_outcomes.json",rows)
    create(root/"already_deviated_cases.json",[
        {"context_id":r["context_id"],"seed":r["seed"],"family":k,"incumbent_energy":r["incumbent_energy"],**f}
        for r in rows for k,f in r["families"].items() if f["complete"] and f["already_deviated"]])
    create(root/"physical_ledgers.json",{
        "events":[{"context_id":c["context_id"],"family":k,**f["contact"]}
                  for c in contexts for k,f in c["families"].items() if f["complete"]],
        "segments":ledgers,"new_oracle_calls_for_export":0,
    })
    result={
        "passed":True,"generated":len(contexts),"continuations":len(runs),
        "solver_failures":[r["run_id"] for r in prefixes+runs+direct if r["failed"]],
        "incomplete_families":[f["family_id"] for f in families if not f["complete"]],
        "incomplete_continuations":[r["run_id"] for r in runs if r["committed_steps"]!=END_STEP-BOUNDARY_STEP],
        "incomplete_forecasts":sum(f.get("forecast_complete") is False for r in rows for f in r["families"].values()),
        "max_residuals":peaks,**counts,
    }
    create(root/"audit.json",result)
    print(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)
