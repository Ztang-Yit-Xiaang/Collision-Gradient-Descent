"""Compare E36's conditional second-step rule and diagnose the frozen E35 misses."""

import argparse
from pathlib import Path
from statistics import median

import numpy as np

from audit_high_kinetic_intervention_timing import summarize
from audit_two_stage_protection import WINDOWS, audit, lookup
from collision_optim.prebounce_warning import directional_quadratic, flight_factors, forecast_rise
from finalize_initial_budget import create, sha
from run_second_step_protection import PARENT, SEEDS
from run_two_stage_protection import ENERGIES, H, LIMIT, read


def select(row, added):
    if not added or not row["second_warning"]:
        return {**row["policies"]["two_stage"], "second_transfer":False}
    f=row["families"]["second"]
    bounce=f["complete"] and f["warning"]
    return {
        "family":"second","mode":"transfer" if bounce else "keep","early_transfer":False,
        "second_transfer":True,"bounce_transfer":bounce,"forecast_queries":f["forecast_queries"],
    }


def metrics(rows, added, horizon):
    chosen=[(r,select(r,added)) for r in rows]
    values=[lookup(r,s) for r,s in chosen]
    old=[lookup(r,select(r,False)) for r in rows]
    keep=[r["families"]["base"]["branches"]["keep"] for r in rows]
    stage_keeps=[r["families"][s["family"]]["branches"]["keep"] for r,s in chosen]
    early=sum(s["early_transfer"] for _,s in chosen)
    second=sum(s["second_transfer"] for _,s in chosen)
    bounce=sum(s["bounce_transfer"] for _,s in chosen)
    no_risk_early=sum(s["early_transfer"] and not b[horizon]["deviated"] for (_,s),b in zip(chosen,keep))
    no_risk_second=sum(s["second_transfer"] and not b[horizon]["deviated"] for (_,s),b in zip(chosen,keep))
    no_risk_bounce=sum(s["bounce_transfer"] and not b[horizon]["deviated"] for (_,s),b in zip(chosen,stage_keeps))
    return {
        **summarize(values,horizon),"step1_transfers":early,"step2_transfers":second,"bounce_transfers":bounce,
        "total_transfers":early+second+bounce,
        "unnecessary_step1":no_risk_early,"unnecessary_step2":no_risk_second,"unnecessary_bounce":no_risk_bounce,
        "unnecessary_transfers":no_risk_early+no_risk_second+no_risk_bounce,
        "step2_transfers_where_old_policy_safe":sum(s["second_transfer"] and not b[horizon]["deviated"]
                                                  for (_,s),b in zip(chosen,old)),
        "prevented_vs_old":sum(b[horizon]["deviated"] and not y[horizon]["deviated"] for b,y in zip(old,values)),
        "induced_vs_old":sum(not b[horizon]["deviated"] and y[horizon]["deviated"] for b,y in zip(old,values)),
        "forecast_queries":sum(s["forecast_queries"] for _,s in chosen),
    }


def recovery_cases(rows, added):
    late=[]
    for r in rows:
        choice=select(r,added)
        f=r["families"][choice["family"]]
        if f["complete"] and f["already_deviated"]:
            late.append({"context_id":r["context_id"],"seed":r["seed"],"incumbent_energy":r["incumbent_energy"],
                         "family":choice["family"],**f})
    return late


def recovery_summary(late):
    return {
        "states":len(late),**{mode:{
            name:{**summarize([r["branches"][mode] for r in late],name),
                  "recrossed":sum(r["branches"][mode]["first_recross"] is not None
                                  and r["branches"][mode]["first_recross"]<=time for r in late)}
            for name,time in WINDOWS.items()} for mode in ("keep","transfer")},
    }


def report(rows):
    valid=[r for r in rows if all(f["complete"] and all(b["complete"] for b in f["branches"].values())
                                 for f in r["families"].values())]
    summary={name:{
        key:metrics(valid,added,name) for key,added in (("two_stage",False),("two_stage_plus_step2",True))
    } for name in WINDOWS}
    old,new=(summary["primary"][p] for p in ("two_stage","two_stage_plus_step2"))
    checks={
        "all_policy_contexts_complete":len(rows)==len(valid),
        "old_policy_at_least_five_deviations":old["deviated"]>=5,
        "prevents_at_least_five_additional_deviations":new["prevented_vs_old"]>=5,
        "no_added_deviations":new["induced_vs_old"]==0,
        "unnecessary_second_step_at_most_10_percent":bool(new["step2_transfers"] and
            new["unnecessary_step2"]/new["step2_transfers"]<=.10),
    }
    return {
        "contexts":len(rows),"complete_contexts":len(valid),"seed_groups":len({r["seed"] for r in valid}),
        "policy_metrics":summary,"second_step_candidates":sum(not r["early_warning"] for r in valid),
        "already_deviated_at_step2":sum(r["already_deviated_at_step2"] for r in valid),
        "new_policy_second_family_bounce_warnings":sum(r["families"]["second"]["warning"]
                                                       for r in valid if "second" in r["families"]),
        "by_initial_kinetic":{str(e):{
            key:metrics([r for r in valid if r["incumbent_energy"]==e],added,"primary")
            for key,added in (("two_stage",False),("two_stage_plus_step2",True))
        } for e in ENERGIES},
        "recovery":{key:recovery_summary(recovery_cases(valid,added))
                    for key,added in (("two_stage",False),("two_stage_plus_step2",True))},
        "checks":checks,"passed_screen":all(checks.values()),
    }


def diagnose_e35():
    contexts=PARENT/"contexts.json"
    recovery=PARENT/"two_stage_recovery_cases.json"
    c_by_id={c["context_id"]:c for c in read(contexts)}
    rows=[]
    factors=flight_factors()
    for source in read(recovery)["rows"]:
        c=c_by_id[source["context_id"]]
        f=c["families"]["base"]
        rises=f["history_rises"]
        speed=float(np.linalg.norm(c["start"]["velocities"][0][:2]))
        _,curvature=directional_quadratic(speed*factors[1:3],rises[1:3])
        second=forecast_rise(rises[1:3],speed,factors)
        assert not c["early_warning"] and c["early_score"]<=LIMIT
        rows.append({
            "context_id":source["context_id"],"seed":c["seed"],"incumbent_energy":c["incumbent_energy"],
            "first_step_rise":rises[1],"second_step_rise":rises[2],"first_step_forecast":c["early_score"],
            "second_step_forecast":second,"fitted_directional_curvature":float(curvature),
            "second_step_timely_warning":max(rises[:3])<=LIMIT and second>LIMIT,
            "already_deviated_at_saved_precontact_time":max(rises)>LIMIT,
            "first_contact_rise":f["contact_rise"],"first_contact_time":f["contact_time"],
        })
    assert len(rows)==17
    curvature_prior=read(PARENT/"config.json")["early_prior_curvature"]
    return {
        "rows":rows,"new_oracle_calls":0,"source_sha256":{str(p):sha(p) for p in (contexts,recovery)},
        "first_step_prior_curvature":curvature_prior,
        "actual_directional_curvature_median":median(r["fitted_directional_curvature"] for r in rows),
        "above_prior":sum(r["fitted_directional_curvature"]>curvature_prior for r in rows),
        "first_step_loss_fell":sum(r["first_step_rise"]<0 for r in rows),
        "would_warn_at_step2":sum(r["second_step_timely_warning"] for r in rows),
        "first_cross_at_contact":sum(not r["already_deviated_at_saved_precontact_time"] for r in rows),
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    root=parser.parse_args().output
    audit(root,seed_sets=SEEDS,evaluator=report,second_stage=True)
    if read(root/"config.json")["stage"]=="formal":
        create(root/"prior_seventeen_diagnosis.json",diagnose_e35())
        cases=read(root/"paired_outcomes.json")
        create(root/"policy_recovery_cases.json",{
            "two_stage":recovery_cases(cases,False),"two_stage_plus_step2":recovery_cases(cases,True),
            "source_sha256":sha(root/"paired_outcomes.json"),"new_oracle_calls":0,
        })
