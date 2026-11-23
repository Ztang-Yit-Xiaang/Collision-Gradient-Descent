"""Independently audit saved predictive runs without changing their trajectories."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def audit(output):
    output=Path(output);config=json.loads((output/"config.json").read_text())
    runs=json.loads((output/"runs.json").read_text());initials=json.loads((output/"initial_states.json").read_text())
    setup={(r["case"],r["seed"]):r for r in initials}
    physics=config["config"]["strategy"]["population"]["physics"];g=physics["gravity"];h=physics["h"]
    package=Path(__file__).resolve().parents[1]/"src"/"collision_optim"
    assert all(hashlib.sha256((package/k).read_bytes()).hexdigest()==v for k,v in config["source_sha256"].items())
    shared={};saved_actions={};saved_events={};terminal=[];branch_reasons={}
    maxima={k:0. for k in ["recomputed_energy","source_energy","source_momentum","gravity_work","pair_energy","pair_momentum","first_impulse","first_time","first_direction"]}
    cap_checks=virtual_candidates=completed_candidates=matched_first=forecast_checks=0
    ball_forecast_errors=[];impulse_forecast_errors=[];decision_costs=[]
    for run in runs:
        case,seed,mode=run["case"],run["seed"],run["mode"]
        stem=f"{case}__s{seed:03d}__{mode}"
        with gzip.open(output/"trajectories"/(stem+".csv.gz"),"rt") as stream:raw=list(csv.DictReader(stream))
        events=json.loads((output/"events"/(stem+".json")).read_text())
        actions=json.loads((output/"interventions"/(stem+".json")).read_text())
        decisions=json.loads((output/"decisions"/(stem+".json")).read_text())
        saved_actions[case,seed,mode]=actions;saved_events[case,seed,mode]=events
        first,last=raw[0],raw[-1];key=case,seed
        signature=(first["positions"],first["velocities"],first["energy"])
        if key in shared:assert shared[key]==signature
        shared[key]=signature
        best=float(first["value"]);previous_calls=-1;previous_time=-1.;previous_count=1
        for row in raw:
            calls=int(row["total_oracle_calls"]);time=float(row["time"])
            assert calls>=previous_calls and time>=previous_time-1e-9
            assert int(row["particle_count"])>=previous_count
            if row["record"]=="observation":best=min(best,float(row["value"]))
            assert abs(float(row["best_value"])-best)<1e-12,"Virtual or partial-step score leaked into the actual best."
            assert int(row["planning_calls"])+int(row["dynamics_calls"])==calls
            assert sum(int(row[k]) for k in ["value_calls","gradient_calls","path_calls"])==calls
            positions=np.array(json.loads(row["positions"]));velocities=np.array(json.loads(row["velocities"]))
            recomputed=float(.5*np.sum(velocities*velocities)+g*np.sum(positions[:,-1]))
            maxima["recomputed_energy"]=max(maxima["recomputed_energy"],abs(recomputed-float(row["energy"])))
            assert float(row["positive_supply"])<=config["config"]["strategy"]["energy_budget"]+1e-10
            assert abs(float(row["energy_balance_error"]))<1e-10
            assert abs(float(row["target_energy_balance_error"]))<1e-10
            assert float(row["momentum_balance_error"])<1e-10
            previous_calls=calls;previous_time=time;previous_count=int(row["particle_count"])
        assert int(last["particle_count"])==1+(int(last["shots"]) if mode=="predictive_ball" else 0)
        assert sum(d["planning_calls"] for d in decisions)==run["planning_calls"]
        supplied=sum(a.get("energy_input",0.) for a in actions)
        supplied_p=np.sum([a["momentum_input"] for a in actions if "momentum_input" in a],axis=0) if actions else np.zeros(len(json.loads(first["velocities"])[0]))
        maxima["source_energy"]=max(maxima["source_energy"],abs(supplied-float(last["cumulative_source_energy"])))
        maxima["source_momentum"]=max(maxima["source_momentum"],float(np.linalg.norm(supplied_p-json.loads(last["cumulative_source_momentum"]))))
        maxima["gravity_work"]=max(maxima["gravity_work"],abs(g*(float(first["target_z"])-float(last["target_z"]))-float(last["target_gravity_work"])))
        if run["cost_covered"]:
            cap=int(config["cost_cap"]);eligible=[r for r in raw if int(r["total_oracle_calls"])<=cap]
            assert int(eligible[-1]["total_oracle_calls"])==cap
            assert run["cost_time"]==float(eligible[-1]["time"])
            assert run["cost_relative_best_gap"]==float(eligible[-1]["relative_best_gap"])
            cap_checks+=1
        for e in events:
            if e["kind"]!="pair":continue
            a,b,c,d=[np.array(e[k]) for k in ["before_i","before_j","after_i","after_j"]]
            maxima["pair_energy"]=max(maxima["pair_energy"],abs(float(.5*(c@c+d@d-a@a-b@b))))
            maxima["pair_momentum"]=max(maxima["pair_momentum"],float(np.linalg.norm(c+d-a-b)))
        for d in decisions:
            branch_reasons[d["reason"]]=branch_reasons.get(d["reason"],0)+1
            decision_costs.append({"case":case,"mode":mode,"calls":d["planning_calls"]})
            virtual_candidates+=len(d["candidates"])
            completed=[c for c in d["candidates"] if c["status"]=="completed"]
            completed_candidates+=len(completed)
            for c in completed:assert len(c["trace"])==round(config["config"]["horizon"]/h)+1
            if d["selected"] is not None:
                selected=d["candidates"][d["selected"]]
                assert selected["status"]=="completed" and selected["gain"]>config["config"]["epsilon_gain"]
                valid=[c for c in completed if c["gain"]>config["config"]["epsilon_gain"]]
                assert selected["energy"]==min(c["energy"] for c in valid)
        for a in actions:
            if "realized_best_at_horizon" in a and "predicted_best" in a:
                error=abs(a["realized_best_at_horizon"]-a["predicted_best"])
                (ball_forecast_errors if mode=="predictive_ball" else impulse_forecast_errors).append(error)
                forecast_checks+=1
        terminal.append({"case":case,"seed":seed,"mode":mode,"current_gap":float(last["value"])-run["minimum"],
                         "best_gap":float(last["best_value"])-run["minimum"]})
    for case,seed in shared:
        balls=saved_actions[case,seed,"predictive_ball"];pulses=saved_actions[case,seed,"predictive_impulse"]
        assert bool(balls)==bool(pulses),"First acceptance should match before divergence."
        if not balls:continue
        ball,pulse=balls[0],pulses[0];matched_first+=1
        maxima["first_time"]=max(maxima["first_time"],abs(ball["planned_contact_time"]-pulse["actual_impulse_time"]))
        maxima["first_direction"]=max(maxima["first_direction"],float(np.linalg.norm(np.array(ball["direction"])-pulse["direction"])))
        assert ball["virtual_supply"]==pulse["virtual_supply"]
        e=next(e for e in saved_events[case,seed,"predictive_ball"] if e["kind"]=="pair" and {e["i"],e["j"]}=={0,ball["helper_index"]} and abs(e["time"]-ball["planned_contact_time"])<1e-8)
        suffix="i" if e["i"]==0 else "j"
        impulse=np.array(e[f"after_{suffix}"])-e[f"before_{suffix}"]
        maxima["first_impulse"]=max(maxima["first_impulse"],float(np.linalg.norm(impulse-pulse["actual_impulse"])))
    assert all(v<1e-10 for v in maxima.values())
    if ball_forecast_errors:assert max(ball_forecast_errors)<1e-10
    results={"runs":len(runs),"source_hashes_match":True,"paired_initial_states":len(shared),"exact_cost_cap_checks":cap_checks,
             "matched_first_interventions":matched_first,"max_errors":maxima,"decision_reasons":branch_reasons,
             "virtual_candidates":virtual_candidates,"completed_candidates":completed_candidates,
             "forecast_checks":forecast_checks,"max_ball_forecast_error":max(ball_forecast_errors,default=None),
             "max_impulse_vs_ball_forecast_difference":max(impulse_forecast_errors,default=None),"per_case":[]}
    for case in config["cases"]:
        for mode in config["modes"]:
            group=[r for r in runs if r["case"]==case and r["mode"]==mode];scores=[r for r in terminal if r["case"]==case and r["mode"]==mode]
            actions=[a for (c,s,m),aa in saved_actions.items() if c==case and m==mode for a in aa]
            supply=[a.get("virtual_supply") for a in actions if "virtual_supply" in a]
            results["per_case"].append({"case":case,"mode":mode,"shots":sum(r["shots"] for r in group),
                "trajectories_with_shots":sum(r["shots"]>0 for r in group),"accepted":sum(r["accepted_decisions"] for r in group),
                "rejected":sum(r["rejected_decisions"] for r in group),"mean_positive_supply":statistics.mean(r["positive_supply"] for r in group),
                "median_planning_calls":statistics.median(r["planning_calls"] for r in group),
                "median_total_calls":statistics.median(r["total_oracle_calls"] for r in group),
                "median_current_gap":statistics.median(r["current_gap"] for r in scores),
                "median_best_gap":statistics.median(r["best_gap"] for r in scores),
                "virtual_supply_counts":{str(e):supply.count(e) for e in sorted(set(supply))}})
    (output/"audit.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    with (output/"terminal_scores.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(terminal[0]));writer.writeheader();writer.writerows(terminal)
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.output),indent=2))
