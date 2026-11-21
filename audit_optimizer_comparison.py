"""Audit independent tuning, paid work and held-out optimizer comparisons."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))


def source_matches(config):
    package=ROOT/"src"/"collision_optim"
    return all(hashlib.sha256((package/k).read_bytes()).hexdigest()==v for k,v in config["source_sha256"].items())


def audit_synthetic(output):
    output=Path(output);cfg=json.loads((output/"config.json").read_text());runs=json.loads((output/"runs.json").read_text())
    tuning=json.loads((output/"tuning.json").read_text());chosen=json.loads((output/"selected.json").read_text())
    assert source_matches(cfg)
    assert not set(cfg["tuning_seeds"]) & set(cfg["test_seeds"])
    for case in cfg["cases"]:
        for family,grid in cfg["grids"].items():
            options=[]
            for setting in grid:
                group=[r for r in tuning if r["case"]==case and r["method"]==family and r["setting"]==setting]
                assert len(group)==len(cfg["tuning_seeds"])
                options.append((sum(r["failure"] for r in group),float(np.mean([np.log10(max(r["relative_best_gap"],1e-12)) for r in group])),setting))
            assert min(options,key=lambda a:(a[0],a[1]))[2]==chosen[case][family]["setting"]
    shared={};cost_checks=0;source_checks=0
    for run in runs:
        key=run["protocol"],run["case"],run["seed"]
        if key in shared:assert shared[key]==run["initial_x"]
        shared[key]=run["initial_x"]
        family="gravity" if run["method"].startswith("gravity") else run["method"].split("_")[0]
        assert run["setting"]==chosen[run["case"]][family]["setting"]
        cap=cfg["primary_budget"] if run["protocol"]=="budget" else cfg["diagnostic_budget"]
        assert run["total_oracle_calls"]<=cap
        name=f"{run['protocol']}__{run['case']}__s{run['seed']}__{run['method']}"
        with gzip.open(output/"trajectories"/(name+".csv.gz"),"rt") as stream:rows=list(csv.DictReader(stream))
        best=float(rows[0]["value"]);last_calls=-1
        for row in rows:
            if row["record"]=="observation":best=min(best,float(row["value"]))
            assert abs(float(row["best_value"])-best)<1e-12
            calls=int(row["total_oracle_calls"])
            assert calls>=last_calls and calls<=cap
            assert int(row["planning_calls"])+int(row["dynamics_calls"])==calls
            assert sum(int(row[k]) for k in ["value_calls","gradient_calls","path_calls"])==calls
            last_calls=calls
        assert run["best_f"]==float(rows[-1]["best_value"])
        if run["protocol"]=="budget" and not run["failure"]:
            assert run["total_oracle_calls"]==cap;cost_checks+=1
        actions=json.loads((output/"interventions"/(name+".json")).read_text())
        supplied=sum(a.get("energy_input",a.get("energy",0.)) for a in actions)
        assert abs(supplied-run["supply_diagnostic"])<1e-10
        source_checks+=1
    audit={"runs":len(runs),"tuning_runs":len(tuning),"source_hashes_match":True,
           "shared_initial_parameter_sets":len(shared),"independent_tuning_verified":True,
           "exact_primary_budget_checks":cost_checks,"source_checks":source_checks,
           "failures":sum(r["failure"] for r in runs),"actual_best_scores_only":True}
    (output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit


def audit_nanogpt(output,*,replay=False):
    output=Path(output);cfg=json.loads((output/"config.json").read_text());runs=json.loads((output/"runs.json").read_text())
    tuning=json.loads((output/"tuning.json").read_text());selected=json.loads((output/"selected.json").read_text())
    assert source_matches(cfg)
    vendor=ROOT/"third_party"/"nanogpt"/"model.py";data=ROOT/"data"/"tinyshakespeare.txt"
    assert hashlib.sha256(vendor.read_bytes()).hexdigest()==cfg["vendor_sha256"]
    assert hashlib.sha256(data.read_bytes()).hexdigest()==cfg["data_sha256"]
    assert cfg["tuning_seed"] not in cfg["test_seeds"]
    for kind in ["momentum","adam","adamw"]:
        group=[r for r in tuning if r["kind"]==kind]
        assert len(group)==9 and all(r["eval_split"]=="tuning" and r["seed"]==cfg["tuning_seed"] for r in group)
        valid=[r for r in group if r["final_loss"] is not None];chosen=min(valid,key=lambda r:r["final_loss"])
        assert selected[kind]=={"lr":chosen["lr"],"weight_decay":chosen["weight_decay"]}
    initial={};aliases=0;matched=[];cache={}
    for row in runs:
        assert row["eval_split"]=="test" and row["seed"] in cfg["test_seeds"]
        assert row["lr"]==selected[row["kind"]]["lr"] and row["weight_decay"]==selected[row["kind"]]["weight_decay"]
        if row["seed"] in initial:assert initial[row["seed"]]==row["initial_hash"]
        initial[row["seed"]]=row["initial_hash"]
        assert sum(row[k] for k in ["real_train_units","virtual_train_units","real_probe_units","virtual_probe_units"])==row["used_units"]<=row["budget_units"]
        assert row["real_train_units"]==3*row["actual_training_steps"]
        assert row["actual_train_tokens"]==row["actual_training_steps"]*cfg["model"]["batch"]*cfg["model"]["block"]
        name=f"{row['method']}__s{row['seed']}"
        actions=json.loads((output/"actions"/(name+".json")).read_text())
        decisions=json.loads((output/"decisions"/(name+".json")).read_text())
        assert len(actions)==row["shots"] and len(decisions)==row["decisions"]
        assert sum(d["units"] for d in decisions)==row["virtual_train_units"]+row["virtual_probe_units"]
        if replay and row["control"]!="none" and row["shots"]==0:
            from collision_optim.nanogpt_benchmark import LMConfig,TextData,train_run
            import torch
            torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
            key=(row["kind"],row["seed"],row["actual_training_steps"])
            if key not in cache:
                model_cfg=LMConfig(**cfg["model"]);dataset=TextData(data,model_cfg)
                cache[key]=train_run(dataset,model_cfg,row["kind"],row["lr"],row["weight_decay"],row["seed"],
                    equivalent_steps=row["actual_training_steps"],eval_batches=row["eval_batches"])[0]
            other=cache[key]
            assert other["final_hash"]==row["final_hash"] and other["final_loss"]==row["final_loss"]
            matched.append({"method":row["method"],"seed":row["seed"],"steps":row["actual_training_steps"],
                            "final_weights_identical":True,"replayed_test_loss":other["final_loss"]})
    for seed in cfg["test_seeds"]:
        a=next(r for r in runs if r["seed"]==seed and r["method"]=="adam")
        b=next(r for r in runs if r["seed"]==seed and r["method"]=="adamw")
        if a["weight_decay"]==b["weight_decay"]==0 and a["lr"]==b["lr"]:
            assert a["final_hash"]==b["final_hash"] and a["final_loss"]==b["final_loss"];aliases+=1
    audit={"runs":len(runs),"tuning_trials":len(tuning),"source_hashes_match":True,"data_and_vendor_hashes_match":True,
           "paired_initial_weight_sets":len(initial),"independent_tuning_and_test":True,
           "compute_accounting_verified":True,"zero_decay_identity_pairs":aliases,
           "interventions":sum(r["shots"] for r in runs),"decisions":sum(r["decisions"] for r in runs),
           "matched_update_replays":matched,"unique_replay_runs":len(cache)}
    (output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if matched:(output/"matched_updates_control.json").write_text(json.dumps(matched,indent=2)+"\n")
    return audit


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("kind",choices=["synthetic","nanogpt"])
    parser.add_argument("output",type=Path);parser.add_argument("--matched-replay",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit_synthetic(args.output) if args.kind=="synthetic" else audit_nanogpt(args.output,replay=args.matched_replay),indent=2))
