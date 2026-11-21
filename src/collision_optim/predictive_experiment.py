"""Fixed-parameter predictive intervention diagnostics on held-out seeds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

from .core import State
from .objectives import TiltedDoubleWell
from .population import PopulationOracle,PopulationState
from .predictive import PredictiveConfig
from .predictive_runner import MODES,run_predictive
from .strategic_experiment import initial_target,_json,_csv,_summary,_step_values


CASES=("bad_well","good_well","saddle")
LABELS={"continue":"Continue","random_kick":"Random kick",
        "predictive_impulse":"Preview + matched first impulse","predictive_ball":"Preview + helper ball"}
COLORS=dict(zip(MODES,["#666666","#d58519","#7557b3","#087e8b"]))


def initial_case(case,seed,config):
    if case in {"bad_well","good_well"}:return initial_target(case,seed,config.strategy)
    if case!="saddle":raise ValueError("Unknown case.")
    objective=TiltedDoubleWell(tilt=0.)
    rng=np.random.default_rng(35239+seed)
    x=np.array([0.,rng.normal(scale=.01)])
    v=np.array([0.,rng.normal(scale=.006)])
    z=config.strategy.population.physics.height_scale*objective.value(x)+rng.uniform(.10,.16)
    return objective,0.,None,PopulationState((State(x,v,z,-.03),))


def aggregate_runs(runs,cases):
    aggregates=[];paired=[]
    lookup={(r["case"],r["seed"],r["mode"]):r for r in runs}
    for r in runs:
        if r["mode"]=="continue":continue
        base=lookup[r["case"],r["seed"],"continue"]
        for view,valid,metric in [("time","time_complete","relative_best_gap"),("cost","cost_valid","cost_relative_best_gap")]:
            comparable=r[valid] and base[valid];difference=r[metric]-base[metric]
            paired.append({"case":r["case"],"seed":r["seed"],"mode":r["mode"],"view":view,
                           "comparable":comparable,"difference":difference,
                           "outcome":("better" if difference < -1e-8 else "worse" if difference > 1e-8 else "tie") if comparable else "unavailable"})
    for case in cases:
        for mode in MODES:
            group=[r for r in runs if r["case"]==case and r["mode"]==mode]
            for view,valid,metric,hit in [("time","time_complete","relative_best_gap","target_hit"),("cost","cost_valid","cost_relative_best_gap","cost_target_hit")]:
                values=[r for r in group if r[valid]]
                p=[r for r in paired if r["case"]==case and r["mode"]==mode and r["view"]==view]
                aggregates.append({"case":case,"mode":mode,"view":view,"runs":len(group),"valid":len(values),
                    "target_hits":sum(r[hit] for r in values),"median_relative_best_gap":float(np.median([r[metric] for r in values])) if values else None,
                    "better":sum(r["outcome"]=="better" for r in p),"worse":sum(r["outcome"]=="worse" for r in p),
                    "ties":sum(r["outcome"]=="tie" for r in p),"unavailable":sum(r["outcome"]=="unavailable" for r in p),
                    "median_calls":float(np.median([r["total_oracle_calls" if view=="time" else "cost_calls"] for r in group])),
                    "median_time":float(np.median([r["final_time" if view=="time" else "cost_time"] for r in group])),
                    "mean_supply":float(np.mean([r["positive_supply" if view=="time" else "cost_positive_supply"] for r in group])),
                    "trajectories_with_shots":sum((r["shots"] if view=="time" else r["cost_shots"])>0 for r in group),
                    "median_planning_calls":float(np.median([r["planning_calls"] for r in group]))})
    return aggregates,paired


def plots(output,curves,cases,config,steps,cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,len(cases),figsize=(5.4*len(cases),7.8),squeeze=False)
    for column,case in enumerate(cases):
        for row,(xkey,limit,label) in enumerate([("time",steps*config.strategy.population.physics.h,"Physical time"),("total_oracle_calls",cap,"Total paid oracle calls (including preview)")]):
            ax=axes[row,column];grid=np.linspace(0 if row==0 else 1,limit,251)
            for mode in MODES:
                paths=[path for (c,s,m),path in curves.items() if c==case and m==mode]
                values=np.array([_step_values(path,grid,xkey,"relative_best_gap") for path in paths])
                valid=np.any(np.isfinite(values),axis=0)
                med=np.full(grid.size,np.nan);lo=med.copy();hi=med.copy()
                if np.any(valid):
                    med[valid]=np.nanmedian(values[:,valid],axis=0)
                    lo[valid],hi[valid]=np.nanquantile(values[:,valid],[.25,.75],axis=0)
                ax.step(grid,np.maximum(med,1e-5),where="post",color=COLORS[mode],label=LABELS[mode])
                ax.fill_between(grid,np.maximum(lo,1e-5),np.maximum(hi,1e-5),step="post",color=COLORS[mode],alpha=.12)
            ax.axhline(.1,color="#aaaaaa",ls=":",lw=1)
            ax.set(title=case.replace("_"," "),xlabel=label,ylabel="Actual target best gap / initial gap",yscale="log",ylim=(1e-5,1.4));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Paid finite-horizon intervention | actual target only\nHeld-out seeds; median and IQR; virtual successes are excluded",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/"comparison.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    for i,case in enumerate(cases):
        group=[path for (c,s,m),path in curves.items() if c==case and m=="predictive_ball"]
        total=np.array([r[-1]["total_oracle_calls"] for r in group]);virtual=np.array([r[-1]["planning_calls"] for r in group])
        axes[0].bar(i,float(np.median(virtual)),color="#7557b3",label="Preview" if i==0 else None)
        axes[0].bar(i,float(np.median(total-virtual)),bottom=float(np.median(virtual)),color="#087e8b",label="Physical execution" if i==0 else None)
        seed=min(s for c,s,m in curves if c==case)
        for mode in ["continue","predictive_ball"]:
            path=curves[case,seed,mode]
            axes[1].step([r["total_oracle_calls"] for r in path],[r["time"] for r in path],where="post",label=f"{case}: {LABELS[mode]}",lw=1.4)
        axes[2].scatter([r[-1]["positive_supply"] for r in group],[r[-1]["relative_best_gap"] for r in group],label=case,alpha=.7)
    axes[0].set(xticks=range(len(cases)),xticklabels=[c.replace("_"," ") for c in cases],ylabel="Median oracle calls",title="Helper-ball query costs");axes[0].legend()
    axes[1].axvline(cap,color="black",ls=":",label=f"Common cap {cap}")
    axes[1].set(xlabel="Paid oracle calls",ylabel="Actual physical time",title="Time spent planning (first held-out seed)");axes[1].legend(fontsize=7)
    axes[2].set(xlabel="Actual positive supply",ylabel="Actual relative best gap",title="Helper-ball outcome and supply",yscale="log");axes[2].legend()
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output/"costs_and_supply.png",dpi=180);plt.close(fig)


def run_experiments(output,*,seeds=20,seed_start=100,steps=500,cases=CASES,oracle_budget=24000,config=None,make_plots=True):
    if any(not isinstance(n,int) or n<1 for n in [seeds,steps,oracle_budget]) or not isinstance(seed_start,int) or seed_start<0 or not cases or any(c not in CASES for c in cases):
        raise ValueError("Require positive counts, nonnegative seeds and known cases.")
    config=config or PredictiveConfig();output=Path(output)
    if (output/"config.json").exists():raise ValueError("Use a new output directory; saved experiments cannot be overwritten.")
    output.mkdir(parents=True,exist_ok=True)
    for name in ["trajectories","events","interventions","detectors","decisions"]:(output/name).mkdir(exist_ok=True)
    cap=3*steps;package=Path(__file__).resolve().parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")}
    _json(output/"config.json",{"config":asdict(config),"seeds":seeds,"seed_start":seed_start,"steps":steps,"cases":list(cases),
          "modes":list(MODES),"oracle_budget":oracle_budget,"cost_cap":cap,"score":"Actual original target at common observation grid only",
          "preview_cost":"All virtual queries are charged to the same parent oracle, separately attributed",
          "energy_reference":"z=0; matched impulse reports its own actual work",
          "python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"source_sha256":hashes})
    runs=[];curves={};initials=[]
    for case in cases:
        for seed in range(seed_start,seed_start+seeds):
            objective,minimum,boundary,initial=initial_case(case,seed,config)
            p=initial.particles[0];initial_f=objective.value(p.x);target=minimum+.1*(initial_f-minimum)
            initials.append({"case":case,"seed":seed,"x":p.x.tolist(),"v":p.v.tolist(),"z":p.z,"vz":p.vz,
                             "minimum":minimum,"basin_boundary":boundary,"target_value":target})
            for mode in MODES:
                result=run_predictive(PopulationOracle(objective,total_budget=oracle_budget),initial,steps,
                    mode=mode,config=config,seed=seed,basin_boundary=boundary,target_value=target,cost_checkpoints=(cap,))
                name=f"{case}__s{seed:03d}__{mode}"
                row=_summary(result,case=case,seed=seed,mode=mode,minimum=minimum,initial=initial,config=config.strategy,steps=steps,cap=cap)
                eligible=[r for r in result.rows if r["total_oracle_calls"]<=cap];cost=eligible[-1]
                row.update({"positive_supply":result.rows[-1]["positive_supply"],"cost_positive_supply":cost["positive_supply"],
                            "cost_shots":cost["shots"],"cost_planning_calls":cost["planning_calls"],
                            "final_current_gap":result.rows[-1]["value"]-minimum,
                            "forecast_checks":sum("realized_best_at_horizon" in s for s in result.interventions),
                            "max_forecast_best_error":max((abs(s["realized_best_at_horizon"]-s["predicted_best"]) for s in result.interventions if "realized_best_at_horizon" in s and "predicted_best" in s),default=None)})
                runs.append(row);curves[case,seed,mode]=result.rows
                _csv(output/"trajectories"/(name+".csv.gz"),result.rows,True)
                for directory,data in [("events",result.events),("interventions",result.interventions),("detectors",result.detector),("decisions",result.decisions)]:_json(output/directory/(name+".json"),data)
                _json(output/"checkpoint.json",runs)
            print(f"{len(runs)}/{len(cases)*seeds*len(MODES)} {case} seed {seed}: "+", ".join(f"{r['mode']} {r['stop_reason']}, shots={r['shots']}" for r in runs[-4:]),flush=True)
    aggregate,paired=aggregate_runs(runs,cases)
    for name,data in [("initial_states",initials),("runs",runs),("aggregate",aggregate),("paired",paired)]:
        _json(output/(name+".json"),data)
        if name!="initial_states":_csv(output/(name+".csv"),data)
    report={"runs":len(runs),"completed":sum(r["time_complete"] for r in runs),
            "failures":sum(r["stop_reason"].startswith(("contact_error:","numerical_error:")) for r in runs),
            "budget_stops":sum(r["stop_reason"].startswith("budget:") for r in runs),
            "cost_covered":sum(r["cost_covered"] for r in runs),"cost_cap":cap,
            "ball_shots":sum(r["shots"] for r in runs if r["mode"]=="predictive_ball"),
            "planned_contacts":sum(r["planned_contacts"] for r in runs),
            "max_energy_balance_error":max(r["max_energy_balance_error"] for r in runs),
            "max_target_energy_balance_error":max(r["max_target_energy_balance_error"] for r in runs),
            "max_momentum_balance_error":max(r["max_momentum_balance_error"] for r in runs),
            "max_pair_energy_error":max(r["max_pair_energy_error"] for r in runs),
            "max_pair_momentum_error":max(r["max_pair_momentum_error"] for r in runs)}
    _json(output/"summary.json",report)
    if make_plots:plots(output,curves,cases,config,steps,cap)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/predictive"))
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--seed-start",type=int,default=100)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--oracle-budget",type=int,default=24000)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    args=parser.parse_args(argv)
    report=run_experiments(args.output,seeds=args.seeds,seed_start=args.seed_start,steps=args.steps,cases=args.cases,oracle_budget=args.oracle_budget)
    print(json.dumps(report,indent=2));return 0
