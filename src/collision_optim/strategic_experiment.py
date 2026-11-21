"""Reproducible target-only stagnation rescue experiment and cost controls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .core import State
from .objectives import Quadratic,TiltedDoubleWell
from .population import PopulationGravity,PopulationOracle,PopulationState
from .population_cost import last_observation
from .strategic import MODES,StrategicConfig,run_strategic


CASES=("bad_well","good_well","quadratic")
LABELS={"continue":"Continue","random_kick":"Random kick",
        "geometry_kick":"Geometry-matched kick","ball":"Helper ball"}
COLORS=dict(zip(MODES,["#666666","#d58519","#7557b3","#087e8b"]))


def case_definition(case):
    if case=="quadratic":
        return Quadratic(),0.,None,np.array([.1,.08])
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    objective=TiltedDoubleWell()
    roots=np.sort(np.roots([1.,0.,-1.,objective.tilt]).real)
    minimum=objective.value(np.array([roots[0],0.]))
    center=np.array([roots[-1] if case=="bad_well" else roots[0],0.])
    return objective,minimum,float(roots[1]),center


def initial_target(case,seed,config=None):
    """Known wells set interpretable starts, never a rescue direction or trigger."""
    config=config or StrategicConfig()
    objective,minimum,boundary,center=case_definition(case)
    rng=np.random.default_rng(35239+seed)
    x=center+rng.normal(scale=.01,size=2)
    velocity=rng.normal(scale=.006,size=2)
    physics=config.population.physics
    z=physics.height_scale*objective.value(x)+rng.uniform(.10,.16)
    initial=PopulationState((State(x,velocity,z,-.03),))
    if case=="bad_well":
        barrier=physics.gravity*physics.height_scale*objective.value(np.array([boundary,0.]))
        if PopulationGravity(config.population).energy(initial)>=barrier:
            raise ValueError("The trapped-well start must have less energy than its barrier.")
    return objective,minimum,boundary,initial


def _json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def _csv(path,rows,compressed=False):
    opener=gzip.open if compressed else open
    with opener(path,"wt",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def _summary(result,*,case,seed,mode,minimum,initial,config,steps,cap):
    initial_f=result.rows[0]["value"]
    denominator=max(initial_f-minimum,1e-12)
    for row in result.rows:
        row["relative_best_gap"]=max(0.,(row["best_value"]-minimum)/denominator)
    cost_row,covered=last_observation(result.rows,cap)
    last=result.rows[-1]
    failure=result.stop_reason.startswith(("contact_error:","numerical_error:"))
    cost_valid=covered and (not failure or last["total_oracle_calls"]>cap)
    shots=[s for s in result.interventions if s["kind"]==mode]
    pairs=[e for e in result.events if e["kind"]=="pair"]
    return {"case":case,"seed":seed,"mode":mode,"stop_reason":result.stop_reason,
            "time_complete":result.stop_reason=="completed" and last["time"]>=steps*config.population.physics.h-1e-9,
            "initial_f":initial_f,"minimum":minimum,"initial_energy":PopulationGravity(config.population).energy(initial),
            "best_f":result.best_value,"relative_best_gap":last["relative_best_gap"],
            "target_hit":result.first_target is not None,"first_target":result.first_target,
            "crossed":result.first_crossing is not None,"first_crossing":result.first_crossing,
            "final_time":last["time"],"supplied_energy":last["cumulative_source_energy"],
            "particle_count":len(result.final_state.particles),"shots":result.counts["shots"],
            "launch_attempts":result.counts["launch_attempts"],
            "infeasible_launches":sum(s["kind"]=="infeasible_launch" for s in result.interventions),
            "planned_contacts":sum(s["planned_contact_observed"] for s in shots),
            "pair_collisions":len(pairs),"wall_collisions":sum(e["kind"]=="wall" for e in result.events),
            "max_energy_balance_error":max(abs(r["energy_balance_error"]) for r in result.rows),
            "max_target_energy_balance_error":max(abs(r["target_energy_balance_error"]) for r in result.rows),
            "max_momentum_balance_error":max(r["momentum_balance_error"] for r in result.rows),
            "max_pair_energy_error":max((abs(e["energy_error"]) for e in pairs),default=0.),
            "max_pair_momentum_error":max((e["momentum_error"] for e in pairs),default=0.),
            "minimum_surface_gap":min([initial.particles[0].z-config.population.physics.height_scale*initial_f]+
                                      [r["minimum_surface_gap"] for r in result.rows if "minimum_surface_gap" in r]),
            "minimum_pair_gap":min((r["minimum_pair_gap"] for r in result.rows if r.get("minimum_pair_gap") is not None),default=None),
            "target_pair_energy_transfer":last["target_pair_energy_transfer"],
            "cost_cap":cap,"cost_covered":covered,"cost_valid":cost_valid,
            "cost_time":cost_row["time"],"cost_calls":cost_row["total_oracle_calls"],
            "cost_relative_best_gap":cost_row["relative_best_gap"],"cost_target_hit":cost_row["target_hit"],
            "cost_crossed":cost_row["first_crossing"] is not None,
            "cost_supplied_energy":cost_row["cumulative_source_energy"],
            "wall_seconds":result.wall_seconds,**result.counts}


def _aggregate(runs,cases):
    aggregate=[]
    paired=[]
    lookup={(r["case"],r["seed"],r["mode"]):r for r in runs}
    for r in runs:
        if r["mode"]=="continue":
            continue
        base=lookup[r["case"],r["seed"],"continue"]
        for view,valid,metric in [("time","time_complete","relative_best_gap"),("cost","cost_valid","cost_relative_best_gap")]:
            comparable=r[valid] and base[valid]
            difference=r[metric]-base[metric]
            paired.append({"case":r["case"],"seed":r["seed"],"mode":r["mode"],"view":view,
                           "comparable":comparable,"difference":difference,
                           "outcome":("better" if difference < -1e-8 else "worse" if difference > 1e-8 else "tie") if comparable else "unavailable"})
    for case in cases:
        for mode in MODES:
            group=[r for r in runs if r["case"]==case and r["mode"]==mode]
            for view,valid,metric,hit in [("time","time_complete","relative_best_gap","target_hit"),("cost","cost_valid","cost_relative_best_gap","cost_target_hit")]:
                values=[r for r in group if r[valid]]
                p=[r for r in paired if r["case"]==case and r["mode"]==mode and r["view"]==view]
                aggregate.append({"case":case,"mode":mode,"view":view,"runs":len(group),"valid":len(values),
                                  "target_hits":sum(r[hit] for r in values),
                                  "crossings":sum(r["crossed" if view=="time" else "cost_crossed"] for r in values),
                                  "median_relative_best_gap":float(np.median([r[metric] for r in values])) if values else None,
                                  "better":sum(r["outcome"]=="better" for r in p),
                                  "worse":sum(r["outcome"]=="worse" for r in p),
                                  "ties":sum(r["outcome"]=="tie" for r in p),
                                  "unavailable":sum(r["outcome"]=="unavailable" for r in p),
                                  "median_supplied_energy":float(np.median([r["supplied_energy" if view=="time" else "cost_supplied_energy"] for r in group])),
                                  "median_oracle_calls":float(np.median([r["total_oracle_calls" if view=="time" else "cost_calls"] for r in group])),
                                  "median_time":float(np.median([r["final_time" if view=="time" else "cost_time"] for r in group]))})
    return aggregate,paired


def _step_values(rows,grid,xkey,ykey):
    x=np.asarray([r[xkey] for r in rows]);y=np.asarray([r[ykey] for r in rows])
    indices=np.searchsorted(x,grid,side="right")-1
    values=y[np.maximum(indices,0)].astype(float)
    values[(indices<0)|(grid>x[-1]+1e-9)]=np.nan
    return values


def _plots(output,curves,cases,config,steps,cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    duration=steps*config.population.physics.h
    fig,axes=plt.subplots(2,len(cases),figsize=(5.3*len(cases),7.6),squeeze=False)
    for column,case in enumerate(cases):
        for row,(xkey,limit,label) in enumerate([("time",duration,"Physical time"),("total_oracle_calls",cap,"Total oracle calls")]):
            ax=axes[row,column];grid=np.linspace(0,limit,251)
            for mode in MODES:
                paths=[path for (c,s,m),path in curves.items() if c==case and m==mode]
                values=np.array([_step_values(path,grid,xkey,"relative_best_gap") for path in paths])
                with np.errstate(invalid="ignore"):
                    med=np.nanmedian(values,axis=0);lo,hi=np.nanquantile(values,[.25,.75],axis=0)
                ax.step(grid,np.maximum(med,1e-5),where="post",color=COLORS[mode],label=LABELS[mode])
                ax.fill_between(grid,np.maximum(lo,1e-5),np.maximum(hi,1e-5),step="post",color=COLORS[mode],alpha=.12)
            ax.set(title=case.replace("_"," "),xlabel=label,yscale="log",ylabel="Target best gap / initial gap",ylim=(1e-5,1.4))
            ax.axhline(.1,color="#aaaaaa",ls=":",lw=1);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Stagnation rescue: target only | median and IQR across seeds\nSame energy cap; actual spending and usable time differ",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/"comparison.png",dpi=180);plt.close(fig)

    case="bad_well" if "bad_well" in cases else cases[0]
    fig,axes=plt.subplots(2,2,figsize=(12,8.6))
    objective,_,boundary,_=case_definition(case)
    sample=curves[case,0,"ball"]
    allx=np.array([r["target_x"] for mode in MODES for r in curves[case,0,mode]])
    lo=allx.min(axis=0)-.15;hi=allx.max(axis=0)+.15
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],180),np.linspace(lo[1],hi[1],150))
    zz=.25*(xx*xx-1)**2+objective.tilt*xx+.5*yy*yy if hasattr(objective,"tilt") else .5*(xx*xx+yy*yy)
    axes[0,0].contour(xx,yy,zz,levels=15,colors="#bbbbbb",linewidths=.6)
    for mode in MODES:
        path=curves[case,0,mode];x=np.array([r["target_x"] for r in path])
        axes[0,0].plot(x[:,0],x[:,1],color=COLORS[mode],label=LABELS[mode],lw=1.4)
        axes[0,1].step([r["time"] for r in path],[max(r["relative_best_gap"],1e-5) for r in path],where="post",color=COLORS[mode])
    launches=[r for r in sample if r["record"]=="intervention"]
    for r in launches:
        axes[0,0].scatter(*r["target_x"],marker="*",s=120,color=COLORS["ball"],edgecolor="black",zorder=5)
        for ax in [axes[0,1],axes[1,0],axes[1,1]]:
            ax.axvline(r["time"],color="#888888",ls=":",lw=.8)
    if boundary is not None:
        axes[0,0].axvline(boundary,color="black",ls="--",lw=.8)
    axes[0,0].set(title=f"{case}, seed 0: target paths; stars = launches",xlabel="x[0]",ylabel="x[1]")
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title="Target best loss",xlabel="Physical time",ylabel="Relative best gap",yscale="log")
    t=[r["time"] for r in sample];h0=sample[0]["energy"]
    axes[1,0].plot(t,[r["energy"]-h0 for r in sample],label="System H - H0",lw=2)
    axes[1,0].plot(t,[r["cumulative_source_energy"]+r["cumulative_damping"] for r in sample],ls="--",label="Supply + damping")
    axes[1,0].set(title="Helper-ball system energy ledger",xlabel="Physical time",ylabel="Energy change");axes[1,0].legend()
    for key,label in [("target_pair_energy_transfer","Pair transfer"),("target_damping_energy","Damping"),("target_gravity_work","Gravity work")]:
        axes[1,1].plot(t,[r[key] for r in sample],label=label)
    axes[1,1].set(title="Target transfer and work (separate terms)",xlabel="Physical time",ylabel="Cumulative energy / work");axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output/"mechanism_seed0.png",dpi=180);plt.close(fig)


def run_experiments(output,*,seeds=20,steps=500,cases=CASES,oracle_budget=12000,config=None,plots=True):
    if any(not isinstance(n,int) or n<1 for n in [seeds,steps,oracle_budget]) or not cases or any(c not in CASES for c in cases):
        raise ValueError("Positive experiment counts and known cases are required.")
    config=config or StrategicConfig();output=Path(output)
    if (output/"config.json").exists():
        raise ValueError("Use a new output directory to preserve existing experiment provenance.")
    output.mkdir(parents=True,exist_ok=True)
    for name in ["trajectories","events","interventions","detectors"]:(output/name).mkdir(exist_ok=True)
    cap=3*steps  # Initial target count is one; fixed before seeing results.
    package=Path(__file__).resolve().parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")}
    _json(output/"config.json",{"config":asdict(config),"seeds":seeds,"steps":steps,"cases":list(cases),
          "modes":list(MODES),"oracle_budget":oracle_budget,"cost_cap":cap,
          "score":"Original target only; best values at saved observation points", "energy_zero":"z=0",
          "oracle_cost":"Each value, gradient, analytic path query counts one; pair predictions separate",
          "python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"source_sha256":hashes})
    runs=[];curves={};initials=[]
    for case in cases:
        for seed in range(seeds):
            objective,minimum,boundary,initial=initial_target(case,seed,config)
            p=initial.particles[0];initial_f=objective.value(p.x)
            target=minimum+.1*(initial_f-minimum)
            initials.append({"case":case,"seed":seed,"x":p.x.tolist(),"v":p.v.tolist(),"z":p.z,"vz":p.vz,
                             "minimum":minimum,"basin_boundary":boundary,"target_value":target})
            for mode in MODES:
                result=run_strategic(PopulationOracle(objective,total_budget=oracle_budget),initial,steps,
                                     mode=mode,config=config,seed=seed,basin_boundary=boundary,target_value=target)
                name=f"{case}__s{seed:02d}__{mode}"
                runs.append(_summary(result,case=case,seed=seed,mode=mode,minimum=minimum,initial=initial,config=config,steps=steps,cap=cap))
                _csv(output/"trajectories"/(name+".csv.gz"),result.rows,True)
                _json(output/"events"/(name+".json"),result.events)
                _json(output/"interventions"/(name+".json"),result.interventions)
                _json(output/"detectors"/(name+".json"),result.detector)
                curves[case,seed,mode]=result.rows
                _json(output/"checkpoint.json",runs)
                print(f"{len(runs):3d}/{len(cases)*seeds*len(MODES)} {name}: {result.stop_reason}; shots={result.counts['shots']}",flush=True)
    aggregate,paired=_aggregate(runs,cases)
    for name,data in [("initial_states",initials),("runs",runs),("aggregate",aggregate),("paired",paired)]:
        _json(output/(name+".json"),data)
        if name!="initial_states":_csv(output/(name+".csv"),data)
    report={"runs":len(runs),"completed":sum(r["time_complete"] for r in runs),
            "failures":sum(r["stop_reason"].startswith(("contact_error:","numerical_error:")) for r in runs),
            "budget_stops":sum(r["stop_reason"].startswith("budget:") for r in runs),
            "cost_covered":sum(r["cost_covered"] for r in runs),"cost_cap":cap,
            "shots":sum(r["shots"] for r in runs),"ball_shots":sum(r["shots"] for r in runs if r["mode"]=="ball"),
            "planned_contacts":sum(r["planned_contacts"] for r in runs),
            "pair_collisions":sum(r["pair_collisions"] for r in runs),
            "max_energy_balance_error":max(r["max_energy_balance_error"] for r in runs),
            "max_target_energy_balance_error":max(r["max_target_energy_balance_error"] for r in runs),
            "max_momentum_balance_error":max(r["max_momentum_balance_error"] for r in runs),
            "max_pair_energy_error":max(r["max_pair_energy_error"] for r in runs),
            "max_pair_momentum_error":max(r["max_pair_momentum_error"] for r in runs)}
    _json(output/"summary.json",report)
    if plots:_plots(output,curves,cases,config,steps,cap)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/strategic"))
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--oracle-budget",type=int,default=12000)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    args=parser.parse_args(argv)
    print(json.dumps(run_experiments(args.output,seeds=args.seeds,steps=args.steps,cases=args.cases,
                                    oracle_budget=args.oracle_budget),indent=2))
    return 0
