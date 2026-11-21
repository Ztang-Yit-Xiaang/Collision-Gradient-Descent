"""Controlled collision mechanisms and paired four-particle optimization trials."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import sys
import numpy as np

from .core import State
from .gravity import GravityConfig
from .highdim import _csv, _json
from .highdim_objectives import CoupledRosenbrock
from .objectives import Affine, Budget, Quadratic, TiltedDoubleWell
from .population import (PopulationConfig, PopulationGravity, PopulationOracle,
                         PopulationState, pair_contact_time, run_population)


CASES = ("quadratic", "double_well", "rosenbrock")


def _save_csv(path, rows):
    temporary = path.with_name(path.stem+".tmp"+path.suffix)
    _csv(temporary, rows)
    temporary.replace(path)


def make_objective(case):
    if case == "quadratic":
        return Quadratic(np.diag([.25,1.])), 0., None
    if case == "double_well":
        objective = TiltedDoubleWell()
        roots = sorted(np.roots([1.,0.,-1.,objective.tilt]).real)
        return objective, objective.value(np.array([roots[0],0.])), float(roots[1])
    if case == "rosenbrock":
        return CoupledRosenbrock(2), 0., None
    raise ValueError("Unknown population experiment objective.")


def initial_population(case, objective, seed, *, rho_max=.24, count=4, total_kinetic=.6):
    """Shared seeded states, generated before switching collisions on/off."""
    rng=np.random.default_rng(20261105+seed)
    particles=[]
    center = np.array([.95,0.])
    if case == "rosenbrock":
        center=objective.from_coordinates(np.array([-1.2,1.]))
    for _ in range(count):
        for attempt in range(10000):
            x=center+rng.normal(0,[.15,.22],2)
            if case == "double_well" and x[0] < .65:
                continue
            z=objective.value(x)+rng.uniform(.2,.6)
            if all(np.linalg.norm(np.r_[x-p.x,z-p.z]) >= rho_max+.02 for p in particles):
                particles.append(State(x,np.zeros(2),z,0.))
                break
        else:
            raise ValueError("Could not generate nonoverlapping initial particles.")
    velocities=rng.normal(size=(count,3))
    velocities-=np.mean(velocities,axis=0)
    velocities*=np.sqrt(2*total_kinetic/np.sum(velocities*velocities))
    return PopulationState(tuple(State(p.x,w[:-1],p.z,w[-1]) for p,w in zip(particles,velocities)))


def _summarize(result, initial, minimum, *, case, seed, method, rho):
    initial_best=min(result.rows[0]["values"])
    denominator=max(1e-12,initial_best-minimum)
    for row in result.rows:
        row["relative_best_gap"]=max(0.,(row["best_value"]-minimum)/denominator)
        row["total_oracle_calls"]=row["value_calls"]+row["gradient_calls"]+row["path_calls"]
    pair=[e for e in result.events if e["kind"]=="pair"]
    fail=result.stop_reason.startswith(("contact_error:","numerical_error:"))
    return {
        "case":case,"seed":seed,"method":method,"rho":rho,
        "stop_reason":result.stop_reason,"failed":fail,
        "completed":result.stop_reason=="completed","completed_time":result.rows[-1]["time"],
        "initial_energy":result.rows[0]["energy"],"initial_best_value":initial_best,
        "initial_kinetic":initial.kinetic,"initial_momentum":initial.momentum.tolist(),
        "best_value":result.best_value,"relative_best_gap":result.rows[-1]["relative_best_gap"],
        "pair_collisions":len(pair),"wall_collisions":sum(e["kind"]=="wall" for e in result.events),
        "first_crossing_times":result.first_crossing_times,
        "crossed_particles":sum(t is not None for t in result.first_crossing_times),
        "first_target_times":result.first_target_times,
        "target_particles":sum(t is not None for t in result.first_target_times),
        "population_target_hit":any(t is not None for t in result.first_target_times),
        "first_population_target_time":min((t for t in result.first_target_times if t is not None),default=None),
        "max_pair_energy_error":max((abs(e["energy_error"]) for e in pair),default=0.),
        "max_pair_momentum_error":max((e["momentum_error"] for e in pair),default=0.),
        "max_normalized_energy_residual":max(abs(r["cumulative_residual"]) for r in result.rows)/(1+abs(result.rows[0]["energy"])),
        "max_momentum_balance_error":max(r["momentum_residual"] for r in result.rows),
        "minimum_surface_gap":min(r["minimum_surface_gap"] for r in result.rows),
        "minimum_pair_gap":min((r["minimum_pair_gap"] for r in result.rows[1:] if r["minimum_pair_gap"] is not None),default=None),
        "wall_seconds":result.wall_seconds,**result.counts,
    }


def mechanism_trials(output):
    """Deliberately constructed examples; separate from random-initial-state trials."""
    records, paths=[],{}
    for name in ("energy_focusing","barrier_transfer"):
        if name=="energy_focusing":
            objective=Affine([0.,0.],-10.)
            initial=PopulationState((State([.5,-.37],[0.,1.],3.,0.),State([-.37,0.],[1.,0.],3.,0.)))
            rho,steps,minimum,boundary,target=.5,16,-10.,None,None
        else:
            objective,minimum,boundary=make_objective("double_well")
            initial=PopulationState((State([.8,-.0225],[0.,.15],.95,0.),State([1.0425,0.],[-.55,0.],.95,0.)))
            rho,steps,target=.16,240,-.025
        for enabled in (False,True):
            method_name="collisions" if enabled else "independent"
            cfg=PopulationConfig(physics=GravityConfig(h=.05,beta=1),rho=rho,collisions=enabled)
            result=run_population(PopulationGravity(cfg),PopulationOracle(objective),initial,steps,
                                  basin_boundary=boundary,target_value=target)
            entry=_summarize(result,initial,minimum,case=name,seed=0,method=method_name,rho=rho)
            records.append(entry)
            paths[(name,method_name)]=result.rows
            _save_csv(output/"traces"/f"mechanism__{name}__{method_name}.csv.gz",result.rows)
            _json(output/"events"/f"mechanism__{name}__{method_name}.json",result.events)
    _json(output/"mechanisms.json",records)
    return records,paths


def encounter_diagnostic(output, *, seeds=100, dimensions=(2,10,100,1000), rhos=(.12,.24)):
    """Probability of any encounter in an unforced relative-flight window.

    Both particles have identical vertical gravity, so the relative motion is
    exactly linear. Once an encounter occurs, later candidate encounters would
    change; only 'at least one in the window' is interpreted as an event rate.
    """
    records=[]
    for dimension in dimensions:
        for rho in rhos:
            encounters=0
            for seed in range(seeds):
                rng=np.random.default_rng(1105239+seed)
                for _ in range(1000):
                    q=rng.normal(size=(4,dimension+1))
                    q/=np.linalg.norm(q,axis=1)[:,None]
                    q[:,-1]+=100.
                    if min(np.linalg.norm(q[i]-q[j]) for i in range(4) for j in range(i+1,4))>=max(rhos)+.02:
                        break
                w=rng.normal(size=q.shape);w-=w.mean(axis=0)
                w*=np.sqrt(1.2/np.sum(w*w))
                particles=[State(a[:-1],b[:-1],a[-1],b[-1]) for a,b in zip(q,w)]
                cfg=PopulationConfig(rho=rho)
                times=[pair_contact_time(particles[i],particles[j],2.,cfg) for i in range(4) for j in range(i+1,4)]
                encounters+=any(t is not None for t in times)
            records.append({"dimension":dimension,"rho":rho,"populations":seeds,
                            "window":2.,"populations_with_encounter":encounters,
                            "encounter_fraction":encounters/seeds,"initial_total_kinetic":.6,
                            "initial_cloud_radius":1.})
    _save_csv(output/"encounters.csv",records)
    return records


def run_experiments(output, *, seeds=20,steps=500,h=.05,rhos=(.12,.24),cases=CASES,
                    oracle_budget=10000,geometry_seeds=100):
    rhos,cases=list(rhos),list(cases)
    if any(not isinstance(n,int) or n<1 for n in (seeds,steps,oracle_budget,geometry_seeds)):
        raise ValueError("Experiment sizes and budgets must be positive integers.")
    if not rhos or any(not np.isfinite(r) or r<=0 for r in rhos):
        raise ValueError("Contact distances must be positive and finite.")
    if not cases or any(c not in CASES for c in cases):
        raise ValueError("Unknown experiment cases.")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name in ("traces","events","initial_states"):
        (output/name).mkdir(exist_ok=True)
    physics=GravityConfig(h=h,beta=.995**(h/.05))
    budget=Budget(values=oracle_budget,gradients=oracle_budget,paths=oracle_budget)
    _json(output/"config.json",{
        "seeds":seeds,"steps":steps,"rhos":rhos,"cases":cases,"physics":asdict(physics),
        "population_size":4,"initial_total_kinetic":.6,"total_oracle_budget":oracle_budget,
        "per_kind_budget":asdict(budget),"pair_prediction_budget":100000,
        "initialization_seed_offset":20261105,"geometry_seeds":geometry_seeds,
        "python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
        "notes":["One shared oracle budget per entire population; all initial states and energies are paired.",
                 "An oracle call is one value, gradient or analytic-path request; categories are separately counted and not assumed to have equal real cost.",
                 "Independent particles pass through one another; their dynamics are independent of rho, so each baseline is executed once.",
                 "Best loss and target times are scored only on the same observation grid; barrier crossings are located analytically.",
                 "Sustained and shared contacts are explicitly unsupported in this first population scheduler.",
                 "No optimizer tuning, reheating, loss-triggered collisions or adaptive contact distances."]})
    mechanisms,mechanism_paths=mechanism_trials(output)
    summaries,paired,curves=[],[],{}
    initial_energy_error=0.
    for case in cases:
        objective,minimum,boundary=make_objective(case)
        for seed in range(seeds):
            initial=initial_population(case,objective,seed,rho_max=max(rhos))
            initial_f=min(objective.value(p.x) for p in initial.particles)
            target=minimum+.1*(initial_f-minimum)
            _json(output/"initial_states"/f"{case}__s{seed:02d}.json",{
                "positions":[np.r_[p.x,p.z].tolist() for p in initial.particles],
                "velocities":[p.velocity.tolist() for p in initial.particles],
                "known_minimum":minimum,"basin_boundary":boundary,"target_value":target})
            base=None
            for rho in [None]+rhos:
                enabled=rho is not None
                method_name="collisions" if enabled else "independent"
                cfg=PopulationConfig(physics=physics,rho=rho if enabled else max(rhos),
                                     collisions=enabled,max_pair_tests=100000)
                result=run_population(PopulationGravity(cfg),PopulationOracle(objective,budget,total_budget=oracle_budget),
                                      initial,steps,basin_boundary=boundary,target_value=target)
                entry=_summarize(result,initial,minimum,case=case,seed=seed,method=method_name,rho=rho)
                summaries.append(entry)
                stem=f"{case}__s{seed:02d}__{method_name}"+(f"__rho{rho:g}" if enabled else "")
                _save_csv(output/"traces"/f"{stem}.csv.gz",result.rows)
                _json(output/"events"/f"{stem}.json",result.events)
                curves[(case,seed,rho)]=np.array([[r["time"],r["total_oracle_calls"],r["relative_best_gap"]] for r in result.rows])
                if not enabled:
                    base=entry
                else:
                    error=abs(entry["initial_energy"]-base["initial_energy"])
                    initial_energy_error=max(initial_energy_error,error)
                    both=entry["completed"] and base["completed"]
                    difference=entry["relative_best_gap"]-base["relative_best_gap"]
                    paired.append({"case":case,"seed":seed,"rho":rho,"both_completed":both,
                                   "baseline_stop":base["stop_reason"],"collision_stop":entry["stop_reason"],
                                   "initial_energy_error":error,"relative_gap_difference":difference if both else None,
                                   "outcome":("improved" if difference < -1e-6 else "worse" if difference > 1e-6 else "tie") if both else "incomplete",
                                   "baseline_relative_gap":base["relative_best_gap"],"collision_relative_gap":entry["relative_best_gap"],
                                   "baseline_crossed":base["crossed_particles"],"collision_crossed":entry["crossed_particles"],
                                   "baseline_target":base["population_target_hit"],"collision_target":entry["population_target_hit"],
                                   "pair_collisions":entry["pair_collisions"],
                                   "baseline_oracle_calls":base["total_oracle_calls"],"collision_oracle_calls":entry["total_oracle_calls"]})
                _save_csv(output/"runs.csv",summaries)
                _save_csv(output/"paired.csv",paired)
        selected=[r for r in summaries if r["case"]==case]
        print(f"Population {case}: {len(selected)} trajectories, "
              f"{sum(r['failed'] for r in selected)} failures, "
              f"{sum(r['pair_collisions'] for r in selected)} binary impacts",flush=True)
    aggregate=[]
    for case in cases:
        base=[r for r in summaries if r["case"]==case and r["method"]=="independent"]
        for rho in rhos:
            collision=[r for r in summaries if r["case"]==case and r["rho"]==rho]
            pairs=[r for r in paired if r["case"]==case and r["rho"]==rho]
            valid=[p for p in pairs if p["both_completed"]]
            valid_seeds={p["seed"] for p in valid}
            aggregate.append({
                "case":case,"rho":rho,"paired_seeds":seeds,"both_completed":len(valid),
                "improved":sum(p["outcome"]=="improved" for p in pairs),
                "worse":sum(p["outcome"]=="worse" for p in pairs),
                "ties":sum(p["outcome"]=="tie" for p in pairs),
                "incomplete":sum(p["outcome"]=="incomplete" for p in pairs),
                "baseline_failures":sum(r["failed"] for r in base),
                "collision_failures":sum(r["failed"] for r in collision),
                "baseline_target_hits":sum(r["population_target_hit"] and r["completed"] for r in base),
                "collision_target_hits":sum(r["population_target_hit"] and r["completed"] for r in collision),
                "baseline_crossed_particles":sum(r["crossed_particles"] for r in base if r["completed"]),
                "collision_crossed_particles":sum(r["crossed_particles"] for r in collision if r["completed"]),
                "median_baseline_relative_gap":float(np.median([r["relative_best_gap"] for r in base if r["seed"] in valid_seeds])) if valid else None,
                "median_collision_relative_gap":float(np.median([r["relative_best_gap"] for r in collision if r["seed"] in valid_seeds])) if valid else None,
                "median_paired_difference":float(np.median([p["relative_gap_difference"] for p in valid])) if valid else None,
                "median_collision_oracle_calls":float(np.median([r["total_oracle_calls"] for r in collision])),
                "median_baseline_oracle_calls":float(np.median([r["total_oracle_calls"] for r in base])),
                "median_pair_tests":float(np.median([r["pair_tests"] for r in collision])),
                "total_pair_collisions":sum(r["pair_collisions"] for r in collision),
                "runs_with_pair_collision":sum(r["pair_collisions"]>0 for r in collision),
            })
    encounters=encounter_diagnostic(output,seeds=geometry_seeds,rhos=rhos)
    collision_rows=[r for r in summaries if r["method"]=="collisions"]
    checks={
        "mechanisms_completed":all(r["completed"] for r in mechanisms),
        "initial_energy_matched":initial_energy_error<1e-12,
        "no_population_failures":all(not r["failed"] for r in summaries),
        "pair_energy_conserved":max(r["max_pair_energy_error"] for r in collision_rows)<1e-9,
        "pair_momentum_conserved":max(r["max_pair_momentum_error"] for r in collision_rows)<1e-9,
        "energy_balance":max(r["max_normalized_energy_residual"] for r in summaries)<1e-7,
        "external_momentum_accounted":max(r["max_momentum_balance_error"] for r in summaries)<1e-8,
        "no_saved_terrain_penetration":min(r["minimum_surface_gap"] for r in summaries)>=-physics.gap_tolerance,
        "no_saved_pair_penetration":all(r["minimum_pair_gap"] is None or r["minimum_pair_gap"]>=-1e-9 for r in collision_rows),
        "within_shared_budget":all(r["total_oracle_calls"]<=oracle_budget for r in summaries),
    }
    checks={name:bool(value) for name,value in checks.items()}
    report={"trajectory_count":len(summaries),"paired_comparisons":len(paired),"checks":checks,
            "maximum_initial_energy_error":initial_energy_error,"aggregate":aggregate,
            "mechanisms":mechanisms,"encounters":encounters,
            "notes":["Constructed mechanisms are separate from paired random-initial-state comparisons.",
                     "Wins/losses use a fixed relative-gap difference tolerance of 1e-6 and only pairs that both complete.",
                     "Failures and budget stops remain in the denominator as incomplete comparisons, not reliable performance scores.",
                     "Baselines are shared across rho; the two rho comparisons are not independent replicates.",
                     "This is a binary-impact population prototype. Unsupported contacts are explicit failures."]}
    _save_csv(output/"aggregate.csv",aggregate);_json(output/"summary.json",report)
    _plots(output,curves,mechanism_paths,cases,rhos,seeds,paired,encounters)
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    _json(output/"source_manifest.json",manifest)
    return report


def _plots(output,curves,mechanism_paths,cases,rhos,seeds,paired,encounters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(cases),figsize=(4.8*len(cases),3.8),squeeze=False)
    for j,case in enumerate(cases):
        ax=axes[0,j]
        for rho,color in zip([None]+rhos,["#555555","#14816a","#d47726","#8057ac"]):
            valid_seeds={p["seed"] for p in paired if p["case"]==case and p["both_completed"] and (rho is None or p["rho"]==rho)}
            paths=[curves[(case,s,rho)] for s in sorted(valid_seeds)]
            if not paths:
                continue
            grid=np.linspace(0,min(p[-1,1] for p in paths),150)
            values=np.array([np.interp(grid,p[:,1],p[:,2]) for p in paths])
            med=np.maximum(np.median(values,axis=0),1e-12)
            lo,hi=np.maximum(np.quantile(values,[.25,.75],axis=0),1e-12)
            label="Independent" if rho is None else f"Collisions, rho={rho:g}"
            ax.semilogy(grid,med,label=label,color=color)
            ax.fill_between(grid,lo,hi,color=color,alpha=.12)
        ax.set_title(case.replace("_"," "));ax.set_xlabel("Total oracle calls (unweighted)")
        ax.set_ylabel("Relative best gap");ax.grid(alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.suptitle(f"Four particles: median/IQR of completed paired seeds; {seeds} initial states")
    fig.subplots_adjust(left=.07,right=.99,bottom=.23,top=.80,wspace=.3)
    fig.savefig(output/"population_comparison.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    focusing=mechanism_paths[("energy_focusing","collisions")]
    collision=next(e for e in json.loads((output/"events"/"mechanism__energy_focusing__collisions.json").read_text()) if e["kind"]=="pair")
    for i in range(2):
        suffix="i" if i==0 else "j"
        before=.5*sum(a*a for a in collision[f"before_{suffix}"][:-1])
        after=.5*sum(a*a for a in collision[f"after_{suffix}"][:-1])
        axes[0].plot([0.,collision["time"],collision["time"],focusing[-1]["time"]],
                     [before,before,after,after],label=f"Particle {i}")
    axes[0].axvline(collision["time"],color="#777777",linestyle=":",label="Contact")
    axes[0].set_ylabel("Horizontal kinetic energy");axes[0].set_xlabel("Time");axes[0].set_title("Constructed oblique impact")
    for method,style in [("independent","--"),("collisions","-")]:
        rows=mechanism_paths[("barrier_transfer",method)]
        for i,color in [(0,"#14816a"),(1,"#8057ac")]:
            axes[1].plot([r["time"] for r in rows],[r["positions"][i][0] for r in rows],
                         linestyle=style,color=color,label=f"{method}, particle {i}")
    boundary=make_objective("double_well")[2]
    axes[1].axhline(boundary,color="#777777",linestyle=":",label="Basin boundary")
    axes[1].set_ylabel("Barrier coordinate x[0]");axes[1].set_xlabel("Time");axes[1].set_title("Constructed barrier-transfer example")
    for ax in axes:
        ax.legend(fontsize=7);ax.grid(alpha=.15);ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(output/"mechanisms.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4))
    for rho in rhos:
        selected=[r for r in encounters if r["rho"]==rho]
        ax.semilogx([r["dimension"] for r in selected],[r["encounter_fraction"] for r in selected],marker="o",label=f"rho={rho:g}")
    ax.set_xlabel("Parameter dimension");ax.set_ylabel("Populations with an encounter")
    ax.set_ylim(-.02,1.02);ax.set_title("Fixed cloud radius and total kinetic energy; free-flight window")
    ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(output/"encounter_frequency.png",dpi=170);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/population"))
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--h",type=float,default=.05)
    parser.add_argument("--rhos",type=float,nargs="+",default=[.12,.24])
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    parser.add_argument("--oracle-budget",type=int,default=10000)
    parser.add_argument("--geometry-seeds",type=int,default=100)
    args=parser.parse_args(argv)
    report=run_experiments(args.output,seeds=args.seeds,steps=args.steps,h=args.h,rhos=args.rhos,cases=args.cases,
                           oracle_budget=args.oracle_budget,geometry_seeds=args.geometry_seeds)
    print({"output":str(args.output.resolve()),"trajectories":report["trajectory_count"],"paired_comparisons":report["paired_comparisons"],"checks":report["checks"]})
    return 0 if all(report["checks"].values()) else 1
