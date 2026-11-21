"""Compare saved trajectories at the same committed oracle-call cap.

No interpolation of best loss across unperformed evaluations, and no extra
dynamics are executed. All oracle categories have unit bookkeeping weight;
pair prediction overhead and real time remain separate costs.
"""

import csv
import gzip
import json
from pathlib import Path
import numpy as np

from .highdim import _csv, _json


def last_observation(rows, cap):
    """Return an actual saved observation at/below cap, with coverage status."""
    eligible=[r for r in rows if int(r["total_oracle_calls"])<=cap]
    if not eligible:
        raise ValueError("The call cap does not cover the initial observation.")
    return eligible[-1],int(rows[-1]["total_oracle_calls"])>=cap


def cost_matched_report(output, *, cap=None):
    output=Path(output)
    cfg=json.loads((output/"config.json").read_text())
    # Each event-free observation step needs 3*N calls: validate, path, score.
    # Thus this cap is covered by every completed trajectory, including the
    # independent baseline; it depends on settings, not observed losses.
    cap=cap if cap is not None else 3*cfg["population_size"]*cfg["steps"]
    if not isinstance(cap,int) or cap<1:
        raise ValueError("The call cap must be a positive integer.")
    with (output/"runs.csv").open() as stream:
        summaries=list(csv.DictReader(stream))
    samples={}
    for summary in summaries:
        case,seed=summary["case"],int(summary["seed"])
        rho=float(summary["rho"]) if summary["rho"] else None
        method=summary["method"]
        stem=f"{case}__s{seed:02d}__{method}"+(f"__rho{rho:g}" if rho is not None else "")
        with gzip.open(output/"traces"/f"{stem}.csv.gz","rt") as stream:
            rows=list(csv.DictReader(stream))
        observation,covered=last_observation(rows,cap)
        samples[(case,seed,rho)]={
            "relative_gap":float(observation["relative_best_gap"]),
            "used_calls":int(observation["total_oracle_calls"]),
            "time":float(observation["time"]),"covered":covered,
            "crossed":int(observation["crossed_particles"]),
            "target":int(observation["target_particles"])>0,
            "source_stop":summary["stop_reason"]}
    pairs=[]
    for case in cfg["cases"]:
        for seed in range(cfg["seeds"]):
            base=samples[(case,seed,None)]
            for rho in cfg["rhos"]:
                collision=samples[(case,seed,rho)]
                valid=base["covered"] and collision["covered"]
                difference=collision["relative_gap"]-base["relative_gap"]
                outcome=("improved" if difference < -1e-6 else "worse" if difference > 1e-6 else "tie") if valid else "not_covered"
                pairs.append({"case":case,"seed":seed,"rho":rho,"call_cap":cap,
                              "both_covered":valid,"outcome":outcome,
                              "baseline_gap":base["relative_gap"] if valid else None,
                              "collision_gap":collision["relative_gap"] if valid else None,
                              "baseline_calls":base["used_calls"],"collision_calls":collision["used_calls"],
                              "baseline_time":base["time"],"collision_time":collision["time"],
                              "baseline_crossed":base["crossed"],"collision_crossed":collision["crossed"],
                              "baseline_target":base["target"],"collision_target":collision["target"],
                              "baseline_source_stop":base["source_stop"],"collision_source_stop":collision["source_stop"]})
    aggregate=[]
    for case in cfg["cases"]:
        for rho in cfg["rhos"]:
            selected=[r for r in pairs if r["case"]==case and r["rho"]==rho]
            valid=[r for r in selected if r["both_covered"]]
            aggregate.append({
                "case":case,"rho":rho,"call_cap":cap,"paired_seeds":len(selected),
                "both_covered":len(valid),
                "improved":sum(r["outcome"]=="improved" for r in selected),
                "worse":sum(r["outcome"]=="worse" for r in selected),
                "ties":sum(r["outcome"]=="tie" for r in selected),
                "not_covered":sum(r["outcome"]=="not_covered" for r in selected),
                "median_baseline_gap":float(np.median([r["baseline_gap"] for r in valid])) if valid else None,
                "median_collision_gap":float(np.median([r["collision_gap"] for r in valid])) if valid else None,
                "baseline_target_hits":sum(r["baseline_target"] for r in valid),
                "collision_target_hits":sum(r["collision_target"] for r in valid),
                "baseline_crossed_particles":sum(r["baseline_crossed"] for r in valid),
                "collision_crossed_particles":sum(r["collision_crossed"] for r in valid),
                "maximum_unused_baseline_calls":max((cap-r["baseline_calls"] for r in valid),default=None),
                "maximum_unused_collision_calls":max((cap-r["collision_calls"] for r in valid),default=None),
            })
    _csv(output/"cost_matched_paired.csv",pairs)
    _csv(output/"cost_matched_aggregate.csv",aggregate)
    report={"call_cap":cap,"aggregate":aggregate,
            "notes":["The default cap 3*N*steps is determined by settings, not by observed losses.",
                     "Only actual committed observations at/below the cap are scored; unused calls are recorded.",
                     "Value, gradient and path requests have equal bookkeeping weights, not equal assumed wall-clock costs.",
                     "The same saved independent baseline is reused across contact distances.",
                     "No trajectories were rerun or retuned for this analysis."]}
    _json(output/"cost_matched_summary.json",report)
    if any(row["both_covered"] for row in aggregate):
        _plot_targets(output,aggregate,cfg,cap)
    return report


def _plot_targets(output,aggregate,cfg,cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    cases,rhos=cfg["cases"],cfg["rhos"]
    x=np.arange(len(cases));width=.8/(len(rhos)+1)
    colors=["#555555","#14816a","#d47726","#8057ac"]
    for index,rho in enumerate([None]+rhos):
        selected=[next(row for row in aggregate if row["case"]==case and row["rho"]==(rhos[0] if rho is None else rho)) for case in cases]
        values=[row["baseline_target_hits"] if rho is None else row["collision_target_hits"] for row in selected]
        label="Independent" if rho is None else f"Collisions, rho={rho:g}"
        bars=ax.bar(x+(index-len(rhos)/2)*width,values,width,label=label,color=colors[index%len(colors)])
        ax.bar_label(bars,padding=3,fontsize=9)
    ax.set_xticks(x,[case.replace("_"," ") for case in cases])
    ax.set_ylim(0,cfg["seeds"]+2)
    ax.set_ylabel("Populations reaching the low-loss target")
    ax.set_title(f"Common cap: {cap:,} oracle calls; {cfg['seeds']} paired initial states")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.2,top=.87)
    fig.savefig(output/"cost_matched_targets.png",dpi=170);plt.close(fig)
