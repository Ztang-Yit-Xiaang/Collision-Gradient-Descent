"""Derived diagnostics of E16, without executing new trajectories."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from collision_optim.cooling import trajectory_metrics
from collision_optim.strategic_experiment import _json, _csv


def analyze(parent, output):
    parent=Path(parent); output=Path(output)
    if output.exists():raise ValueError("Use a fresh derived-output directory.")
    output.mkdir(parents=True)
    inputs={};metrics=[]
    for r in json.loads((parent/"runs.json").read_text()):
        name=f"{r['case']}__s{r['seed']}__{r['method']}__rho{r['rho']}"
        trace=parent/"traces"/(name+".csv.gz")
        with gzip.open(trace,"rt") as f:rows=list(csv.DictReader(f))
        initial=json.loads((parent/"initial_states"/f"{r['case']}__s{r['seed']}.json").read_text())
        inputs[str(trace)]=hashlib.sha256(trace.read_bytes()).hexdigest()
        derived=trajectory_metrics(rows,r["target"],initial["boundary"])
        assert derived["observed_ever_hit"]==r["population_target_hit"]
        assert derived["observed_final_hit"]==r["final_target_hit"]
        metrics.append({"case":r["case"],"seed":r["seed"],"method":r["method"],"rho":r["rho"],"failed":r["failed"],**derived})
    groups=defaultdict(list)
    for r in metrics:groups[r["case"],r["method"],r["rho"]].append(r)
    summaries=[]
    for (case,mode,rho),group in groups.items():
        summaries.append({"case":case,"method":mode,"rho":rho,"runs":len(group),
            "ever_hits":sum(r["observed_ever_hit"] for r in group),"final_hits":sum(r["observed_final_hit"] for r in group),
            "lost_after_hit":sum(r["lost_after_any_hit"] for r in group),
            **{"median_"+k:float(np.median([r[k] for r in group])) for k in
               ("final_kinetic","removed_energy","final_clearance_potential","tail_target_occupancy","hit_exit_transitions")}})
    for name,data in (("metrics",metrics),("aggregate",summaries)):_json(output/(name+".json"),data);_csv(output/(name+".csv"),data)
    source=Path(__file__);shutil.copy2(source,output/source.name)
    _json(output/"config.json",{"parent":str(parent),"new_optimization_runs":0,"derived_rows":len(metrics),
          "inputs_sha256":inputs,"source_sha256":{source.name:hashlib.sha256(source.read_bytes()).hexdigest()},
          "note":"E16 diagnostic view, not additional executions. Baseline drag exposure not stored here; no inferred continuous hits."})
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("parent");p.add_argument("output");a=p.parse_args();analyze(a.parent,a.output)
