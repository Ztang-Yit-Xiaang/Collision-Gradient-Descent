"""Saved-data audit reconstructing physical drag independently of model helpers."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def audit(output):
    output=Path(output);load=lambda p:json.loads(p.read_text())
    cfg=load(output/"config.json");runs=load(output/"runs.json")
    assert len(runs)==len(cfg["cases"])*len(cfg["seeds"])*9
    assert len({(r["case"],r["seed"],r["method"],r["schedule"]) for r in runs})==len(runs)
    assert hashlib.sha256((output/"source.zip").read_bytes()).hexdigest()==cfg["source_archive_sha256"]
    with zipfile.ZipFile(output/"source.zip") as archive:
        for name,h in cfg["source_sha256"].items(): assert hashlib.sha256(archive.read(name)).hexdigest()==h
    peak_q=peak_h=peak_p=peak_pair=0.;checked=events_checked=0
    for r in runs:
        name=f"{r['case']}__s{r['seed']}__{r['method']}__{r['schedule']}"
        with gzip.open(output/"traces"/(name+".csv.gz"),"rt") as f:trace=list(csv.DictReader(f))
        events=load(output/"events"/(name+".json"));attempts=load(output/"map_attempts"/(name+".json"))
        initial=load(output/"initial_states"/f"{r['case']}__s{r['seed']}.json")
        assert json.loads(trace[0]["positions"])==initial["positions"]
        assert json.loads(trace[0]["velocities"])==initial["velocities"]
        assert r["total_oracle_calls"]<=cfg["oracle_budget"]
        assert r["total_oracle_calls"]==sum(r[k] for k in ("value_calls","gradient_calls","path_calls"))
        assert r["positive_supply"]==0 and r["no_new_particles"]
        assert len(attempts)==r["map_attempts"]
        pairs=[e for e in events if e["kind"]=="pair"]
        assert len(pairs)==r["pair_collisions"]==sum(a["committed"] for a in attempts)
        assert len(attempts)-len(pairs)==r["uncommitted_maps"]>=0
        bystep={}
        for e in events:bystep.setdefault(e["step"],[]).append(e)
        previous=None;initial_h=None;total_q=0.;hits=[];exposure=0.
        for t in trace:
            pos=np.array(json.loads(t["positions"]));vel=np.array(json.loads(t["velocities"]));vals=json.loads(t["values"])
            k=.5*np.sum(vel*vel);p=vel.sum(axis=0);h=k+.5*sum(pos[:,-1])
            assert abs(k-float(t["kinetic"]))<1e-10 and abs(h-float(t["energy"]))<1e-10
            hits.append(any(v<=initial["target"] and (initial["boundary"] is None or q[0]<initial["boundary"]) for v,q in zip(vals,pos)))
            if previous is None:initial_h=h
            else:
                index=int(t["step"]);dt=cfg["physics"]["h"];time=(index-.5)*dt
                base=-np.log(.995)/.05
                rate=base if r["schedule"]=="weak" else (base+.5)/2 if r["schedule"]=="constant" else base+(.5-base)*min(time/25,1)
                beta=.995 if r["schedule"]=="weak" else np.exp(-rate*dt)
                assert abs(float(t["drag_beta"])-beta)<1e-15 and abs(float(t["drag_rate"])-rate)<1e-14
                q=(beta**2-1)*previous[0];total_q+=q;exposure+=rate*dt
                qerr=abs(q-float(t["damping_energy"]));peak_q=max(peak_q,qerr);assert qerr<1e-10
                wall=sum((np.array(e["after_i"])-np.array(e["before_i"]) for e in bystep.get(index,[]) if e["kind"]=="wall"),np.zeros(3))
                perr=np.linalg.norm(p-beta*previous[1]-wall-np.array([0,0,-4*.5*dt]))
                peak_p=max(peak_p,float(perr));assert perr<1e-10
            herr=abs(h-initial_h-total_q-float(t["cumulative_impulse_error"])-float(t["cumulative_residual"]))
            peak_h=max(peak_h,herr);assert herr<1e-10
            previous=(k,p);checked+=1
        assert abs(r["removed_energy"]+total_q)<1e-10
        assert abs(r["committed_drag_exposure"]-exposure)<1e-10
        assert r["final_target_hit"]==hits[-1] and r["observed_ever_hit"]==any(hits)
        assert r["hit_exit_transitions"]==sum(a and not b for a,b in zip(hits,hits[1:]))
        assert r["final_value"]==min(json.loads(trace[-1]["values"]))
        assert r["best_value"]==min(min(json.loads(t["values"])) for t in trace)
        assert r["final_observation_calls"]==int(trace[-1]["total_oracle_calls"])<=r["total_oracle_calls"]
        for e in pairs:
            a,b,c,d=[np.array(e[k]) for k in ("before_i","before_j","after_i","after_j")]
            err=max(abs(.5*(c@c+d@d-a@a-b@b)),np.linalg.norm(c+d-a-b));peak_pair=max(peak_pair,float(err));assert err<1e-10
            events_checked+=1
    result={"passed":True,"runs":len(runs),"failures_retained":sum(r["failed"] for r in runs),
            "trajectory_rows_checked":checked,"pair_events_checked":events_checked,
            "max_heat_formula_error":peak_q,"max_energy_ledger_error":peak_h,
            "max_momentum_ledger_error":peak_p,"max_pair_conservation_error":peak_pair}
    (output/"audit.json").write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");print(json.dumps(audit(p.parse_args().output),indent=2))
