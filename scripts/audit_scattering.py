"""Independent raw-event and trajectory audit for conserving scattering."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from _source_paths import source_path


ROOT = Path(__file__).resolve().parents[1]


def load(p): return json.loads(p.read_text())


def audit(output):
    output=Path(output); cfg=load(output/"config.json"); runs=load(output/"runs.json")
    assert len(runs)==len(cfg["cases"])*len(cfg["seeds"])*5
    assert hashlib.sha256((output/"source.zip").read_bytes()).hexdigest()==cfg["source_archive_sha256"]
    for name,digest in load(output/"source_manifest.json").items():
        assert hashlib.sha256(source_path(name, ROOT).read_bytes()).hexdigest()==digest,name
    maps=load(output/"mechanism_maps.json"); assert len(maps)==1515
    peak_transfer=peak_energy=peak_momentum=peak_balance=peak_p_balance=0.; pair_count=rows_checked=0
    for e in maps:
        vi,vj,ai,aj,n=[np.array(e[k]) for k in ("before_i","before_j","after_i","after_j","normal")]
        center=(vi+vj)/2; gain=.5*(ai@ai-vi@vi); predicted=center@((ai-aj-vi+vj)/2)
        assert abs(gain-predicted)<1e-12
        assert abs(ai@ai+aj@aj-vi@vi-vj@vj)<1e-12 and np.linalg.norm(ai+aj-vi-vj)<1e-12
        assert (ai-aj)@n>0
        if e["center"]=="zero": assert abs(gain)<1e-12
    for r in runs:
        name=f"{r['case']}__s{r['seed']}__{r['method']}__rho{r['rho']}"
        with gzip.open(output/"traces"/(name+".csv.gz"),"rt") as stream: trace=list(csv.DictReader(stream))
        events=load(output/"events"/(name+".json")); attempts=load(output/"map_attempts"/(name+".json"))
        impacts=[e for e in events if e["kind"]=="pair"]
        initial=load(output/"initial_states"/f"{r['case']}__s{r['seed']}.json")
        assert np.array_equal(json.loads(trace[0]["positions"]),initial["positions"])
        assert np.array_equal(json.loads(trace[0]["velocities"]),initial["velocities"])
        assert abs(r["initial_energy"]-initial["initial_energy"])<1e-12
        assert abs(r["initial_kinetic"]-.6)<1e-12 and np.linalg.norm(r["initial_momentum"])<1e-12
        assert r["positive_supply"]==0 and r["no_new_particles"]
        assert r["total_oracle_calls"]<=cfg["oracle_budget"]
        assert r["total_oracle_calls"]==sum(r[k] for k in ("value_calls","gradient_calls","path_calls"))
        assert r["pair_collisions"]==len(impacts)
        assert len(attempts)==r["map_attempts"] and r["uncommitted_maps"]==len(attempts)-len(impacts)>=0
        assert sum(a["committed"] for a in attempts)==len(impacts)
        assert min(min(json.loads(t["values"])) for t in trace)==r["best_value"]
        assert min(json.loads(trace[-1]["values"]))==r["final_value"]
        assert int(trace[-1]["total_oracle_calls"])==r["final_observation_calls"]<=r["total_oracle_calls"]
        last_values=json.loads(trace[-1]["values"]); last_positions=json.loads(trace[-1]["positions"])
        hit=any(v<=r["target"] and (initial["boundary"] is None or p[0]<initial["boundary"])
                for v,p in zip(last_values,last_positions))
        assert hit==r["final_target_hit"]
        step_events={}
        for e in events: step_events.setdefault(e["step"],[]).append(e)
        previous=None
        for t in trace:
            q=np.array(json.loads(t["positions"])); v=np.array(json.loads(t["velocities"])); kinetic=.5*np.sum(v*v)
            h=kinetic+cfg["physics"]["gravity"]*sum(q[:,-1]); momentum=v.sum(axis=0)
            assert len(q)==4
            assert abs(h-float(t["energy"]))<1e-10
            assert np.linalg.norm(momentum-np.array(json.loads(t["momentum"])))<1e-10
            error=abs(h-r["initial_energy"]-float(t["cumulative_damping"])-float(t["cumulative_impulse_error"])-float(t["cumulative_residual"]))
            assert error<1e-10; peak_balance=max(peak_balance,error)
            if previous is not None:
                wall=sum((np.array(e["after_i"])-np.array(e["before_i"]) for e in step_events.get(int(t["step"]),[]) if e["kind"]=="wall"),np.zeros(3))
                gravity=np.array([0.,0.,-4*cfg["physics"]["gravity"]*cfg["physics"]["h"]])
                err=np.linalg.norm(momentum-cfg["physics"]["beta"]*previous-wall-gravity)
                assert err<1e-10; peak_p_balance=max(peak_p_balance,float(err))
            previous=momentum; rows_checked+=1
        for index,e in enumerate(impacts):
            vi,vj,ai,aj,n=[np.array(e[k]) for k in ("before_i","before_j","after_i","after_j","normal")]
            energy=abs(.5*(ai@ai+aj@aj-vi@vi-vj@vj)); p_error=np.linalg.norm(ai+aj-vi-vj)
            transfer=abs(.5*(ai@ai-vi@vi)-((vi+vj)/2)@((ai-aj-vi+vj)/2))
            assert max(energy,p_error,transfer)<1e-10 and (ai-aj)@n>0
            assert np.allclose(ai,attempts[index]["after_i"],rtol=0,atol=1e-12)
            assert abs(np.linalg.norm(n)-1)<1e-12
            if r["method"]=="elastic": assert e["angular_impulse_frobenius"]<1e-10
            peak_energy=max(peak_energy,float(energy)); peak_momentum=max(peak_momentum,float(p_error)); peak_transfer=max(peak_transfer,float(transfer)); pair_count+=1
    result={"passed":True,"runs":len(runs),"failures_retained":sum(r["failed"] for r in runs),
            "mechanism_records":len(maps),"trajectory_rows_checked":rows_checked,"pair_events_checked":pair_count,
            "max_pair_energy_error":peak_energy,"max_pair_momentum_error":peak_momentum,
            "max_transfer_formula_error":peak_transfer,"max_raw_energy_ledger_error":peak_balance,
            "max_raw_momentum_ledger_error":peak_p_balance}
    (output/"audit.json").write_text(json.dumps(result,indent=2)+"\n"); return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    print(json.dumps(audit(parser.parse_args().output),indent=2))
