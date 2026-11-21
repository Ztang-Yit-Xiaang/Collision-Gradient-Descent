"""Merge disjoint scattering shards; merged tables are views, not extra runs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from collision_optim.scattering_experiment import aggregate, summarize_pairs
from collision_optim.strategic_experiment import _csv, _json


def merge(output):
    output=Path(output); config=json.loads((output/"config.json").read_text())
    if (output/"runs.json").exists(): raise ValueError("Merged runs already exist.")
    shards=sorted((output/"shards").iterdir()); runs=[]; all_seeds=[]; provenance=[]; maps=None
    for shard in shards:
        cfg=json.loads((shard/"config.json").read_text())
        for field in ("steps","cases","rhos","methods","physics","oracle_budget","initial_total_kinetic","population_size"):
            assert cfg[field]==config[field],(shard,field)
        assert cfg["source_sha256"]==config["source_sha256"]
        all_seeds.extend(cfg["seeds"])
        data=json.loads((shard/"runs.json").read_text())
        assert len(data)==3*len(cfg["seeds"])*5
        runs.extend([{**r,"source_shard":str(shard)} for r in data])
        current=json.loads((shard/"mechanism_maps.json").read_text())
        if maps is None: maps=current
        else: assert maps==current
        for folder in ("traces","events","initial_states","map_attempts"):
            (output/folder).mkdir(exist_ok=True)
            for source in (shard/folder).iterdir():
                target=output/folder/source.name
                assert not target.exists(),target
                shutil.copy2(source,target)
        provenance.append({"shard":str(shard),"unique_runs":len(data),"config_sha256":hashlib.sha256((shard/"config.json").read_bytes()).hexdigest(),"source_archive_sha256":cfg["source_archive_sha256"]})
    assert sorted(all_seeds)==sorted(config["seeds"]) and len(set(all_seeds))==len(all_seeds)
    keys=[(r["case"],r["seed"],r["method"],r["rho"]) for r in runs]
    assert len(keys)==len(set(keys))==300
    runs.sort(key=lambda r:(r["case"],r["seed"],r["method"],r["rho"] or 0))
    for name,data in (("runs",runs),("paired",summarize_pairs(runs)),("aggregate",aggregate(runs)),("mechanism_maps",maps)):
        _json(output/(name+".json"),data);_csv(output/(name+".csv"),data)
    _json(output/"merge_provenance.json",{"shards":provenance,"unique_formal_runs":len(runs),"derived_view_not_additional_runs":True,"mechanism_scans_identical_across_shards":True})
    print(f"Merged {len(runs)} unique formal runs from {len(shards)} disjoint shards.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    merge(parser.parse_args().output)
