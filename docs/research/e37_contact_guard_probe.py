"""Exercise the locked E37 guard on the three known E36 contact crossings."""

from pathlib import Path

from collision_optim.contact_window import forecast_first_contact
from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.event_replay import unpack
from collision_optim.gravity import GravityConfig
from collision_optim.population import PopulationConfig, PopulationOracle
from finalize_initial_budget import create, sha
from run_conditional_transfer import read


def probe():
    parent=Path("output/collision_second_step_protection")
    target=Path("output/collision_contact_window_guard_preflight")
    assert read(target/"audit.json")["passed"]
    fraction=read(target/"config.json")["contact_guard_fraction"]
    manifest={a["path"]:a["sha256"] for a in read(parent/"raw_artifact_manifest.json")["files"]}
    files=[parent/"contexts.json",parent/"policy_recovery_cases.json"]
    for p in files:
        assert sha(p)==manifest[str(p)]
    cases=read(files[1])["two_stage_plus_step2"]
    contexts={c["context_id"]:c for c in read(files[0])}
    rows=[]
    for case in cases:
        c=contexts[case["context_id"]]
        f=c["families"]["base"]
        assert fraction*1e-4<=c["second_score"]<=1e-4
        oracle=PopulationOracle(RotatedEnergyWells(8,c["seed"]))
        cfg=PopulationConfig(physics=GravityConfig(),rho=.24,collisions=False)
        found=forecast_first_contact(oracle,unpack(c["second_state"]).particles[0],cfg,2,8)
        counts=oracle.counts()
        assert counts=={"value_calls":0,"gradient_calls":0,"path_calls":1}
        assert found is not None and found["loss"]-c["initial_loss"]>1e-4
        assert abs(found["time"]-f["contact_time"])<1e-12 and abs(found["loss"]-f["contact"]["contact_value"])<1e-12
        rows.append({"seed":c["seed"],"context_id":c["context_id"],"score":c["second_score"],
                     "predicted_contact":found,"actual_contact_time":f["contact_time"],
                     "actual_contact_loss":f["contact"]["contact_value"],"oracle_calls":counts})
    assert len(rows)==3
    create(target/"engineering_guard_probe.json",{
        "passed":True,"source_sha256":{str(p):sha(p) for p in files},"cases":rows,
        "new_trajectories":0,"actual_probe_queries":sum(sum(r["oracle_calls"].values()) for r in rows),
        "command":"PYTHONPATH=src:. python3 docs/research/e37_contact_guard_probe.py",
    })


if __name__=="__main__":
    probe()
