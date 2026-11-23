"""E39: add one capped continuation at each archived E38 checkpoint."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.event_replay import unpack
from collision_optim.population import PopulationOracle
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_contact_warning_replication import continue_branch, snapshot
from run_later_reflection_warning import CAP, CFG, END_STEP, H, LIMIT, read, sha

SPEC = Path("docs/superpowers/specs/2026-11-12-capped-reflection-transfer-design.md")
BASE = Path("output/collision_capped_reflection_transfer")
PARENTS = {"preflight":Path("output/collision_later_reflection_warning_preflight_verified"),
           "replay":Path("output/collision_later_reflection_warning")}


def checked_hashes(parent, paths):
    sealed = {r["path"]:r["sha256"] for r in read(parent/"raw_artifact_manifest.json")["files"]}
    actual = {str(p):sha(p) for p in paths}
    assert all(sealed[p]==digest for p,digest in actual.items())
    return actual


def run(args):
    root,parent = args.output,PARENTS[args.stage]
    paths = [parent/(name+".json") for name in ("config","audit","contexts","runs","warning_checkpoints")]
    dependencies = checked_hashes(parent,paths)
    dependencies[str(parent/"raw_artifact_manifest.json")] = sha(parent/"raw_artifact_manifest.json")
    assert read(parent/"audit.json")["passed"]
    if args.lock:
        assert read(args.lock/"audit.json")["passed"]
        for f in ("audit.json","config.json"):
            dependencies[str(args.lock/f)] = sha(args.lock/f)
    checks = [c for c in read(parent/"warning_checkpoints.json") if c["branches"]]
    assert len(checks)==(2 if args.stage=="preflight" else 40)
    if args.stage=="replay":
        assert sum(c["warning"] and c["ordinal"]==2 for c in checks)==19
        assert sum(c["already_deviated"] for c in checks)==21
    parent_cases = {c["context_id"]:c for c in read(parent/"contexts.json")}
    source_e37 = Path(read(parent/"config.json")["parent"])
    dependencies.update(checked_hashes(source_e37,[source_e37/"contexts.json"]))
    root.mkdir(parents=True)
    for folder in ("traces","events","fractions","uncommitted"):
        (root/folder).mkdir()
    source = snapshot(root,SPEC)
    if args.lock:
        assert source==read(args.lock/"config.json")["source_sha256"]
    _json(root/"config.json",{
        "phase":"E39","stage":args.stage,"parent":str(parent),"source_e37":str(source_e37),
        "modes":["keep","transfer","capped"],"new_independent_seed_groups":0,
        "reused_seeds":sorted({parent_cases[c["context_id"]]["seed"] for c in checks}),
        "h":H,"end_step":END_STEP,"threshold":LIMIT,"gravity":CFG.physics.gravity,
        "binary_interactions":False,"continuation_query_cap":CAP,"total_new_query_cap":len(checks)*CAP,
        "source_sha256":source,"source_archive_sha256":sha(root/"source.zip"),"dependency_data_sha256":dependencies,
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"python":sys.version,"numpy":np.__version__,
        "platform":platform.platform(),"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in (
            "PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
    })
    _json(root/"contexts.json",checks)
    runs = []
    try:
        for c in checks:
            seed = parent_cases[c["context_id"]]["seed"]
            state = unpack(c["post_reflection"])
            moved = capped_transfer_at_checkpoint(state,0)
            oracle = PopulationOracle(RotatedEnergyWells(8,seed),total_budget=CAP)
            row = continue_branch(root,c["checkpoint_id"],"capped",moved,oracle,CFG,c["remaining_duration"],
                                  end_step=END_STEP,boundary_step=c["boundary_step"])
            runs.append({**row,"seed":seed,"checkpoint_id":c["checkpoint_id"],"context_id":c["context_id"],
                         "no_op":state is moved})
            _json(root/"runs.json",runs)
    except Exception:
        _json(root/"task_failure.json",{"traceback":traceback.format_exc(),"returned_continuations":len(runs)})
        raise
    _csv(root/"runs.csv",runs)
    counts = {"reused_checkpoint_states":len(checks),"new_initial_states":0,"new_forecast_queries":0,
              "reused_continuations":2*len(checks),"continuations":len(runs),
              "actual_fraction_queries":sum(r["actual_fraction_calls"] for r in runs),
              "actual_tail_queries":sum(r["actual_tail_calls"] for r in runs),
              "actual_new_solver_queries":sum(r["counts"]["total_oracle_calls"] for r in runs)}
    _json(root/"execution_counts.json",counts)
    _json(root/"completion_observation.json",{**counts,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(counts)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=PARENTS,required=True)
    parser.add_argument("--lock",type=Path)
    args = parser.parse_args()
    if (args.stage=="replay")!=bool(args.lock):
        parser.error("Replay requires the audited preflight lock.")
    run(args)
