"""Execute E47's curvature screen on the sealed E46 launch states."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import platform
import sys
import traceback

from collision_optim.curvature_energy import RotatedEnergyWells
from collision_optim.event_replay import unpack
from collision_optim.population import PopulationOracle
from collision_optim.reflection_window import forecast_flight
from collision_optim.shared_prefix import capped_transfer_at_checkpoint
from collision_optim.strategic_experiment import _json
from run_capped_reflection_transfer import checked_hashes
from run_contact_warning_replication import continue_branch, snapshot
from run_post_contact_positive_stress import (
    ADDED_BUDGET, BASE as E46, BOUNDARY_STEP, CFG, H, LIMIT, START, STOP_STEP, read, sha,
)

BASE = Path("output/collision_post_contact_curvature_screen")
SPEC = Path("docs/superpowers/specs/2026-11-13-post-contact-curvature-screen-design.md")


def directional_curvature_bound(objective,x,displacement):
    coordinate = objective.coordinates(x)[0]
    dx,dy = objective.coordinates(displacement)
    radius = max(abs(coordinate),abs(coordinate+dx))
    return float(max(0.,3.*radius**2-1.)*dx**2+objective.stiffness*dy**2)


def predict(oracle,objective,state,initial,*,second_saved_point=False,follow_second_contact=False,
            third_saved_point=False,reflection_loss_bounds=None,fourth_saved_point=False):
    forecast = forecast_flight(oracle,state.particles[0],CFG,START,BOUNDARY_STEP,STOP_STEP,
        saved_limit=initial+LIMIT,
        curvature_bound=lambda x,d:directional_curvature_bound(objective,x,d),
        second_saved_point=second_saved_point, follow_second_contact=follow_second_contact,
        third_saved_point=third_saved_point, fourth_saved_point=fourth_saved_point,
        reflection_loss_bounds=(None if reflection_loss_bounds is None else
            lambda *args: reflection_loss_bounds(objective,*args)))
    points = forecast["observations"]+[forecast[k] for k in ("after_contact_saved","second_after_contact_saved")
                                       if forecast.get(k) is not None]
    points += forecast.get("extended_observations", [])
    return {**forecast,"counts":oracle.counts(),"warning":any(p["loss"]-initial>LIMIT for p in points)}


def run(args,*,phase="E47",spec=SPEC,comparison=None,second_saved_point=False):
    root = args.output
    parent = Path(str(E46)+"_preflight") if args.stage=="preflight" else E46
    paths = [parent/(n+".json") for n in ("contexts","runs","paired_outcomes","audit","config","physical_ledgers")]
    deps = checked_hashes(parent,paths)
    manifest = parent/"raw_artifact_manifest.json"
    deps[str(manifest)] = sha(manifest)
    assert read(parent/"audit.json")["passed"]
    if comparison is not None:
        comparison = Path(str(comparison)+"_preflight") if args.stage=="preflight" else comparison
        deps.update(checked_hashes(comparison,[comparison/(n+".json") for n in ("audit","paired_outcomes","results")]))
        manifest = comparison/"raw_artifact_manifest.json"
        deps[str(manifest)] = sha(manifest)
        assert read(comparison/"audit.json")["passed"]
    if args.lock:
        lock = args.lock/"audit.json"
        assert read(lock)["passed"]
        deps[str(lock)] = sha(lock)
    contexts = read(parent/"contexts.json")
    archived = {r["run_id"]:r for r in read(parent/"runs.json")}
    assert len(contexts)==(3 if args.stage=="preflight" else 48)
    root.mkdir(parents=True)
    for d in ("traces","events","fractions","uncommitted"):
        (root/d).mkdir()
    source = snapshot(root,spec)
    if args.lock:
        assert source==read(args.lock/"config.json")["source_sha256"]
    _json(root/"config.json",{"phase":phase,"stage":args.stage,"parent":str(parent),
        "comparison":str(comparison) if comparison is not None else None,"second_saved_point":second_saved_point,
        "new_independent_seed_groups":0,
        "threshold":LIMIT,"start":START,"end_step":STOP_STEP,"h":H,"gravity":CFG.physics.gravity,
        "floating_loss_margin":1e-12,"binary_interactions":False,"forecast_query_cap":3*len(contexts),
        "continuation_query_cap":ADDED_BUDGET,"source_sha256":source,"source_archive_sha256":sha(root/"source.zip"),
        "dependency_data_sha256":deps,"launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),
        "python":sys.version,"platform":platform.platform(),"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    decisions,runs = [],[]
    probes = 0
    try:
        for c in contexts:
            objective = RotatedEnergyWells(8,c["seed"])
            oracle = PopulationOracle(objective,total_budget=3)
            state = unpack(c["start"])
            try:
                forecast = predict(oracle,objective,state,c["initial_loss"],second_saved_point=second_saved_point)
            finally:
                probes += sum(oracle.counts().values())
            mode = "capped" if forecast["warning"] else "keep"
            key = c["context_id"]+"__"+mode
            reused = key in archived
            if not reused:
                moved = capped_transfer_at_checkpoint(state,0)
                branch = continue_branch(root,c["context_id"],mode,moved,
                    PopulationOracle(objective,total_budget=ADDED_BUDGET),CFG,BOUNDARY_STEP*H-START,
                    end_step=STOP_STEP,boundary_step=BOUNDARY_STEP)
                runs.append(branch)
                if branch["stop_reason"]!="completed":
                    raise RuntimeError(branch)
            decisions.append({"context_id":c["context_id"],"seed":c["seed"],"forecast":forecast,
                              "selected_run":key,"reused":reused})
    except Exception:
        _json(root/"failure.json",{"traceback":traceback.format_exc(),"paid_forecast_queries":probes,
                                   "returned_continuations":len(runs)})
        raise
    finally:
        _json(root/"decisions.json",decisions)
        _json(root/"runs.json",runs)
        cost = {"contexts":len(decisions),"continuations":len(runs),"reused_selected_continuations":sum(d["reused"] for d in decisions),
                "actual_new_forecast_queries":probes,
                "actual_new_continuation_queries":sum(r["counts"]["total_oracle_calls"] for r in runs)}
        cost["actual_new_queries"] = cost["actual_new_forecast_queries"]+cost["actual_new_continuation_queries"]
        _json(root/"execution_counts.json",cost)
    _json(root/"completion_observation.json",{**cost,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(cost)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","formal"),required=True)
    parser.add_argument("--lock",type=Path)
    args = parser.parse_args()
    if (args.stage=="formal")!=bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args)
