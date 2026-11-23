"""E41: extend the selected E40 trajectories on their original damping clock."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from collision_optim.curvature_energy import IndependentEnergyPopulation, RotatedEnergyWells
from collision_optim.event_replay import unpack
from collision_optim.population import PopulationOracle, run_population
from collision_optim.strategic_experiment import _csv, _json
from run_capped_reflection_transfer import checked_hashes
from run_contact_warning_replication import snapshot
from run_later_reflection_warning import CFG, END_STEP, H, decode_rows, read, sha
from run_postbounce_warning_transfer import load_counters
from run_shared_prefix_transfer import save_segment, stop_record

PARENT = Path("output/collision_second_reflection_capped_policy")
ANALYSIS = Path("docs/research/e40_exploration")
BASE = Path("output/collision_long_horizon_exploration")
SPEC = Path("docs/superpowers/specs/2026-11-12-e40-long-horizon-exploration-design.md")
STOP_STEP = 100
ADDED_BUDGET = 4096


def sources(case,parent=PARENT):
    yield "e37",parent,case["source"]
    if case["transferred"]:
        yield "capped_policy",parent/"second_reflection",case["checkpoint"]["checkpoint_id"]+"__capped"


def simulate(root,case,mode,folder,key,digest):
    trace = folder/"traces"/(key+".csv.gz")
    assert sha(trace)==digest
    last = decode_rows(folder,key)[-1]
    initial = unpack(last)
    paid = int(last["total_oracle_calls"])
    objective = RotatedEnergyWells(8,case["seed"])
    oracle = PopulationOracle(objective,total_budget=paid+ADDED_BUDGET)
    load_counters(oracle,last)
    method = IndependentEnergyPopulation(CFG)
    method.committed_steps = END_STEP
    result = run_population(method,oracle,initial,STOP_STEP-END_STEP)
    name = case["context_id"]+"__"+mode
    assert result.rows[0]["positions"]==last["positions"] and result.rows[0]["velocities"]==last["velocities"]
    assert result.rows[0]["values"]==last["values"]
    save_segment(root,name,result)
    return {"run_id":name,"context_id":case["context_id"],"mode":mode,"seed":case["seed"],
            "source_run_id":key,"source_folder":str(folder),"source_trace":str(trace),"source_sha256":digest,
            "historical_queries_at_fork":paid,"actual_new_queries":result.counts["total_oracle_calls"]-paid,
            **stop_record(result)}


def save(root,runs):
    _json(root/"runs.json",runs)
    if runs:
        _csv(root/"runs.csv",runs)
    counts = {"continuations":len(runs),"new_initial_states":0,"new_forecast_queries":0,
              "actual_new_queries":sum(r["actual_new_queries"] for r in runs),
              "completed":sum(r["stop_reason"]=="completed" and r["committed_steps"]==STOP_STEP-END_STEP for r in runs)}
    _json(root/"execution_counts.json",counts)
    return counts


def run(args):
    root = args.output
    paths = [PARENT/"contexts.json",PARENT/"audit.json",PARENT/"second_reflection"/"contexts.json",
             PARENT/"second_reflection"/"paired_outcomes.json",PARENT/"second_reflection"/"audit.json"]
    dependencies = checked_hashes(PARENT,paths)
    manifest = PARENT/"raw_artifact_manifest.json"
    dependencies[str(manifest)] = sha(manifest)
    assert read(paths[1])["passed"] and read(paths[-1])["passed"]
    record = ANALYSIS/"record.json"
    analysis_hashes = read(record)["derived_artifacts_sha256"]
    explorer_path = ANALYSIS/"per_context.json"
    assert sha(explorer_path)==analysis_hashes[str(explorer_path)]
    dependencies.update({str(record):sha(record),str(explorer_path):sha(explorer_path)})
    if args.lock:
        check = args.lock/"audit.json"
        assert read(check)["passed"]
        dependencies[str(check)] = sha(check)
    cases = read(paths[2])
    assert len(cases)==480 and sum(r["transferred"] for r in cases)==20
    if args.stage=="preflight":
        entrant = {r["context_id"] for r in read(explorer_path) if r["transferred"] and r["e37"]["entry_events"]}
        cases = [next(r for r in cases if not r["transferred"]),next(r for r in cases if r["transferred"]),
                 next(r for r in cases if r["context_id"] in entrant)]
        assert len({r["context_id"] for r in cases})==3
    root.mkdir(parents=True)
    for d in ("traces","events","uncommitted"):
        (root/d).mkdir()
    source = snapshot(root,SPEC,extra_paths=(Path("docs/research/e40_exploration.py"),))
    if args.lock:
        assert source==read(args.lock/"config.json")["source_sha256"]
    hashes = {x["path"]:x["sha256"] for x in read(manifest)["files"]}
    jobs = [(c,m,f,k,hashes[str(f/"traces"/(k+".csv.gz"))]) for c in cases for m,f,k in sources(c)]
    assert len(jobs)==(5 if args.stage=="preflight" else 500)
    _json(root/"config.json",{"phase":"E41","stage":args.stage,"parent":str(PARENT),"contexts":len(cases),
        "new_independent_seed_groups":0,"reused_seeds":sorted({r["seed"] for r in cases}),"start_step":END_STEP,
        "end_step":STOP_STEP,"h":H,"gravity":CFG.physics.gravity,"binary_interactions":False,
        "new_interventions":0,"new_query_cap_per_continuation":ADDED_BUDGET,"new_query_cap":len(jobs)*ADDED_BUDGET,
        "source_sha256":source,"source_archive_sha256":sha(root/"source.zip"),"dependency_data_sha256":dependencies,
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"python":sys.version,"numpy":np.__version__,
        "platform":platform.platform(),"workers":args.workers,"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    _json(root/"contexts.json",cases)
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(simulate,root,*job) for job in jobs]
        try:
            for i,job in enumerate(pending,1):
                runs.append(job.result())
                if i%50==0 or i==len(jobs):
                    counts = save(root,runs)
                    print(f"{args.stage}: {i}/{len(jobs)} continuations",flush=True)
        except BaseException:
            save(root,runs)
            _json(root/"task_failure.json",{"traceback":traceback.format_exc(),"unreturned_cost":"Unknown for workers without returned data"})
            raise
    _json(root/"completion_observation.json",{**counts,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(counts,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","formal"),required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=="formal")!=bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight lock.")
    run(args)
