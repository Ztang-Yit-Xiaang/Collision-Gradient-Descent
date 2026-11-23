"""Complete the common E40 path and branch once at the third reflection."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from _source_paths import manifest_path
from audit_second_reflection_capped_policy import audit as audit_second
from collision_optim.strategic_experiment import _json
from run_long_horizon_exploration import simulate as extend, sources
from run_second_reflection_branches import run as run_second
from run_third_reflection_guard import Archive, save, simulate, read, sha
from run_third_reflection_replication import SEEDS


class FreshArchive(Archive):
    def __init__(self):
        self.dependencies = {}

    def verify(self,path):
        self.dependencies.setdefault(str(path),sha(path))
        return path


def process(root,case,context,saved_window=False,contact_fraction=None):
    extension,guard = root/"extension",root/"third_reflection"
    mode,folder,key = list(sources(case,root))[-1]
    digest = sha(folder/"traces"/(key+".csv.gz"))
    retained = extend(extension,case,mode,folder,key,digest)
    assert retained["stop_reason"]=="completed",retained
    decision,runs = simulate(guard,case,context,data=FreshArchive(),parent=root,extension=extension,
                             saved_window=saved_window,contact_fraction=contact_fraction)
    return retained,decision,runs


def save_all(root,retained,decisions,runs):
    _json(root/"extension"/"runs.json",retained)
    _json(root/"extension"/"execution_counts.json",{"continuations":len(retained),
        "actual_new_queries":sum(r["actual_new_queries"] for r in retained)})
    return save(root/"third_reflection",decisions,runs)


def run(root,workers,*,phase="E43",seed_sets=SEEDS,saved_window=False,contact_fraction=None):
    cfg = read(root/"config.json")
    assert cfg["phase"]==phase
    manifest = cfg["source_sha256"]
    for name in ("run_third_reflection_replication_branches.py","run_third_reflection_guard.py"):
        path = Path("scripts", name)
        assert sha(path)==manifest[manifest_path(path, manifest)]
    run_second(root,workers,phase=phase,seed_sets=seed_sets)
    audit_second(root)
    cases = read(root/"second_reflection"/"contexts.json")
    contexts = {r["context_id"]:r for r in read(root/"contexts.json")}
    for d in ("extension","third_reflection"):
        for sub in ("traces","events","fractions","uncommitted"):
            (root/d/sub).mkdir(parents=True)
    dependencies = [root/"config.json",root/"audit.json",root/"second_reflection"/"audit.json",
                    root/"second_reflection"/"contexts.json",root/"second_reflection"/"paired_outcomes.json"]
    if cfg["stage"]=="formal":
        lock = Path(cfg["launch_command"][cfg["launch_command"].index("--lock")+1])/"third_reflection"/"audit.json"
        assert read(lock)["passed"]
        dependencies.append(lock)
    _json(root/"third_reflection"/"config.json",{"phase":phase,"stage":cfg["stage"],"contexts":len(cases),
        "saved_window":saved_window,"contact_fraction":contact_fraction,"new_query_cap":len(cases)*(4096+(3 if saved_window else 1)),
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"python":sys.version,"numpy":np.__version__,
        "platform":platform.platform(),"workers":workers,"launch_utc":datetime.now(timezone.utc).isoformat(),
        "dependency_data_sha256":{str(p):sha(p) for p in dependencies},"thread_environment":{k:os.environ.get(k) for k in (
            "PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    retained,decisions,runs = [],[],[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(process,root,c,contexts[c["context_id"]],saved_window,contact_fraction) for c in cases]
        try:
            for i,job in enumerate(jobs,1):
                ext,d,new = job.result()
                retained.append(ext)
                decisions.append(d)
                runs.extend(new)
                if i%80==0 or i==len(jobs):
                    costs = save_all(root,retained,decisions,runs)
                    print(f"third reflection: {i}/{len(jobs)} contexts, {len(runs)} capped continuations",flush=True)
        except BaseException:
            save_all(root,retained,decisions,runs)
            _json(root/"third_reflection"/"task_failure.json",{"traceback":traceback.format_exc(),"unreturned_cost":"Unknown"})
            raise
    _json(root/"third_reflection"/"completion_observation.json",{**costs,"observed_utc":datetime.now(timezone.utc).isoformat()})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    run(args.output,args.workers)
