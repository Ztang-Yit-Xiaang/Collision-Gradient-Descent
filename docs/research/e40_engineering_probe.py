"""Check the E40 triggered branch and charged prefix on one archived positive."""

from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys

from audit_capped_reflection_transfer import capped_fraction
from audit_later_reflection_warning import audit_branch
from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_second_reflection_branches import selected_source, simulate
from run_later_reflection_warning import read, sha


def run():
    root = Path("output/collision_contact_window_guard")
    e39 = Path("output/collision_capped_reflection_transfer")
    target = Path("output/collision_second_reflection_capped_policy_preflight/engineering_probe")
    deps = checked_hashes(root,[root/f for f in ("contexts.json","paired_outcomes.json","runs.json")])
    deps.update(checked_hashes(e39,[e39/"paired_outcomes.json"]))
    known = next(r for r in read(e39/"paired_outcomes.json") if r["warning"])
    name = known["context_id"]
    row = next(r for r in read(root/"paired_outcomes.json") if r["context_id"]==name)
    context = next(r for r in read(root/"contexts.json") if r["context_id"]==name)
    *_,files = selected_source(root,row,context)
    deps.update(checked_hashes(root,files))
    target.mkdir()
    for f in ("traces","events","uncommitted","fractions"):
        (target/f).mkdir()
    create(target/"config.json",{"context_id":name,"source_sha256":sha(Path(__file__)),"parent_hashes":deps,
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"python":sys.version,"platform":platform.platform(),
        "launch_utc":datetime.now(timezone.utc).isoformat(),"thread_environment":{k:os.environ.get(k) for k in (
            "PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    case,branch = simulate(root,target,row,context)
    create(target/"contexts.json",[case])
    create(target/"runs.json",[branch] if branch else [])
    assert branch is not None and case["warning"] and case["forecast_queries"]==1
    checks,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    outcome,_,_,_,_,ledger = audit_branch(target,branch,case["checkpoint"],context,checks,peaks,
                                         fraction_audit=capped_fraction)
    for k in ("first_saved_crossing","first_crossing_including_contacts","final_rise","within_limit_at_end"):
        assert outcome[k]==known["branches"]["capped"][k]
    frac = read(target/"fractions"/(branch["run_id"]+".json"))
    assert frac["total_oracle_calls_at_end"]-branch["actual_fraction_calls"]==branch["charged_prefix_queries"]
    assert branch["charged_prefix_queries"]+branch["actual_fraction_calls"]+branch["actual_tail_calls"]==branch["counts"]["total_oracle_calls"]
    create(target/"physical_ledgers.json",[ledger])
    result = {"passed":True,"new_independent_seeds":0,"continuations":1,"forecast_queries":1,
              "actual_new_solver_queries":1+branch["actual_fraction_calls"]+branch["actual_tail_calls"],
              "charged_prefix_queries":branch["charged_prefix_queries"],"max_residuals":peaks,**checks}
    create(target/"audit.json",result)
    print(result)


if __name__=="__main__":
    run()
