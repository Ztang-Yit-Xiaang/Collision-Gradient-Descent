"""Verify fresh shared paths and compare full-policy costs for E43."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit_shared_prefix_transfer import segment_audit
from audit_third_reflection_guard import audit
from finalize_initial_budget import create
from run_later_reflection_warning import CFG, END_STEP, H, decode_rows, read, sha
from run_long_horizon_exploration import ADDED_BUDGET, STOP_STEP, sources
from run_third_reflection_replication_branches import FreshArchive


def audit_replication(root,*,phase="E43"):
    target,extension = root/"third_reflection",root/"extension"
    cfg = read(root/"config.json")
    assert cfg["phase"]==phase and sha(root/"source.zip")==cfg["source_archive_sha256"]
    costs = read(extension/"execution_counts.json")
    ext_runs = {r["context_id"]:r for r in read(extension/"runs.json")}
    baseline = {r["context_id"]:r for r in read(root/"second_reflection"/"paired_outcomes.json")}
    assert len(ext_runs)==costs["continuations"]==len(baseline)
    assert costs["actual_new_queries"]==sum(r["actual_new_queries"] for r in ext_runs.values())
    ledgers = []

    def extra(case,d,ctx,saved,hits,raw,events,start,runs,counts,peaks):
        key = case["context_id"]
        r = ext_runs[key]
        mode,folder,name = list(sources(case,root))[-1]
        assert r["mode"]==mode and r["source_run_id"]==name
        assert sha(folder/"traces"/(name+".csv.gz"))==r["source_sha256"]
        assert r["stop_reason"]=="completed" and r["committed_steps"]==STOP_STEP-END_STEP
        (q,v,f,calls,_),ledger = segment_audit(extension,r,ctx,counts,peaks,start_step=END_STEP)
        assert all(np.array_equal(data[0],raw[-1][k]) for data,k in ((q,"positions"),(v,"velocities"),(f,"values")))
        assert calls[0]==int(raw[-1]["total_oracle_calls"])+4
        assert calls[-1]-r["historical_queries_at_fork"]==r["actual_new_queries"]<=ADDED_BUDGET
        ledgers.append({**ledger,"new_interventions":0,"external_energy_supply":0.})
        baseline_record = baseline[key]
        assert abs(f[0,0]-ctx["initial_loss"]-baseline_record["policy"]["final_rise"])<1e-12
        base_queries = baseline_record["baseline_query_count"]+baseline_record["net_query_change"]+r["actual_new_queries"]
        delta = d["forecast_queries"]
        if d["transferred"]:
            cp = d["checkpoint"]
            step = cp["boundary_step"]-start
            event_index = next(i for i,e in enumerate(events) if e==cp["event"])
            paid = [e for e in events[:event_index+1] if e["step"]==step]
            groups = 1+sum(b["time"]-a["time"]>CFG.physics.time_tolerance for a,b in zip(paid,paid[1:]))
            fork = int(raw[step-1]["total_oracle_calls"])+8*groups+2*len(paid)
            suffix = int(raw[-1]["total_oracle_calls"])-fork+r["actual_new_queries"]
            delta += runs[key,"capped"]["counts"]["total_oracle_calls"]-suffix
        return {"e40_policy_queries":base_queries,"net_policy_query_change":delta,
                "transfer_suffix_query_change":delta-d["forecast_queries"],
                "e40_transfers":baseline_record["baseline_transfers"]+int(case["transferred"])}

    audit(target,data=FreshArchive(),parent_root=root,extension=extension,case_root=root/"second_reflection",extra=extra)
    rows = read(target/"paired_outcomes.json")
    result = read(target/"results.json")
    k = result["saved"]
    eligible = sum(r["baseline"]["saved"] is not None and r["retained_recovery"] is None for r in rows)
    checks = {"at_least_five_eligible":eligible>=5,"at_least_75_percent_prevented":bool(eligible and k["prevented"]>=.75*eligible),
              "no_new_or_earlier_crossings":all(result[m]["induced"]==result[m]["earlier"]==0 for m in ("saved","including_contacts")),
              "at_most_10_percent_unnecessary":bool(result["new_transfers"] and k["unnecessary_transfers"]<=.1*result["new_transfers"])}
    base = sum(r["e40_policy_queries"] for r in rows)
    delta = sum(r["net_policy_query_change"] for r in rows)
    transfers = sum(r["e40_transfers"] for r in rows)
    create(target/"replication_results.json",{**result,"eligible_saved_deviations":eligible,"checks":checks,"passed_screen":all(checks.values()),
        "total_policy_queries":[base,base+delta],"net_policy_query_change":delta,
        "net_suffix_query_change":delta-result["new_forecast_queries"],"total_transfers":[transfers,transfers+result["new_transfers"]],
        "warnings_by_initial_energy":dict(Counter(r["incumbent_energy"] for r in rows if r["warning"]))})
    create(extension/"physical_ledgers.json",ledgers)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit_replication(parser.parse_args().output)
