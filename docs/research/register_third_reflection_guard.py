"""Register E42's scan, replays, and engineering attempts without changing parents."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_third_reflection_guard import BASE, SPEC


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry,validation = read(registry_path),read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers,inventory = reader.fieldnames,list(reader)
    stages = [(Path(str(BASE)+suffix),classification,failure) for suffix,classification,failure in (
        ("_preflight","engineering_preflight_startup_failure","Missing ProcessPoolExecutor import before workers launched"),
        ("_preflight_verified","engineering_preflight_audit_retry","Initial audit inherited E38 stop and query cap; corrected audit subsequently passed"),
        ("_preflight_final","engineering_preflight",None),
        ("","retrospective_third_reflection_guard",None))]
    folders = {r for r,_,_ in stages}
    previous = {Path(r["folder"]) for r in registry["entries"]}
    assert not folders & previous and {p for p in Path("output").iterdir() if p.is_dir()}==folders|previous
    history = [a for r in registry["entries"] for a in r["top_level_artifacts"]]
    assert all(sha(Path(a["path"]))==a["sha256"] for a in history)
    links = check_links()
    assert links>373
    total, audits = Counter(), Counter()
    accepted = 0
    source = read(BASE/"config.json")["source_sha256"]
    for root,classification,failure in stages:
        cfg = read(root/"config.json")
        startup = classification=="engineering_preflight_startup_failure"
        audit = {"passed":False,"failure":failure} if startup else read(root/"audit.json")
        cost = {"continuations":0,"actual_new_queries":0,"forecast_queries":0} if startup else read(root/"execution_counts.json")
        assert cfg["phase"]=="E42" and sha(root/"source.zip")==cfg["source_archive_sha256"]
        if not startup:
            assert audit["passed"] and not audit["solver_failures"]
            assert cost["continuations"]==audit["continuations"] and audit["contexts"]==(480 if root==BASE else 3)
        if not failure:
            assert source==cfg["source_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip",cfg["source_sha256"]))
        validation["source_archive_checks"].append(seal(root,tests,audit,"third_reflection_guard",str(SPEC),(
            str(Path(__file__).resolve().relative_to(Path.cwd())),"audit_later_reflection_warning.py",
            "docs/research/e42_policy_cost.py")))
        tables = {}
        for name,role in (("contexts","archived_E41_contexts"),("runs","new_continuations"),
                          ("paired_outcomes","derived_policy_comparison"),("decisions","paid_forecasts_and_checkpoint_states"),
                          ("already_deviated_cases","derived_recovery_states"),("late_crossing_mechanisms","derived_late_crossing_diagnosis")):
            path = root/(name+".json")
            if path.exists():
                tables[name] = {"path":str(path),"role":role,"records":len(read(path))}
        entry = {"phase_id":"E42","folder":str(root),"classification":classification,"run_date":"2026-11-12",
                 "run_rows_are_derived":False,"config":cfg,"config_path":str(root/"config.json"),
                 "record_tables":tables,"failures":[failure] if failure else [],"execution_counts":cost,
                 "counting_note":"Reuse E41's retained path as the control. Only newly executed continuations count as runs.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if root==BASE:
            entry["third_reflection_guard_result"] = read(root/"results.json")
            accepted = cost["continuations"]
        registry["entries"].append(entry)
        row = {k:0 for k in headers}
        row.update({k:entry[k] for k in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        row.update(runs_records=cost["continuations"],suffix_execution_records=cost["continuations"],
                   contact_context_records=tables.get("decisions",{}).get("records",0),
                   derived_records=tables.get("paired_outcomes",{}).get("records",0),recorded_failures=int(bool(failure)))
        inventory.append(row)
        total.update(cost)
        audits.update({k:audit.get(k,0) for k in ("value_checks","gradient_checks")})
    registry["entries"].sort(key=lambda r:r["folder"])
    registry["as_of"] = "2026-11-12"
    validation.update({"date":"2026-11-12","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E42":len(history),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":total["continuations"],"new_run_breakdown":dict(total),
        "new_optimization_runs_note":"E42: new third-reflection continuations only. Archived retained paths and prediction checks are shared inputs.",
        "new_audit_oracle_calls":dict(audits),"new_execution_segments_with_complete_tables":total["continuations"],
        "new_accepted_execution_segments":accepted,"new_formal_mechanism_execution_segments":accepted,
        "excluded_engineering_executions":total["continuations"]-accepted,
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered {total['continuations']} continuations. Checked {links} links and {len(history)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
