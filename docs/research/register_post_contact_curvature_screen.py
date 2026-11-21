"""Seal E47's paid forecasts and register their reused E46 continuations."""

import argparse
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_post_contact_curvature_screen import BASE, SPEC


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry,validation = read(registry_path),read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers,inventory = reader.fieldnames,list(reader)
    stages = [(Path(str(BASE)+"_preflight"),"engineering_forecast_preflight",3,8),
              (BASE,"paired_forecast_with_archived_stress_continuations",48,132)]
    previous = {Path(r["folder"]) for r in registry["entries"]}
    folders = {r[0] for r in stages}
    assert not previous & folders
    assert {p for p in Path("output").iterdir() if p.is_dir()}==previous|folders
    historical = [a for r in registry["entries"] for a in r["top_level_artifacts"]]
    assert all(sha(Path(a["path"]))==a["sha256"] for a in historical)
    links = check_links()
    source = read(BASE/"config.json")["source_sha256"]
    for root,classification,n,queries in stages:
        cfg,cost,audit = [read(root/(k+".json")) for k in ("config","execution_counts","audit")]
        assert cfg["phase"]=="E47" and cfg["source_sha256"]==source and audit["passed"]
        assert cost["contexts"]==audit["contexts"]==n and cost["actual_new_queries"]==queries
        assert cost["continuations"]==audit["continuations"]==0 and not audit["solver_failures"]
        assert sha(root/"source.zip")==cfg["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip",source))
        validation["source_archive_checks"].append(seal(root,tests,audit,"post_contact_curvature_screen",str(SPEC),(
            str(Path(__file__).resolve().relative_to(Path.cwd())),)))
        tables = {}
        for name,role in (("decisions","new_paid_forecasts"),("runs","new_continuations"),
                          ("paired_outcomes","derived_policy_comparisons"),("second_saved_cases","derived_later_crossings")):
            path = root/(name+".json")
            tables[name] = {"path":str(path),"role":role,"records":len(read(path))}
        entry = {"phase_id":"E47","folder":str(root),"classification":classification,"run_date":"2026-11-13",
                 "run_rows_are_derived":False,"config":cfg,"config_path":str(root/"config.json"),
                 "record_tables":tables,"failures":[],"execution_counts":cost,
                 "counting_note":"New forecasts only. Outcomes and physical ledgers reuse the archived E46 selected branches.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if root==BASE:
            entry["curvature_screen_result"] = read(root/"results.json")
        registry["entries"].append(entry)
        row = {k:0 for k in headers}
        row.update({k:entry[k] for k in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        row.update(derived_records=n)
        inventory.append(row)
    registry["entries"].sort(key=lambda r:r["folder"])
    registry["as_of"] = "2026-11-13"
    validation.update({"date":"2026-11-13","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E47":len(historical),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":0,"new_run_breakdown":{"new_continuations":0,"paid_forecasts":51,"actual_new_queries":140},
        "new_optimization_runs_note":"E47: paid forecast probes with archived E46 dynamics.",
        "new_audit_oracle_calls":{"value_checks":0,"gradient_checks":0},
        "new_execution_segments_with_complete_tables":0,"new_accepted_execution_segments":0,
        "new_formal_mechanism_execution_segments":0,"excluded_engineering_executions":0,
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered 51 forecasts and zero new continuations. Checked {links} links and {len(historical)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
