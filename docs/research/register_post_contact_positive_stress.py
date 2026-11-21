"""Seal and register the E46 engineering and stress batches."""

import argparse
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_post_contact_positive_stress import BASE, SPEC


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry,validation = read(registry_path),read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers,inventory = reader.fieldnames,list(reader)
    stages = [(Path(str(BASE)+"_preflight"),"engineering_preflight",3,5),
              (BASE,"constructed_post_contact_positive_stress",48,84)]
    previous = {Path(r["folder"]) for r in registry["entries"]}
    assert {p for p in Path("output").iterdir() if p.is_dir()}==previous|{r[0] for r in stages}
    historical = [a for r in registry["entries"] for a in r["top_level_artifacts"]]
    assert all(sha(Path(a["path"]))==a["sha256"] for a in historical)
    links = check_links()
    source = read(BASE/"config.json")["source_sha256"]
    queries = value_checks = gradient_checks = 0
    for root,classification,n,executions in stages:
        cfg,cost,audit = [read(root/(k+".json")) for k in ("config","execution_counts","audit")]
        assert cfg["phase"]=="E46" and cfg["source_sha256"]==source and audit["passed"]
        assert cost["contexts"]==audit["contexts"]==n and cost["continuations"]==audit["continuations"]==executions
        assert not audit["solver_failures"] and sha(root/"source.zip")==cfg["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip",source))
        validation["source_archive_checks"].append(seal(root,tests,audit,"post_contact_positive_stress",str(SPEC),(
            str(Path(__file__).resolve().relative_to(Path.cwd())),)))
        tables = {}
        for name,role in (("contexts","constructed_initial_states"),("runs","new_trajectory_continuations"),
                          ("paired_outcomes","derived_policy_comparisons"),("physical_ledgers","physical_ledgers")):
            path = root/(name+".json")
            tables[name] = {"path":str(path),"role":role,"records":len(read(path))}
        entry = {"phase_id":"E46","folder":str(root),"classification":classification,"run_date":"2026-11-13",
                 "run_rows_are_derived":False,"config":cfg,"config_path":str(root/"config.json"),
                 "record_tables":tables,"failures":[],"execution_counts":cost,
                 "counting_note":"The two predictions are actually executed. Retained and capped dynamics are shared where the rules choose the same branch.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if root==BASE:
            entry["stress_result"] = read(root/"results.json")
        registry["entries"].append(entry)
        row = {k:0 for k in headers}
        row.update({k:entry[k] for k in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        row.update(runs_records=executions,suffix_execution_records=executions,derived_records=n)
        inventory.append(row)
        queries += cost["actual_new_queries"]
        value_checks += audit["value_checks"]
        gradient_checks += audit["gradient_checks"]
    registry["entries"].sort(key=lambda r:r["folder"])
    registry["as_of"] = "2026-11-13"
    validation.update({"date":"2026-11-13","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E46":len(historical),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":89,"new_run_breakdown":{"new_continuations":89,"constructed_contexts":51,"actual_new_queries":queries},
        "new_optimization_runs_note":"E46: 84 main and five engineering continuations from constructed launch states.",
        "new_audit_oracle_calls":{"value_checks":value_checks,"gradient_checks":gradient_checks},
        "new_execution_segments_with_complete_tables":89,"new_accepted_execution_segments":84,
        "new_formal_mechanism_execution_segments":84,"excluded_engineering_executions":5,
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered 89 continuations. Checked {links} links and {len(historical)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
