"""Seal E39 and register new continuations separately from reused controls."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_capped_reflection_transfer import BASE, SPEC


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    stages = {Path(str(BASE)+"_preflight"): "engineering_preflight", BASE: "retrospective_capped_transfer_comparison"}
    previous = {Path(e["folder"]) for e in registry["entries"]}
    assert not previous & stages.keys()
    assert {p for p in Path("output").iterdir() if p.is_dir()}==previous | stages.keys()
    history = [a for e in registry["entries"] for a in e["top_level_artifacts"]]
    for a in history:
        assert sha(Path(a["path"]))==a["sha256"],a["path"]
    links = check_links()
    assert links>338
    totals, audit_totals = Counter(), Counter()
    main_source = read(BASE/"config.json")["source_sha256"]
    for root, classification in stages.items():
        config, counts, audit = [read(root/(n+".json")) for n in ("config","execution_counts","audit")]
        assert config["phase"]=="E39" and config["source_sha256"]==main_source
        assert audit["passed"] and not audit["solver_failures"]
        assert counts["continuations"]==audit["continuations"]==(40 if root==BASE else 2)
        assert counts["reused_continuations"]==2*counts["continuations"] and counts["new_initial_states"]==0
        assert sha(root/"source.zip")==config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip",main_source))
        validation["source_archive_checks"].append(seal(root,tests,audit,"capped_reflection_transfer",str(SPEC),(
            str(Path(__file__).resolve().relative_to(Path.cwd())),)))
        tables = {}
        for name, role in (("contexts","reused_E38_checkpoints"),("runs","new_capped_continuations"),
                           ("paired_outcomes","derived_three_way_comparisons"),("already_deviated_cases","derived_recovery_states"),
                           ("physical_ledgers","new_and_reused_branch_ledgers")):
            path = root/(name+".json")
            tables[name] = {"path":str(path),"role":role,"records":len(read(path))}
        entry = {"phase_id":"E39","folder":str(root),"classification":classification,"run_date":"2026-11-12",
                 "run_rows_are_derived":False,"config":config,"config_path":str(root/"config.json"),
                 "record_tables":tables,"failures":[],"execution_counts":counts,
                 "counting_note":"Each comparison has two archived E38 branches and one new continuation. Only the latter counts as a new run.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if root==BASE:
            entry["capped_reflection_transfer_result"] = read(root/"results.json")
        registry["entries"].append(entry)
        row = {k:0 for k in headers}
        row.update({k:entry[k] for k in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        row.update(runs_records=counts["continuations"],suffix_execution_records=counts["continuations"],
                   derived_records=tables["paired_outcomes"]["records"])
        inventory.append(row)
        totals.update(counts)
        audit_totals.update({k:audit[k] for k in ("value_checks","gradient_checks")})
    registry["entries"].sort(key=lambda e:e["folder"])
    registry["as_of"] = "2026-11-12"
    validation.update({"date":"2026-11-12","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E39":len(history),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":totals["continuations"],"new_run_breakdown":dict(totals),
        "new_optimization_runs_note":"E39: new capped continuations only, with E38 controls reused.",
        "new_audit_oracle_calls":dict(audit_totals),"new_execution_segments_with_complete_tables":totals["continuations"],
        "new_accepted_execution_segments":40,"new_formal_mechanism_execution_segments":40,"excluded_engineering_executions":2,
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered {totals['continuations']} new continuations. Checked {links} links and {len(history)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
