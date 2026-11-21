"""Register E43's fresh baselines, long extensions, and capped continuations."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_third_reflection_replication import BASE, SPEC


def register(tests, *, base_dir=BASE, spec=SPEC, phase="E43", stem="third_reflection_replication",
             result_file="replication_results.json", min_links=384, extra_sources=()):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry,validation = read(registry_path),read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers,inventory = reader.fieldnames,list(reader)
    stages = [(Path(str(base_dir)+"_preflight"),"engineering_preflight",6),
              (base_dir,"fresh_seed_"+stem,480)]
    folders = {root for root,_,_ in stages}
    previous = {Path(r["folder"]) for r in registry["entries"]}
    assert not folders & previous and {p for p in Path("output").iterdir() if p.is_dir()}==folders|previous
    history = [a for r in registry["entries"] for a in r["top_level_artifacts"]]
    assert all(sha(Path(a["path"]))==a["sha256"] for a in history)
    links = check_links()
    assert links>min_links
    source = read(base_dir/"config.json")["source_sha256"]
    total, audits, segments = Counter(),Counter(),{}
    for root,classification,n in stages:
        cfg = read(root/"config.json")
        assert cfg["phase"]==phase and cfg["source_sha256"]==source and sha(root/"source.zip")==cfg["source_archive_sha256"]
        base,second,extension,third = [read(p/"execution_counts.json") for p in (
            root,root/"second_reflection",root/"extension",root/"third_reflection")]
        checks = {name:read(folder/"audit.json") for name,folder in (
            ("base",root),("second_reflection",root/"second_reflection"),("long_extension_and_third_reflection",root/"third_reflection"))}
        assert all(a["passed"] and not a["solver_failures"] for a in checks.values())
        assert base["contexts"]==second["contexts"]==extension["continuations"]==third["contexts"]==n
        validation["source_archive_checks"].append(archive_check(root/"source.zip",source))
        validation["source_archive_checks"].append(seal(root,tests,{"passed":True,**checks},stem,str(spec),(
            str(Path(__file__).resolve().relative_to(Path.cwd())),"run_third_reflection_replication_branches.py",
            "audit_third_reflection_guard.py",*extra_sources)))
        tables = {}
        for relative,role in (("contexts","fresh_initial_states"),("prefixes","executed_E37_prefixes"),("runs","executed_E37_continuations"),
                ("uninterrupted","engineering_controls"),("second_reflection/runs","executed_second_reflection_transfers"),
                ("extension/runs","executed_common_long_extensions"),("third_reflection/runs","executed_third_reflection_transfers"),
                ("third_reflection/paired_outcomes","derived_policy_comparisons"),
                ("third_reflection/already_deviated_cases","derived_recovery_states")):
            path = root/(relative+".json")
            tables[relative.replace("/","_")] = {"path":str(path),"role":role,"records":len(read(path))}
        cost = {"fresh_initial_states":n,"base_queries":base["actual_queries_including_setup"],
                "second_reflection_queries":second["actual_new_solver_queries"],"extension_queries":extension["actual_new_queries"],
                "third_reflection_queries":third["actual_new_queries"],"base_continuations":base["continuations"],
                "second_reflection_continuations":second["continuations"],"extension_continuations":extension["continuations"],
                "third_reflection_continuations":third["continuations"]}
        cost["actual_new_queries"] = sum(cost[k] for k in ("base_queries","second_reflection_queries","extension_queries","third_reflection_queries"))
        total.update(cost)
        for check in checks.values():
            audits.update({k:check[k] for k in ("value_checks","gradient_checks")})
        entry = {"phase_id":phase,"folder":str(root),"classification":classification,"run_date":"2026-11-12",
                 "run_rows_are_derived":False,"config":cfg,"config_path":str(root/"config.json"),
                 "record_tables":tables,"failures":[],"execution_counts":cost,
                 "counting_note":"The E40 and added policies share a path until the third decision. Extension and transfer traces count separately.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if root==base_dir:
            entry["replication_result"] = read(root/"third_reflection"/result_file)
        registry["entries"].append(entry)
        row = {k:0 for k in headers}
        row.update({k:entry[k] for k in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        suffixes = sum(cost[k] for k in ("base_continuations","second_reflection_continuations","extension_continuations","third_reflection_continuations"))
        row.update(runs_records=base["continuations"],suffix_execution_records=suffixes,derived_records=n,
                   source_execution_records=base["prefix_executions"],contact_context_records=base["exact_contacts"],
                   uninterrupted_replication_records=base["uninterrupted_preflight_executions"])
        inventory.append(row)
        segments[classification] = suffixes+sum(base[k] for k in ("prefix_executions","exact_contacts","uninterrupted_preflight_executions"))
    registry["entries"].sort(key=lambda r:r["folder"])
    registry["as_of"] = "2026-11-12"
    validation.update({"date":"2026-11-12","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        f"historical_output_artifact_hashes_unchanged_for_{phase}":len(history),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":sum(segments.values()),"new_run_breakdown":dict(total),
        "new_optimization_runs_note":f"{phase}: new baseline segments, long extensions, and capped continuations.",
        "new_audit_oracle_calls":dict(audits),"new_execution_segments_with_complete_tables":sum(segments.values()),
        "new_accepted_execution_segments":segments[stages[1][1]],"new_formal_mechanism_execution_segments":segments[stages[1][1]],
        "excluded_engineering_executions":segments["engineering_preflight"],
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered {total['fresh_initial_states']} fresh contexts and {sum(segments.values())} segments. Checked {links} links and {len(history)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
