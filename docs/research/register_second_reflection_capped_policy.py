"""Seal E40 and count baseline, added branches, and engineering work separately."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_second_reflection_capped_policy import BASE, SPEC


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    stages = {Path(str(BASE)+"_preflight"): "engineering_preflight", BASE: "fresh_seed_second_reflection_capped_policy"}
    previous = {Path(e["folder"]) for e in registry["entries"]}
    assert not previous & stages.keys()
    assert {p for p in Path("output").iterdir() if p.is_dir()}==previous | stages.keys()
    history = [a for e in registry["entries"] for a in e["top_level_artifacts"]]
    for a in history:
        assert sha(Path(a["path"]))==a["sha256"],a["path"]
    links = check_links()
    assert links>348
    totals, audit_totals = Counter(), Counter()
    segments = Counter()
    main_source = read(BASE/"config.json")["source_sha256"]
    for root, classification in stages.items():
        config, counts, baseline_audit = [read(root/(n+".json")) for n in ("config","execution_counts","audit")]
        added = root/"second_reflection"
        costs, added_audit = [read(added/(n+".json")) for n in ("execution_counts","audit")]
        probe = read(root/"engineering_probe"/"audit.json") if root!=BASE else None
        assert config["phase"]=="E40" and config["source_sha256"]==main_source
        assert baseline_audit["passed"] and added_audit["passed"]
        assert not baseline_audit["solver_failures"] and not added_audit["solver_failures"]
        assert counts["contexts"]==costs["contexts"]==added_audit["contexts"]==(480 if root==BASE else 6)
        assert counts["continuations"]==baseline_audit["continuations"] and costs["continuations"]==added_audit["continuations"]
        assert sha(root/"source.zip")==config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip",main_source))
        audit = {"passed":True,"baseline":baseline_audit,"second_reflection":added_audit,"engineering_probe":probe}
        extra = (str(Path(__file__).resolve().relative_to(Path.cwd())),"run_second_reflection_branches.py",
                 "docs/research/e40_engineering_probe.py")
        validation["source_archive_checks"].append(seal(root,tests,audit,"second_reflection_capped_policy",str(SPEC),extra))
        tables = {}
        for folder, name, role in ((root,"contexts","fresh_initial_states"),(root,"prefixes","executed_baseline_prefixes"),
                                   (root,"runs","executed_E37_tree_continuations"),(root,"uninterrupted","engineering_direct_controls"),
                                   (added,"contexts","new_second_reflection_decisions"),(added,"runs","new_capped_continuations"),
                                   (added,"paired_outcomes","derived_policy_comparisons"),(added,"already_deviated_cases","derived_recovery_states")):
            path = folder/(name+".json")
            table = name if folder==root else "second_reflection_"+name
            tables[table] = {"path":str(path),"role":role,"records":len(read(path))}
        if probe:
            path = root/"engineering_probe"/"runs.json"
            tables["engineering_probe_runs"] = {"path":str(path),"role":"known_positive_preflight","records":len(read(path))}
        entry = {"phase_id":"E40","folder":str(root),"classification":classification,"run_date":"2026-11-12",
                 "run_rows_are_derived":False,"config":config,"config_path":str(root/"config.json"),
                 "record_tables":tables,"failures":[],"execution_counts":counts,"second_reflection_execution_counts":costs,
                 "engineering_probe_execution_counts":probe,
                 "counting_note":"New baseline tree and capped continuation executions are separate. Paired outcomes reuse a common path up to the decision.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if root==BASE:
            entry["second_reflection_capped_policy_result"] = read(added/"results.json")
        registry["entries"].append(entry)
        row = {k:0 for k in headers}
        row.update({k:entry[k] for k in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        row.update(runs_records=counts["continuations"],suffix_execution_records=counts["continuations"]+costs["continuations"]+int(bool(probe)),
                   source_execution_records=counts["prefix_executions"],contact_context_records=counts["exact_contacts"],
                   uninterrupted_replication_records=counts["uninterrupted_preflight_executions"],
                   derived_records=tables["second_reflection_paired_outcomes"]["records"])
        inventory.append(row)
        segments[classification] = sum(counts[k] for k in ("continuations","prefix_executions","exact_contacts","uninterrupted_preflight_executions"))+costs["continuations"]+int(bool(probe))
        totals.update({"fresh_initial_states":counts["contexts"],"baseline_continuations":counts["continuations"],
                       "baseline_prefixes":counts["prefix_executions"],"baseline_first_reflections":counts["exact_contacts"],
                       "uninterrupted_preflight_executions":counts["uninterrupted_preflight_executions"],
                       "baseline_queries_including_setup":counts["actual_queries_including_setup"],
                       "new_second_reflection_continuations":costs["continuations"],
                       "new_second_reflection_queries":costs["actual_new_solver_queries"],
                       "engineering_probe_queries":probe["actual_new_solver_queries"] if probe else 0,
                       "engineering_probe_continuations":int(bool(probe))})
        for part in (baseline_audit,added_audit,probe or {}):
            audit_totals.update({k:part.get(k,0) for k in ("value_checks","gradient_checks")})
    totals["all_actual_queries_including_setup"] = sum(totals[k] for k in (
        "baseline_queries_including_setup","new_second_reflection_queries","engineering_probe_queries"))
    registry["entries"].sort(key=lambda e:e["folder"])
    registry["as_of"] = "2026-11-12"
    validation.update({"date":"2026-11-12","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E40":len(history),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":sum(segments.values()),"new_run_breakdown":dict(totals),
        "new_optimization_runs_note":"E40: E37 tree segments, capped continuations, and separately charged engineering probes.",
        "new_audit_oracle_calls":dict(audit_totals),"new_execution_segments_with_complete_tables":sum(segments.values()),
        "new_accepted_execution_segments":segments[stages[BASE]],"new_formal_mechanism_execution_segments":segments[stages[BASE]],
        "excluded_engineering_executions":segments["engineering_preflight"],
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered {totals['fresh_initial_states']} fresh initial states. Checked {links} links and {len(history)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
