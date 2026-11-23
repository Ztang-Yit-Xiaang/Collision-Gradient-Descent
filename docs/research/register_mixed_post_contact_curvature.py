"""Seal the E49 mixed comparison and append its actual execution rows."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_mixed_post_contact_curvature import BASE, SPEC


def register(tests, *, phase="E49", base=BASE, spec=SPEC, stem="mixed_post_contact_curvature",
             result_key="mixed_post_contact_result", extra_sources=()):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    stages = [(Path(str(base)+"_preflight"), "engineering_mixed_preflight"),
              (base, "fresh_seed_stratified_mixed_mechanism_comparison")]
    known = {Path(e["folder"]) for e in registry["entries"]}
    assert {p for p in Path("output").iterdir() if p.is_dir()} == known | {root for root, _ in stages}
    assert not known & {root for root, _ in stages}
    historical = [a for e in registry["entries"] for a in e["top_level_artifacts"]]
    for a in historical:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    links = check_links()
    source = read(base/"config.json")["source_sha256"]
    totals, audits = Counter(), Counter()
    new_artifacts = 0
    for root, classification in stages:
        cfg, cost, audit = (read(root/(n+".json")) for n in ("config", "execution_counts", "audit"))
        assert cfg["phase"] == phase and cfg["source_sha256"] == source
        assert audit["passed"] and not audit["solver_failures"]
        assert len(cfg["jobs"]) == cost["contexts"] == audit["contexts"]
        assert cost["continuations"] == audit["continuations"]
        assert sha(root/"source.zip") == cfg["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip", source))
        validation["source_archive_checks"].append(seal(root, tests, audit, stem, str(spec), (
            str(Path(__file__).resolve().relative_to(Path.cwd())), "scripts/run_post_contact_positive_stress.py",
            "scripts/run_post_contact_curvature_screen.py", *extra_sources)))
        tables = {}
        for name, role in (("contexts", "new_paired_states_and_paid_forecasts"),
                           ("runs", "executed_unique_continuations"),
                           ("paired_outcomes", "derived_policy_comparisons"),
                           ("physical_ledgers", "derived_work_impulse_and_energy_ledgers")):
            path = root/(name+".json")
            tables[name] = {"path": str(path), "role": role, "records": len(read(path))}
        entry = {"phase_id": phase, "folder": str(root), "classification": classification, "run_date": "2026-11-15",
                 "run_rows_are_derived": False, "config": cfg, "config_path": str(root/"config.json"),
                 "record_tables": tables, "failures": [], "execution_counts": cost,
                 "counting_note": "Each policy purchases its own forecasts. Each newly executed dynamical branch is counted once, including keep controls. A same-choice pair shares the branch.",
                 "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts": {p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir()
                     if (files := [f for f in p.rglob("*") if f.is_file()])}}
        if root == base:
            entry[result_key] = read(root/"results.json")
        registry["entries"].append(entry)
        new_artifacts += len(entry["top_level_artifacts"])
        row = {k: 0 for k in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=cost["continuations"], suffix_execution_records=cost["continuations"],
                   derived_records=cost["contexts"])
        inventory.append(row)
        totals.update(cost)
        audits.update({k: audit[k] for k in ("value_checks", "gradient_checks")})
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = "2026-11-15"
    validation.update({"date": "2026-11-15", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [], "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": len(historical) + new_artifacts,
        f"historical_output_artifact_hashes_unchanged_for_{phase}": len(historical),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": totals["continuations"], "new_run_breakdown": dict(totals),
        "new_optimization_runs_note": f"{phase}: fresh-state keep controls and selected capped continuations, counted once.",
        "new_audit_oracle_calls": dict(audits),
        "new_execution_segments_with_complete_tables": totals["continuations"],
        "new_accepted_execution_segments": totals["continuations"],
        "new_formal_mechanism_execution_segments": read(base/"execution_counts.json")["continuations"],
        "excluded_engineering_executions": 0, "engineering_audit_failures": 0,
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"]})
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda r: r["folder"]))
    print(f"Registered {dict(totals)}. Checked {links} links and {len(historical)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
        "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
