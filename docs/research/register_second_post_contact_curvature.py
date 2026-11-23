"""Seal E48 and register the two engineering executions and the matched study."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links
from run_second_post_contact_curvature import BASE, SPEC


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    initial = Path(str(BASE) + "_preflight")
    verified = Path(str(BASE) + "_preflight_verified")
    stages = [(initial, "engineering_preflight_initial_audit_failed", 3, 1),
              (verified, "engineering_preflight_verified_lock", 3, 1),
              (BASE, "paired_stress_extended_forecast_and_new_continuations", 48, 12)]
    old = {Path(r["folder"]) for r in registry["entries"]}
    folders = {r[0] for r in stages}
    assert not old & folders
    assert {p for p in Path("output").iterdir() if p.is_dir()} == old | folders
    historical = [a for r in registry["entries"] for a in r["top_level_artifacts"]]
    for a in historical:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    links = check_links()
    source = read(BASE / "config.json")["source_sha256"]
    totals, audit_totals = Counter(), Counter()
    for root, classification, n, continuations in stages:
        cfg, cost, audit = [read(root / (k + ".json")) for k in ("config", "execution_counts", "audit")]
        assert cfg["phase"] == "E48" and audit["passed"] and not audit["solver_failures"]
        assert root == initial or cfg["source_sha256"] == source
        assert cost["contexts"] == audit["contexts"] == n
        assert cost["continuations"] == audit["continuations"] == continuations
        assert cost["actual_new_queries"] == cost["actual_new_continuation_queries"] + cost["actual_new_forecast_queries"]
        assert sha(root / "source.zip") == cfg["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", cfg["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests, audit, "second_post_contact_curvature",
            str(SPEC), (str(Path(__file__).resolve().relative_to(Path.cwd())), "scripts/run_post_contact_curvature_screen.py")))
        tables = {}
        for name, role in (("decisions", "new_paid_forecasts"), ("runs", "new_continuations"),
                          ("paired_outcomes", "derived_policy_comparisons"),
                          ("second_saved_cases", "derived_later_crossings")):
            path = root / (name + ".json")
            tables[name] = {"path": str(path), "role": role, "records": len(read(path))}
        failures = [read(root / "initial_audit_failure.json")] if root == initial else []
        entry = {"phase_id": "E48", "folder": str(root), "classification": classification, "run_date": "2026-11-15",
                 "run_rows_are_derived": False, "config": cfg, "config_path": str(root / "config.json"),
                 "record_tables": tables, "failures": failures, "execution_counts": cost,
                 "counting_note": "Every forecast was newly paid. Runs are newly executed capped continuations; unchanged selections reuse E46. Preflights repeat the same three states and add no independent contexts.",
                 "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts": {p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir()
                     if (files := [f for f in p.rglob("*") if f.is_file()])}}
        if root == BASE:
            entry["second_saved_curvature_result"] = read(root / "results.json")
        registry["entries"].append(entry)
        row = {k: 0 for k in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=continuations, suffix_execution_records=continuations,
                   derived_records=n, recorded_failures=len(failures))
        inventory.append(row)
        totals.update(cost)
        audit_totals.update({k: audit[k] for k in ("value_checks", "gradient_checks")})
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = "2026-11-15"
    validation.update({"date": "2026-11-15", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [], "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": len(historical) + sum(len(e["top_level_artifacts"]) for e in registry["entries"]
            if e["phase_id"] == "E48"),
        "historical_output_artifact_hashes_unchanged_for_E48": len(historical),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": totals["continuations"], "new_run_breakdown": dict(totals),
        "new_optimization_runs_note": "E48: 12 formal continuations and two engineering continuations; no new seeds.",
        "new_audit_oracle_calls": dict(audit_totals),
        "new_execution_segments_with_complete_tables": totals["continuations"],
        "new_accepted_execution_segments": 13, "new_formal_mechanism_execution_segments": 12,
        "excluded_engineering_executions": 1, "engineering_audit_failures": 1,
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
