"""Register E30 and verify that previous experiment artifacts are unchanged."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import zipfile

from finalize_initial_budget import create, sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_overlap_transfer import archive_check, check_links

STAGES = {
    "preflight": ("collision_postbounce_warning_transfer_preflight", 4, "postbounce_forecast_preflight"),
    "formal": ("collision_postbounce_warning_transfer", 136, "frozen_E29_postbounce_intervention_pairs"),
}


def seal(root, tests):
    files = [Path(name) for name in (
        "scripts/run_postbounce_warning_transfer.py", "scripts/audit_postbounce_warning_transfer.py",
        "docs/research/register_postbounce_warning_transfer.py", "docs/research/register_overlap_transfer.py",
        "docs/superpowers/specs/2026-11-11-postbounce-warning-transfer-design.md",
        "docs/collision_postbounce_warning_transfer_results.md", "docs/experiment_log.md",
    )]
    create(root / "verification.json", {
        "audit": read(root / "audit.json"), "test_run": tests,
        "audit_command": ["python3", "scripts/audit_postbounce_warning_transfer.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    })
    path = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, str(file))
    members = {str(file): sha(file) for file in files}
    create(root / "postprocessing_sources.json", {
        "archive": str(path), "archive_sha256": sha(path), "files": members,
    })
    create(root / "raw_artifact_manifest.json", {"files": [
        artifact(file) for file in sorted(root.rglob("*")) if file.is_file()
    ]})
    return archive_check(path, members)


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    old = {entry["folder"]: entry for entry in registry["entries"]}
    fresh = {str(Path("output") / name) for name, _, _ in STAGES.values()}
    assert not old.keys() & fresh
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == old.keys() | fresh
    historical = [a for entry in old.values() for a in entry["top_level_artifacts"]]
    for a in historical:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(Path("output") / name / "config.json") for name, _, _ in STAGES.values()]
    assert configs[0]["source_sha256"] == configs[1]["source_sha256"]
    totals, audit_totals = Counter(), Counter()
    for (stage, (name, n, classification)), config in zip(STAGES.items(), configs):
        root = Path("output") / name
        audit, counts = read(root / "audit.json"), read(root / "execution_counts.json")
        assert config["stage"] == stage and audit["passed"]
        assert counts["reused_prefixes"] == n and counts["suffix_executions"] == 2 * n
        assert counts["new_prefix_executions"] == config["new_independent_seed_groups"] == 0
        assert sha(root / "source.zip") == config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", config["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests))
        tables = {}
        for table, role in (
            ("runs", "executed_continuation"), ("checkpoints", "reused_E29_checkpoint_and_paid_forecast"),
            ("paired_outcomes", "derived_matched_branch_comparison"), ("already_deviated_cases", "derived_recovery_cases"),
        ):
            path = root / (table + ".json")
            tables[table] = {"path": str(path), "records": len(read(path)), "role": role}
        entry = {
            "phase_id": "E30", "folder": str(root), "classification": classification, "run_date": "2026-11-11",
            "config": config, "config_path": str(root / "config.json"), "run_rows_are_derived": False,
            "failures": audit["solver_failures"], "record_tables": tables, "execution_counts": counts,
            "counting_note": "Root runs are actual continuations. Forecasts are actual paid queries or cached math, reused by both continuations. Parent states and seed groups are historical.",
            "top_level_artifacts": [artifact(file) for file in sorted(root.iterdir()) if file.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(f for f in p.rglob("*") if f.is_file()))
            },
        }
        if stage == "formal":
            result = read(root / "results.json")
            entry["postbounce_screen"] = {"passed": result["passed_screen"], "checks": result["checks"]}
        registry["entries"].append(entry)
        row = {key: 0 for key in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=2 * n, suffix_execution_records=2 * n, checkpoint_records=n,
                   derived_records=n, recorded_failures=len(entry["failures"]))
        inventory.append(row)
        totals.update(counts)
        audit_totals.update({k: audit[k] for k in ("value_checks", "gradient_checks")})
    links = check_links()
    assert links > 255
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = "2026-11-11"
    validation.update({
        "date": "2026-11-11", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E30": len(historical),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": totals["suffix_executions"],
        "new_optimization_runs_note": "E30: eight engineering and 272 formal continuations. Reused prefixes, no new seed groups.",
        "new_run_breakdown": dict(totals), "new_audit_oracle_calls": dict(audit_totals),
        "new_execution_segments_with_complete_tables": 280, "new_accepted_execution_segments": 272,
        "new_formal_mechanism_execution_segments": 272, "excluded_engineering_executions": 8,
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered {totals['suffix_executions']} continuations and checked {links} links and {len(historical)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
