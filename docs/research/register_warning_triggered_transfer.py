"""Seal E29 continuations and preserve the frozen E28 source records."""

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
    "preflight": ("collision_warning_triggered_transfer_preflight", 6, 12, "frozen_checkpoint_engineering_preflight"),
    "formal": ("collision_warning_triggered_transfer", 240, 344, "frozen_E28_warning_intervention_pairs"),
}


def seal(root, tests):
    files = [Path(name) for name in (
        "scripts/run_warning_triggered_transfer.py", "scripts/audit_warning_triggered_transfer.py",
        "docs/research/register_warning_triggered_transfer.py", "docs/research/register_overlap_transfer.py",
        "docs/superpowers/specs/2026-11-11-warning-triggered-transfer-design.md",
        "docs/collision_warning_triggered_transfer_results.md", "docs/experiment_log.md",
    )]
    create(root / "verification.json", {
        "audit": read(root / "audit.json"), "test_run": tests,
        "audit_command": ["python3", "scripts/audit_warning_triggered_transfer.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    })
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as out:
        for f in files:
            out.write(f, str(f))
    members = {str(f): sha(f) for f in files}
    create(root / "postprocessing_sources.json", {
        "archive": str(archive), "archive_sha256": sha(archive), "files": members,
    })
    create(root / "raw_artifact_manifest.json", {"files": [
        artifact(p) for p in sorted(root.rglob("*")) if p.is_file()
    ]})
    return archive_check(archive, members)


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    historical = {e["folder"]: e for e in registry["entries"]}
    fresh = {str(Path("output") / folder) for folder, _, _, _ in STAGES.values()}
    assert not historical.keys() & fresh
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == historical.keys() | fresh
    artifacts = [a for e in historical.values() for a in e["top_level_artifacts"]]
    for a in artifacts:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(Path("output") / folder / "config.json") for folder, _, _, _ in STAGES.values()]
    assert configs[0]["source_sha256"] == configs[1]["source_sha256"]
    totals, audit_counts = Counter(), Counter()
    for (stage, (folder, contexts, expected, classification)), config in zip(STAGES.items(), configs):
        root = Path("output") / folder
        audit, counts = read(root / "audit.json"), read(root / "execution_counts.json")
        assert config["stage"] == stage and audit["passed"]
        assert counts["reused_prefixes"] == contexts and counts["suffix_executions"] == expected
        assert counts["new_prefix_executions"] == config["new_independent_seed_groups"] == 0
        assert sha(root / "source.zip") == config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", config["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests))
        tables = {}
        for name, role in (
            ("runs", "executed_continuation"), ("checkpoints", "reused_E28_checkpoint"),
            ("paired_outcomes", "derived_matched_branch_comparison"), ("already_deviated_cases", "derived_late_case_detail"),
        ):
            path = root / (name + ".json")
            tables[name] = {"path": str(path), "records": len(read(path)), "role": role}
        entry = {
            "phase_id": "E29", "folder": str(root), "classification": classification, "run_date": "2026-11-11",
            "config": config, "config_path": str(root / "config.json"), "run_rows_are_derived": False,
            "failures": audit["solver_failures"], "record_tables": tables, "execution_counts": counts,
            "counting_note": "Root runs are new paid continuations. Prefixes and seeds were executed in E28 and reused. Paired comparisons and the warning policy view are derived.",
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(x for x in p.rglob("*") if x.is_file()))
            },
        }
        if stage == "formal":
            result = read(root / "results.json")
            entry["warning_intervention_screen"] = {"passed": result["passed_screen"], "checks": result["success_checks"]}
        registry["entries"].append(entry)
        row = {k: 0 for k in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=expected, suffix_execution_records=expected, checkpoint_records=contexts,
                   derived_records=tables["paired_outcomes"]["records"], recorded_failures=len(entry["failures"]))
        inventory.append(row)
        totals.update(counts)
        audit_counts.update({k: audit[k] for k in ("value_checks", "gradient_checks")})
    links = check_links()
    assert links > 246
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = "2026-11-11"
    validation.update({
        "date": "2026-11-11", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E29": len(artifacts),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": totals["suffix_executions"],
        "new_optimization_runs_note": "E29: 12 engineering and 344 formal continuations from frozen E28 prefixes, no new seeds or prefix executions.",
        "new_run_breakdown": dict(totals), "new_audit_oracle_calls": dict(audit_counts),
        "new_execution_segments_with_complete_tables": 356, "new_accepted_execution_segments": 344,
        "new_formal_mechanism_execution_segments": 344, "excluded_engineering_executions": 12,
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered {totals['suffix_executions']} continuations and checked {links} links and {len(artifacts)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
