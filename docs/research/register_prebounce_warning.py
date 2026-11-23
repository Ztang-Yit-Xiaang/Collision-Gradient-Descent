"""Seal E28 traces and register actual runs separately from predictions."""

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
    "preflight": ("collision_prebounce_warning_preflight", 6, "engineering_preflight"),
    "development": ("collision_prebounce_warning_development", 120, "warning_threshold_development"),
    "formal": ("collision_prebounce_warning", 240, "held_out_prebounce_warning"),
}


def seal(root, tests):
    files = [Path(name) for name in (
        "scripts/run_prebounce_warning.py", "scripts/audit_prebounce_warning.py", "docs/research/register_prebounce_warning.py",
        "docs/research/register_overlap_transfer.py", "docs/superpowers/specs/2026-11-11-prebounce-warning-design.md",
        "docs/collision_prebounce_warning_results.md", "docs/experiment_log.md",
    )]
    create(root / "verification.json", {
        "audit_command": ["python3", "scripts/audit_prebounce_warning.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "test_run": tests, "audit": read(root / "audit.json"), "same_runtime_source_all_stages": True,
    })
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as out:
        for path in files:
            out.write(path, str(path))
    members = {str(p): sha(p) for p in files}
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
    old = {e["folder"]: e for e in registry["entries"]}
    fresh = {str(Path("output") / folder) for folder, _, _ in STAGES.values()}
    assert not old.keys() & fresh
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == old.keys() | fresh
    history = [a for e in old.values() for a in e["top_level_artifacts"]]
    for a in history:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(Path("output") / folder / "config.json") for folder, _, _ in STAGES.values()]
    assert all(cfg["source_sha256"] == configs[0]["source_sha256"] for cfg in configs)
    totals, audit_totals = Counter(), Counter()
    for (stage, (folder, expected, classification)), cfg in zip(STAGES.items(), configs):
        root = Path("output") / folder
        audit, counts = read(root / "audit.json"), read(root / "execution_counts.json")
        assert cfg["stage"] == stage and audit["passed"]
        assert audit["trajectories"] == counts["trajectory_executions"] == expected
        assert sha(root / "source.zip") == cfg["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", cfg["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests))
        tables = {
            name: {"path": str(path), "records": len(read(path)), "role": role}
            for name, role in (
                ("runs", "executed_no_intervention_trajectory"), ("contexts", "initial_conditions"),
                ("labeled_observations", "derived_prebounce_labels_and_scores"),
            )
            if (path := root / (name + ".json")).exists()
        }
        entry = {
            "phase_id": "E28", "folder": str(root), "classification": classification, "run_date": "2026-11-11",
            "run_rows_are_derived": False, "failures": audit["solver_failures"], "record_tables": tables,
            "config": cfg, "config_path": str(root / "config.json"), "execution_counts": counts,
            "counting_note": "One executed trajectory per seed and kinetic level. All five warning rules reuse it; the trajectory label is derived.",
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(x for x in p.rglob("*") if x.is_file()))
            },
        }
        if stage == "formal":
            outcome = read(root / "results.json")
            entry["warning_screen"] = {"rules": outcome["screen"], "passed": outcome["any_physical_rule_passed"]}
            tables["predictions"] = {"path": str(root / "results.json"), "records": len(outcome["predictions"]),
                                     "role": "derived_comparison_of_five_warning_rules"}
        registry["entries"].append(entry)
        row = {key: 0 for key in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=expected, derived_records=tables["labeled_observations"]["records"],
                   recorded_failures=len(entry["failures"]))
        inventory.append(row)
        totals.update(counts)
        audit_totals.update({key: audit[key] for key in ("value_checks", "gradient_checks")})
    links = check_links()
    assert links > 236
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = "2026-11-11"
    validation.update({
        "date": "2026-11-11", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E28": len(history),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": 366,
        "new_optimization_runs_note": "E28: 6 preflight, 120 development and 240 held-out short trajectories. Five predictors share them.",
        "new_run_breakdown": dict(totals), "new_audit_oracle_calls": dict(audit_totals),
        "new_execution_segments_with_complete_tables": 366, "new_accepted_execution_segments": 360,
        "new_formal_mechanism_execution_segments": 240, "excluded_engineering_executions": 6,
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda r: r["folder"]))
    print(f"Registered 366 trajectories and {links} links; verified {len(history)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
