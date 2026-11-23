"""Seal and register E25 while retaining previous experiment records."""

import argparse
from collections import Counter
import csv
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote
import zipfile

from _source_paths import source_path
from finalize_initial_budget import create, sha
from refresh_experiment_inventory import read, write

DOCS = Path("docs")
OUTPUT = Path("output")
FOLDERS = ("collision_conditional_transfer_preflight", "collision_conditional_transfer")
RECORDS = {
    "runs": ("executed_control_trajectories", lambda data: len(data)),
    "contexts": ("shared_initial_value_contexts", lambda data: len(data)),
    "conditional": ("derived_choice_of_control_trajectory", lambda data: len(data)),
    "common_time": ("derived_common_time_comparisons", lambda data: len(data)),
    "summary": ("derived_policy_cells", lambda data: len(data["same_budget"]) + len(data["common_time"])),
    "physical_ledgers": ("derived_from_raw_trajectories", lambda data: len(data["runs"])),
    "development": ("derived_E24_threshold_search", lambda data: len(data["trials"])),
}


def artifact(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def seal(root, audit, config, counts, tests):
    create(root / "verification.json", {
        "date": "2026-11-10", "audit": audit, "test_run": tests, "execution_counts": counts,
        "audit_command": ["python3", "scripts/audit_conditional_transfer.py", str(root)],
        "audit_environment": {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        | {"PYTHONPATH": "src"},
        "unchanged_run_and_audit_source_between_stages": True,
        "development_sha256": config["development_sha256"],
    })
    paths = [
        Path("scripts/run_conditional_transfer.py"), Path("scripts/audit_conditional_transfer.py"),
        Path(__file__).resolve().relative_to(Path.cwd()),
        Path("docs/superpowers/specs/2026-11-10-conditional-transfer-design.md"),
        Path("docs/collision_conditional_transfer_results.md"), Path("docs/experiment_log.md"),
    ]
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, str(path))
    create(root / "postprocessing_sources.json", {
        "archive": str(archive), "archive_sha256": sha(archive),
        "files": {str(p): sha(p) for p in paths},
    })
    create(root / "raw_artifact_manifest.json", {"files": [
        artifact(p) for p in sorted(root.rglob("*")) if p.is_file()
    ]})


def register(tests):
    registry_path = DOCS / "experiment_registry.json"
    validation_path = DOCS / "recordkeeping_validation.json"
    inventory_path = DOCS / "experiment_inventory.csv"
    registry = read(registry_path)
    validation = read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    old_entries = {e["folder"]: e for e in registry["entries"]}
    new_folders = {str(OUTPUT / folder) for folder in FOLDERS}
    assert not (new_folders & old_entries.keys())
    assert {str(p) for p in OUTPUT.iterdir() if p.is_dir()} == old_entries.keys() | new_folders
    assert len(inventory) == len(old_entries)
    old_hashes = [a for e in old_entries.values() for a in e["top_level_artifacts"]]
    for a in old_hashes:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(OUTPUT / folder / "config.json") for folder in FOLDERS]
    assert configs[0]["source_sha256"] == configs[1]["source_sha256"]
    assert configs[0]["development_sha256"] == configs[1]["development_sha256"]
    for name, digest in configs[0]["source_sha256"].items():
        assert sha(source_path(name)) == digest, name
    totals = Counter()
    archive_checks = []
    for folder, config in zip(FOLDERS, configs):
        root = OUTPUT / folder
        audit = read(root / "audit.json")
        counts = read(root / "execution_counts.json")
        runs = read(root / "runs.json")
        is_formal = config["stage"] == "formal"
        assert audit["passed"] and len(runs) == (80 if is_formal else 4)
        assert counts["trajectory_executions"] == len(runs)
        tables = {
            name: {"path": str(path), "records": size(read(path)), "role": role}
            for name, (role, size) in RECORDS.items()
            if (path := root / (name + ".json")).exists() and (name != "development" or not is_formal)
        }
        failures = [{"run_id": r["run_id"], "seed": r["seed"], "condition": r["condition"],
                     "policy": r["policy"], "reason": r["stop_reason"]} for r in runs if r["failed"]]
        seal(root, audit, config, counts, tests)
        for archive_name, members in (
            ("source.zip", config["source_sha256"]),
            ("postprocessing_sources.zip", read(root / "postprocessing_sources.json")["files"]),
        ):
            archive_path = root / archive_name
            with zipfile.ZipFile(archive_path) as archive:
                for name, digest in members.items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            archive_checks.append({"archive": str(archive_path), "members_checked": len(members),
                                   "integrity_passed": True, "current_differences": []})
        entry = {
            "phase_id": "E25", "folder": str(root), "run_date": "2026-11-10",
            "classification": "held_out_paired_initialization_screen" if is_formal else "engineering_preflight_and_derived_threshold_search",
            "counting_note": "Two executed controls per context. The conditional and common-time rows reuse those trajectories. The same frozen development table is copied into the formal folder.",
            "config_path": str(root / "config.json"), "config": config, "record_tables": tables,
            "run_rows_are_derived": False, "failures": failures, "execution_counts": counts,
            "engineering_events": [str(p) for p in sorted(root.glob("engineering_attempts/*/failure.json"))],
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(x for x in p.rglob("*") if x.is_file()))
            },
        }
        if is_formal:
            entry["descriptive_screen"] = read(root / "screen.json")
        registry["entries"].append(entry)
        row = {k: 0 for k in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=len(runs), allocation_context_records=counts["contexts"],
                   derived_records=tables["conditional"]["records"] + tables.get("development", {}).get("records", 0),
                   recorded_failures=len(failures))
        inventory.append(row)
        totals.update(counts)

    links, missing = 0, []
    for document in [Path("README.md"), *DOCS.rglob("*.md")]:
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            target = link.strip().strip("<>")
            if target.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            links += 1
            if not (document.parent / unquote(target.split("#")[0])).exists():
                missing.append((str(document), target))
    assert not missing, missing
    registry["entries"].sort(key=lambda entry: entry["folder"])
    registry["as_of"] = "2026-11-10"
    validation["source_archive_checks"].extend(archive_checks)
    validation.update({
        "date": "2026-11-10", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E25": len(old_hashes),
        "threshold_candidates_evaluated_on_existing_trajectories": 7,
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": totals["trajectory_executions"],
        "new_optimization_runs_note": "E25: 80 held-out executions, four engineering executions. Conditional rows are derived and reuse the executed controls.",
        "new_run_breakdown": dict(totals), "new_execution_segments_with_complete_tables": totals["trajectory_executions"],
        "new_accepted_execution_segments": 80, "new_formal_mechanism_execution_segments": 80,
        "excluded_engineering_executions": 4, "latest_full_test_count": tests["passed"],
        "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered {totals['trajectory_executions']} executions. Verified {len(old_hashes)} historical artifacts and {links} document links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
