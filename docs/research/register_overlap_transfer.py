"""Register the three E26 stages and seal their auditable artifacts."""

import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

from _source_paths import source_path
from finalize_initial_budget import create, sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact

DOCS = Path("docs")
STAGES = {
    "preflight": ("collision_overlap_transfer_preflight", 8, "engineering_preflight"),
    "development": ("collision_overlap_transfer_development", 96, "development_feature_direction"),
    "formal": ("collision_overlap_transfer", 160, "held_out_loss_overlap_pressure_test"),
}


def archive_check(path, members):
    with zipfile.ZipFile(path) as archive:
        for name, digest in members.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    return {"archive": str(path), "members_checked": len(members), "integrity_passed": True,
            "current_differences": [name for name, digest in members.items() if not (current := source_path(name)).is_file() or sha(current) != digest]}


def seal(root, tests):
    paths = [
        Path("scripts/run_overlap_transfer.py"), Path("scripts/audit_overlap_transfer.py"),
        Path(__file__).resolve().relative_to(Path.cwd()), Path("scripts/register_conditional_transfer.py"),
        Path("scripts/audit_conditional_transfer.py"), Path("scripts/finalize_initial_budget.py"),
        Path("docs/superpowers/specs/2026-11-10-overlap-transfer-design.md"),
        Path("docs/collision_overlap_transfer_results.md"), Path("docs/experiment_log.md"),
    ]
    create(root / "verification.json", {
        "audit_command": ["python3", "scripts/audit_overlap_transfer.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "test_run": tests, "audit": read(root / "audit.json"),
        "all_stages_same_launch_source": True,
    })
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            output.write(path, str(path))
    members = {str(path): sha(path) for path in paths}
    create(root / "postprocessing_sources.json", {
        "archive": str(archive), "archive_sha256": sha(archive), "files": members,
    })
    create(root / "raw_artifact_manifest.json", {"files": [
        artifact(path) for path in sorted(root.rglob("*")) if path.is_file()
    ]})
    return archive_check(archive, members)


def check_links():
    checked = 0
    for doc in [Path("README.md"), *DOCS.rglob("*.md")]:
        for ref in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            target = ref.strip().strip("<>")
            if target.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            checked += 1
            assert (doc.parent / unquote(target.split("#")[0])).exists(), (doc, target)
    return checked


def register(tests):
    registry_path = DOCS / "experiment_registry.json"
    inventory_path = DOCS / "experiment_inventory.csv"
    validation_path = DOCS / "recordkeeping_validation.json"
    registry = read(registry_path)
    validation = read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    old = {e["folder"]: e for e in registry["entries"]}
    new = {str(Path("output") / folder) for folder, _, _ in STAGES.values()}
    assert not old.keys() & new
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == old.keys() | new
    history = [a for entry in old.values() for a in entry["top_level_artifacts"]]
    for item in history:
        assert sha(Path(item["path"])) == item["sha256"], item["path"]
    configs = [read(Path("output") / folder / "config.json") for folder, _, _ in STAGES.values()]
    assert all(cfg["source_sha256"] == configs[0]["source_sha256"] for cfg in configs)
    totals, audit_costs = Counter(), Counter()
    for (stage, (folder, expected, classification)), config in zip(STAGES.items(), configs):
        root = Path("output") / folder
        audit = read(root / "audit.json")
        counts = read(root / "execution_counts.json")
        assert config["stage"] == stage and audit["passed"] and counts["trajectory_executions"] == expected
        assert sha(root / "source.zip") == config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", config["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests))
        tables = {}
        for table, role in (
            ("runs", "executed_trajectory"), ("contexts", "shared_paid_context"),
            ("conditional", "derived_choice_of_executed_trajectory"), ("common_time", "derived_time_matched"),
            ("features", "derived_physical_prefix"), ("physical_ledgers", "derived_raw_physical_accounting"),
        ):
            path = root / (table + ".json")
            data = read(path)
            tables[table] = {"path": str(path), "records": len(data["runs"] if table == "physical_ledgers" else data),
                             "role": role}
        failures = [{"run_id": r["run_id"], "reason": r["stop_reason"]} for r in read(root / "runs.json") if r["failed"]]
        entry = {
            "phase_id": "E26", "folder": str(root), "classification": classification, "run_date": "2026-11-10",
            "run_rows_are_derived": False, "failures": failures, "execution_counts": counts,
            "config": config, "config_path": str(root / "config.json"), "record_tables": tables,
            "counting_note": "One keep and one transfer execution per context. E25 policy choices, time truncations, and prefix features reuse raw trajectories.",
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(x for x in p.rglob("*") if x.is_file()))
            },
        }
        if stage == "formal":
            entry["paired_actions"] = read(root / "summary.json")["paired_actions"]
            entry["heldout_feature_results"] = read(root / "heldout_feature_results.json")
        registry["entries"].append(entry)
        row = {key: 0 for key in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=expected, allocation_context_records=counts["contexts"],
                   derived_records=tables["conditional"]["records"], recorded_failures=len(failures))
        inventory.append(row)
        totals.update(counts)
        audit_costs.update({key: audit[key] for key in ("posthoc_value_calls", "posthoc_gradient_calls")})
    links = check_links()
    registry["entries"].sort(key=lambda entry: entry["folder"])
    registry["as_of"] = "2026-11-10"
    validation.update({
        "date": "2026-11-10", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E26": len(history),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": totals["trajectory_executions"],
        "new_optimization_runs_note": "E26: 8 preflight, 96 development, 160 held-out trajectories. E25 policy results and prefix analyses are derived.",
        "new_run_breakdown": dict(totals), "new_audit_oracle_calls": dict(audit_costs),
        "new_execution_segments_with_complete_tables": totals["trajectory_executions"],
        "new_accepted_execution_segments": 256, "new_formal_mechanism_execution_segments": 160,
        "excluded_engineering_executions": 8, "latest_full_test_count": tests["passed"],
        "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered {totals['trajectory_executions']} trajectories. Verified {len(history)} historical artifacts and {links} links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
