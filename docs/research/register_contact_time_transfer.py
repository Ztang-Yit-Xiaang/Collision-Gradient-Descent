"""Seal the E31 fractional-step replays and append them to the inventory."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import zipfile

from finalize_initial_budget import create, sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_overlap_transfer import archive_check, check_links

BASE = "collision_contact_time_transfer"
STAGES = {
    "preflight": (BASE + "_preflight", 4, "fractional_step_engineering_preflight"),
    "formal": (BASE, 10, "frozen_E30_exact_contact_pairs"),
}


def seal(root, tests, audit):
    paths = [Path(n) for n in (
        "run_contact_time_transfer.py", "audit_contact_time_transfer.py", "docs/research/register_contact_time_transfer.py",
        "docs/research/register_overlap_transfer.py", "docs/superpowers/specs/2026-11-11-contact-time-transfer-design.md",
        "docs/collision_contact_time_transfer_results.md", "docs/experiment_log.md",
    )]
    create(root / "verification.json", {
        "audit": audit, "test_run": tests, "audit_command": ["python3", "audit_contact_time_transfer.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    })
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        for path in paths:
            stream.write(path, str(path))
    members = {str(path): sha(path) for path in paths}
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
    fresh = {str(Path("output") / n) for n, _, _ in STAGES.values()}
    assert not old.keys() & fresh
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == old.keys() | fresh
    history = [a for e in old.values() for a in e["top_level_artifacts"]]
    for a in history:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(Path("output") / n / "config.json") for n, _, _ in STAGES.values()]
    assert configs[0]["source_sha256"] == configs[1]["source_sha256"]
    totals, audit_totals = Counter(), Counter()
    for (stage, (folder, n, classification)), config in zip(STAGES.items(), configs):
        root = Path("output") / folder
        audit, counts = read(root / "audit.json"), read(root / "execution_counts.json")
        assert config["stage"] == stage and audit["passed"]
        assert counts["reused_prefixes"] == counts["new_event_reconstructions"] == n
        assert counts["suffix_executions"] == 2 * n and config["new_independent_seed_groups"] == 0
        assert sha(root / "source.zip") == config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", config["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests, audit))
        tables = {}
        for name, role in (
            ("runs", "executed_fraction_and_tail_continuation"), ("checkpoints", "executed_shared_contact_reconstruction"),
            ("paired_outcomes", "derived_matched_branch_comparison"), ("contact_already_deviated_cases", "derived_recovery_cases"),
        ):
            path = root / (name + ".json")
            tables[name] = {"path": str(path), "records": len(read(path)), "role": role}
        entry = {
            "phase_id": "E31", "folder": str(root), "classification": classification, "run_date": "2026-11-11",
            "config": config, "config_path": str(root / "config.json"), "run_rows_are_derived": False,
            "failures": audit["solver_failures"], "record_tables": tables, "execution_counts": counts,
            "counting_note": "One shared paid exact-contact reconstruction per historical state. Root rows are continuations, each including a fractional step and its tail.",
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(f for f in p.rglob("*") if f.is_file()))
            },
        }
        if stage == "formal":
            entry["contact_time_screen"] = read(root / "results.json")
        registry["entries"].append(entry)
        row = {key: 0 for key in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=2*n, suffix_execution_records=2*n, checkpoint_records=n, source_execution_records=n,
                   derived_records=n, recorded_failures=len(entry["failures"]))
        inventory.append(row)
        totals.update(counts)
        audit_totals.update({k: audit[k] for k in ("value_checks", "gradient_checks")})
    links = check_links()
    assert links > 263
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = "2026-11-11"
    validation.update({
        "date": "2026-11-11", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E31": len(history),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": 42,
        "new_optimization_runs_note": "E31: 14 shared paid event reconstructions and 28 continuations, no new seed groups.",
        "new_run_breakdown": dict(totals), "new_audit_oracle_calls": dict(audit_totals),
        "new_execution_segments_with_complete_tables": 42, "new_accepted_execution_segments": 30,
        "new_formal_mechanism_execution_segments": 30, "excluded_engineering_executions": 12,
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered 14 event reconstructions, 28 continuations and {links} links. Verified {len(history)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
