"""Seal E27 and add its segment counts to the experiment registry."""

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
    "preflight": ("collision_shared_prefix_transfer_preflight", (2, 4, 2), "continuation_equivalence_preflight"),
    "development": ("collision_shared_prefix_transfer_development", (36, 72, 0), "paired_regime_and_threshold_development"),
    "formal": ("collision_shared_prefix_transfer", (40, 80, 0), "held_out_shared_prefix_intervention"),
}


def seal(root, tests):
    files = [Path(n) for n in (
        "run_shared_prefix_transfer.py", "audit_shared_prefix_transfer.py",
        "docs/research/register_shared_prefix_transfer.py", "docs/research/register_overlap_transfer.py",
        "docs/research/e27_prefix_timing.py",
        "docs/superpowers/specs/2026-11-10-shared-prefix-transfer-design.md",
        "docs/collision_shared_prefix_transfer_results.md", "docs/experiment_log.md",
    )]
    create(root / "verification.json", {
        "audit_command": ["python3", "audit_shared_prefix_transfer.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "test_run": tests, "audit": read(root / "audit.json"), "same_runtime_source_all_stages": True,
    })
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as out:
        for path in files:
            out.write(path, str(path))
    members = {str(path): sha(path) for path in files}
    create(root / "postprocessing_sources.json", {
        "archive": str(archive), "archive_sha256": sha(archive), "files": members,
    })
    create(root / "raw_artifact_manifest.json", {"files": [
        artifact(path) for path in sorted(root.rglob("*")) if path.is_file()
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
    fresh = {str(Path("output") / name) for name, _, _ in STAGES.values()}
    assert not old.keys() & fresh
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == old.keys() | fresh
    history = [a for entry in old.values() for a in entry["top_level_artifacts"]]
    for a in history:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(Path("output") / name / "config.json") for name, _, _ in STAGES.values()]
    assert all(cfg["source_sha256"] == configs[0]["source_sha256"] for cfg in configs)
    counts_total, audit_total = Counter(), Counter()
    for (stage, (folder, expected, classification)), config in zip(STAGES.items(), configs):
        root = Path("output") / folder
        audit, counts = read(root / "audit.json"), read(root / "execution_counts.json")
        assert config["stage"] == stage and audit["passed"]
        assert tuple(counts[k] for k in ("shared_prefix_executions", "suffix_executions",
                                         "uninterrupted_replication_executions")) == expected
        assert sha(root / "source.zip") == config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", config["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests))
        tables = {}
        for table, role in (
            ("contexts", "shared_observation_context"),
            ("prefixes", "executed_shared_observation_segment"),
            ("runs", "executed_continuation_segment"),
            ("uninterrupted", "executed_engineering_equivalence_control"),
            ("paired_examples", "derived_policy_comparisons"),
        ):
            path = root / (table + ".json")
            tables[table] = {"path": str(path), "records": len(read(path)), "role": role}
        entry = {
            "phase_id": "E27", "folder": str(root), "classification": classification, "run_date": "2026-11-10",
            "run_rows_are_derived": False, "failures": audit["solver_failures"], "record_tables": tables,
            "config": config, "config_path": str(root / "config.json"), "execution_counts": counts,
            "counting_note": "One executed shared observation and two paid continuations per complete context. Root runs are continuations. The two fitted rules reuse these branches.",
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(x for x in p.rglob("*") if x.is_file()))
            },
        }
        if stage == "formal":
            entry["descriptive_screen"] = read(root / "screen.json")
        registry["entries"].append(entry)
        row = {key: 0 for key in headers}
        row.update({key: entry[key] for key in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=expected[1], suffix_execution_records=expected[1], source_execution_records=expected[0],
                   uninterrupted_replication_records=expected[2], allocation_context_records=counts["contexts"],
                   derived_records=tables["paired_examples"]["records"], recorded_failures=len(audit["solver_failures"]))
        inventory.append(row)
        counts_total.update(counts)
        audit_total.update({key: audit[key] for key in ("value_checks", "gradient_checks")})
    links = check_links()
    assert links > 226
    registry["entries"].sort(key=lambda entry: entry["folder"])
    registry["as_of"] = "2026-11-10"
    validation.update({
        "date": "2026-11-10", "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E27": len(history),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": 236,
        "new_optimization_runs_note": "E27: 78 shared prefixes, 156 continuations, and two uninterrupted engineering runs. Fitted policy comparisons are derived.",
        "new_run_breakdown": dict(counts_total), "new_audit_oracle_calls": dict(audit_total),
        "new_execution_segments_with_complete_tables": 236, "new_accepted_execution_segments": 228,
        "new_formal_mechanism_execution_segments": 120, "excluded_engineering_executions": 8,
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered {counts_total['contexts']} contexts, 236 solver segments and {links} links. Verified {len(history)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
