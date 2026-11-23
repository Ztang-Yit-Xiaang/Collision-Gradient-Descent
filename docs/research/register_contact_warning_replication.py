"""Seal fresh first-reflection studies and append their execution counts."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import zipfile

from finalize_initial_budget import create, sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_overlap_transfer import archive_check, check_links

def seal(root, tests, audit, stem, spec, extra_sources):
    files = [Path(n) for n in (
        f"scripts/run_{stem}.py", f"scripts/audit_{stem}.py",
        "docs/research/register_contact_warning_replication.py", "docs/research/register_overlap_transfer.py",
        spec, f"docs/collision_{stem}_results.md", "docs/experiment_log.md", *extra_sources,
    )]
    create(root / "verification.json", {
        "audit": audit, "test_run": tests, "audit_command": ["python3", f"scripts/audit_{stem}.py", str(root)],
        "audit_environment": {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    })
    archive = root / "postprocessing_sources.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, str(path))
    members = {str(p): sha(p) for p in files}
    create(root / "postprocessing_sources.json", {
        "archive": str(archive), "archive_sha256": sha(archive), "files": members,
    })
    create(root / "raw_artifact_manifest.json", {"files": [
        artifact(p) for p in sorted(root.rglob("*")) if p.is_file()
    ]})
    return archive_check(archive, members)


def register(tests, *, phase="E32", stem="contact_warning_replication",
             spec="docs/superpowers/specs/2026-11-11-contact-warning-replication-design.md",
             formal_classification="fresh_seed_all_eligible_first_reflections", minimum_links=271,
             extra_sources=(), batch_sizes=(6, 240), run_date="2026-11-11"):
    base = "collision_" + stem
    stages = ((base + "_preflight", "engineering_preflight", batch_sizes[0]),
              (base, formal_classification, batch_sizes[1]))
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    historical = {e["folder"]: e for e in registry["entries"]}
    new = {str(Path("output") / n) for n, _, _ in stages}
    assert not historical.keys() & new
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == historical.keys() | new
    artifacts = [a for e in historical.values() for a in e["top_level_artifacts"]]
    for a in artifacts:
        assert sha(Path(a["path"])) == a["sha256"], a["path"]
    configs = [read(Path("output") / folder / "config.json") for folder, _, _ in stages]
    assert configs[0]["source_sha256"] == configs[1]["source_sha256"]
    totals, audit_totals = Counter(), Counter()
    segments_by_stage = {}
    for (folder, kind, n), config in zip(stages, configs):
        root = Path("output") / folder
        audit, counts = read(root / "audit.json"), read(root / "execution_counts.json")
        assert config["phase"] == phase and audit["passed"] and counts["contexts"] == n
        assert counts["prefix_executions"] >= n
        expected_continuations = audit["continuations"] if "continuations" in audit else 2 * audit["eligible"]
        assert counts["continuations"] == expected_continuations
        assert sha(root / "source.zip") == config["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root / "source.zip", config["source_sha256"]))
        validation["source_archive_checks"].append(seal(root, tests, audit, stem, spec, extra_sources))
        tables = {}
        for table, role in (
            ("contexts", "new_initial_states"), ("prefixes", "executed_precontact_prefix"),
            ("runs", "executed_fraction_and_tail_continuation"), ("uninterrupted", "executed_engineering_control"),
            ("paired_outcomes", "derived_warning_and_intervention_comparison"),
            ("already_deviated_cases", "derived_recovery_cases"),
        ):
            path = root / (table + ".json")
            tables[table] = {"path": str(path), "records": len(read(path)), "role": role}
        entry = {
            "phase_id": phase, "folder": str(root), "classification": kind, "run_date": run_date,
            "config": config, "config_path": str(root / "config.json"), "run_rows_are_derived": False,
            "failures": audit["solver_failures"], "record_tables": tables, "execution_counts": counts,
            "counting_note": "A prefix and first contact are shared by both continuations. Root runs are actual continuations. All rules reuse the same pairs.",
            "top_level_artifacts": [artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
            "nested_artifact_counts": {
                p.name: {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
                for p in sorted(root.iterdir()) if p.is_dir()
                if (files := list(f for f in p.rglob("*") if f.is_file()))
            },
        }
        if config["stage"] == "formal":
            entry["contact_warning_result"] = read(root / "results.json")
        probe_file = root / "engineering_guard_probe.json"
        if probe_file.exists():
            probe = read(probe_file)
            assert probe["passed"] and probe["new_trajectories"] == 0
            entry["engineering_probe_queries"] = probe["actual_probe_queries"]
            totals["actual_engineering_probe_queries"] += probe["actual_probe_queries"]
        registry["entries"].append(entry)
        row = {k: 0 for k in headers}
        row.update({k: entry[k] for k in ("phase_id", "folder", "classification", "config_path", "run_rows_are_derived")})
        row.update(runs_records=counts["continuations"], suffix_execution_records=counts["continuations"],
                   source_execution_records=counts["prefix_executions"], contact_context_records=counts["exact_contacts"],
                   uninterrupted_replication_records=counts["uninterrupted_preflight_executions"],
                   derived_records=n, recorded_failures=len(audit["solver_failures"]))
        inventory.append(row)
        totals.update(counts)
        segments_by_stage[config["stage"]] = sum(counts[k] for k in (
            "prefix_executions", "exact_contacts", "continuations", "uninterrupted_preflight_executions"))
        audit_totals.update({k: audit[k] for k in ("value_checks", "gradient_checks")})
    links = check_links()
    assert links > minimum_links
    registry["entries"].sort(key=lambda e: e["folder"])
    registry["as_of"] = run_date
    segments = sum(segments_by_stage.values())
    validation.update({
        "date": run_date, "passed": True, "local_document_links_checked": True,
        "local_document_link_count": links, "local_document_link_errors": [],
        "retained_output_directories": len(inventory),
        "root_run_rows_retained": sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained": sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked": sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        f"historical_output_artifact_hashes_unchanged_for_{phase}": len(artifacts),
        "source_archive_members_checked": sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs": segments,
        "new_optimization_runs_note": f"{phase}: shared prefixes and contacts, paired continuations, and separate engineering replications.",
        "new_run_breakdown": dict(totals), "new_audit_oracle_calls": dict(audit_totals),
        "new_execution_segments_with_complete_tables": segments,
        "new_accepted_execution_segments": segments_by_stage["formal"],
        "new_formal_mechanism_execution_segments": segments_by_stage["formal"],
        "excluded_engineering_executions": segments_by_stage["preflight"],
        "latest_full_test_count": tests["passed"], "latest_full_test_seconds": tests["seconds"],
    })
    write(registry_path, registry)
    write(validation_path, validation)
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda row: row["folder"]))
    print(f"Registered {totals['contexts']} initial states and {segments} segments. Checked {links} links and {len(artifacts)} historical artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed": args.tests_passed, "seconds": args.test_seconds,
              "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"})
