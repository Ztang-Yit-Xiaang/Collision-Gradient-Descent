"""Record E38, including the first engineering run's failed export."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from finalize_initial_budget import create, sha
from refresh_experiment_inventory import read, write
from register_conditional_transfer import artifact
from register_contact_warning_replication import seal
from register_overlap_transfer import archive_check, check_links

STEM = "later_reflection_warning"
BASE = "output/collision_" + STEM
SPEC = "docs/superpowers/specs/2026-11-12-later-reflection-warning-design.md"
STAGES = {
    BASE+"_preflight": "engineering_preflight_audit_export_failure",
    BASE+"_preflight_verified": "engineering_preflight",
    BASE: "retrospective_paired_later_reflection_mechanism",
}
RECORDS = {
    "contexts": "reused_parent_trajectories",
    "runs": "executed_fraction_and_tail_continuations",
    "paired_outcomes": "derived_policy_comparison",
    "already_deviated_cases": "derived_recovery_states",
    "late_crossing_mechanisms": "derived_parent_crossing_timing_and_geometry",
    "warning_checkpoints": "audited_reflection_states_and_forecasts",
    "transfer_effects": "derived_transfer_energy_and_loss_effects",
}


def register(tests):
    registry_path = Path("docs/experiment_registry.json")
    inventory_path = Path("docs/experiment_inventory.csv")
    validation_path = Path("docs/recordkeeping_validation.json")
    registry, validation = read(registry_path), read(validation_path)
    with inventory_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, inventory = reader.fieldnames, list(reader)
    historical = {e["folder"]: e for e in registry["entries"]}
    assert not historical.keys() & STAGES.keys()
    assert {str(p) for p in Path("output").iterdir() if p.is_dir()} == historical.keys() | STAGES.keys()
    old_artifacts = [a for e in historical.values() for a in e["top_level_artifacts"]]
    for item in old_artifacts:
        assert sha(Path(item["path"]))==item["sha256"],item["path"]
    links = check_links()
    assert links>326
    replay_source = read(Path(BASE)/"config.json")["source_sha256"]
    totals,audit_totals = Counter(),Counter()
    for folder,classification in STAGES.items():
        root = Path(folder)
        cfg = read(root/"config.json")
        counts = read(root/"execution_counts.json")
        failed = folder.endswith("_preflight")
        assert (cfg["source_sha256"]!=replay_source)==failed
        assert cfg["phase"]=="E38" and counts["new_initial_states"]==0
        assert sha(root/"source.zip")==cfg["source_archive_sha256"]
        validation["source_archive_checks"].append(archive_check(root/"source.zip",cfg["source_sha256"]))
        audit = read(root/"audit.json") if not failed else {
            "passed":False,"failure_kind":"audit_export_serialization","solver_failures":[],
            "audit_checks_performed_before_serialization_failure":True,
            "value_checks":265,"gradient_checks":9,
            "count_provenance":"The failing export follows all raw audits; reconstruct counts from the archived identical engineering trajectory outputs and audit source.",
        }
        failures = [] if not failed else [{"kind":"audit_export_serialization",
            "reason":"NumPy boolean serialization failed while writing warning_checkpoints.json after computing the audits.",
            "execution_counts_path":str(root/"execution_counts.json")}]
        assert not audit["solver_failures"] and (audit["passed"] or failed)
        extra = (str(Path(__file__).resolve().relative_to(Path.cwd())),"docs/research/e38_transfer_effects.py")
        validation["source_archive_checks"].append(seal(root,tests,audit,STEM,SPEC,extra))
        tables = {}
        for name,role in RECORDS.items():
            path = root/(name+".json")
            if path.exists() and path.stat().st_size:
                data = read(path)
                records = len(data["rows"]) if name=="transfer_effects" else len(data)
                tables[name] = {"path":str(path),"role":role,"records":records}
        entry = {"phase_id":"E38","folder":folder,"classification":classification,"run_date":"2026-11-12",
                 "config":cfg,"config_path":str(root/"config.json"),"run_rows_are_derived":False,
                 "failures":failures,"record_tables":tables,"execution_counts":counts,
                 "counting_note":"All initial states and reflection events reuse E37. A paired checkpoint contributes two actual continuations; forecasts and derived comparisons are not trajectories.",
                 "top_level_artifacts":[artifact(p) for p in sorted(root.iterdir()) if p.is_file()],
                 "nested_artifact_counts":{p.name:{"files":len(files),"bytes":sum(f.stat().st_size for f in files)}
                     for p in sorted(root.iterdir()) if p.is_dir() if (files:=list(f for f in p.rglob("*") if f.is_file()))}}
        if folder==BASE:
            entry["later_reflection_warning_result"] = read(root/"results.json")
        registry["entries"].append(entry)
        row = {key:0 for key in headers}
        row.update({key:entry[key] for key in ("phase_id","folder","classification","config_path","run_rows_are_derived")})
        row.update(runs_records=counts["continuations"],suffix_execution_records=counts["continuations"],
                   contact_context_records=counts["saved_event_reconstructions"],
                   derived_records=tables.get("paired_outcomes",{}).get("records",0),recorded_failures=len(failures))
        inventory.append(row)
        totals.update(counts)
        audit_totals.update({key:audit[key] for key in ("value_checks","gradient_checks")})
    registry["entries"].sort(key=lambda e:e["folder"])
    registry["as_of"] = "2026-11-12"
    accepted = read(Path(BASE)/"execution_counts.json")["continuations"]
    validation.update({"date":"2026-11-12","passed":True,"local_document_links_checked":True,
        "local_document_link_count":links,"local_document_link_errors":[],"retained_output_directories":len(inventory),
        "root_run_rows_retained":sum(int(r["runs_records"]) for r in inventory),
        "root_tuning_rows_retained":sum(int(r["tuning_records"]) for r in inventory),
        "top_level_artifact_hashes_checked":sum(len(e["top_level_artifacts"]) for e in registry["entries"]),
        "historical_output_artifact_hashes_unchanged_for_E38":len(old_artifacts),
        "source_archive_members_checked":sum(a["members_checked"] for a in validation["source_archive_checks"]),
        "new_optimization_runs":totals["continuations"],
        "new_optimization_runs_note":"E38: new paired continuations from archived reflections; forecast-only checks are not runs.",
        "new_run_breakdown":dict(totals),"new_audit_oracle_calls":dict(audit_totals),
        "new_execution_segments_with_complete_tables":totals["continuations"],
        "new_accepted_execution_segments":accepted,"new_formal_mechanism_execution_segments":accepted,
        "excluded_engineering_executions":totals["continuations"]-accepted,
        "latest_full_test_count":tests["passed"],"latest_full_test_seconds":tests["seconds"]})
    write(registry_path,registry)
    write(validation_path,validation)
    with inventory_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted(inventory,key=lambda r:r["folder"]))
    print(f"Registered {totals['continuations']} continuations. Checked {links} links and {len(old_artifacts)} historical artifacts.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"})
