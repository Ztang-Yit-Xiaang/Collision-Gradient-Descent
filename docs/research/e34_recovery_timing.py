"""Read-only E34 timing diagnostic for recovery and a later crossing."""

import csv
import gzip
import json
from pathlib import Path

from finalize_initial_budget import create, sha
from run_conditional_transfer import read


def diagnose(root):
    assert read(root/"audit.json")["passed"]
    rows = []
    for r in read(root/"paired_outcomes.json"):
        transfer = r["branches"]["transfer"]
        if not r["already_deviated_at_reflection"] or not transfer["complete"]:
            continue
        key = transfer["run_id"]
        event_file = root/"events"/(key+".json")
        trace_file = root/"traces"/(key+".csv.gz")
        checkpoint = root/"checkpoints"/(r["context_id"]+".json")
        c = read(checkpoint)
        h = .05
        step = c["boundary_step"]
        contacts = [step*h + event["time"] for event in read(event_file) if event["i"] == 0]
        recovery = transfer["first_saved_recovery"]
        recross = None
        with gzip.open(trace_file, "rt") as stream:
            for row in csv.DictReader(stream):
                time = (step+int(row["step"]))*h
                rise = json.loads(row["values"])[0] - c["initial_loss"]
                if recovery is not None and time > recovery and rise > 1e-4:
                    recross = time
                    break
        rows.append({
            "seed": r["seed"], "run_id": key, "first_recovery": recovery, "first_recross": recross,
            "later_own_reflections": contacts, "within_limit_at_end": transfer["extended"]["within_limit"],
            "source_sha256": {str(p): sha(p) for p in (event_file, trace_file, checkpoint)},
        })
    assert len(rows) == 40
    create(root/"late_recovery_timing.json", {
        "rows": rows, "new_oracle_calls": 0,
        "first_recovery_after_second_reflection": sum(
            r["first_recovery"] is not None and r["later_own_reflections"]
            and r["first_recovery"] > r["later_own_reflections"][0] for r in rows),
        "recrossed": sum(r["first_recross"] is not None for r in rows),
        "recross_before_third_reflection": sum(
            r["first_recross"] is not None and len(r["later_own_reflections"]) >= 2
            and r["first_recross"] < r["later_own_reflections"][1] for r in rows),
    })


if __name__ == "__main__":
    diagnose(Path("output/collision_high_kinetic_intervention_timing"))
