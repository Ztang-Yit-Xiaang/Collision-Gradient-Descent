"""Read completed E27 prefix artifacts and compare two observed times."""

from pathlib import Path

from audit_shared_prefix_transfer import load_segment
from finalize_initial_budget import create, sha
from run_conditional_transfer import read


def timing(root):
    assert read(root / "audit.json")["passed"]
    observations = []
    for c in read(root / "contexts.json"):
        if c["condition"] != "good_present":
            continue
        key = c["prefix_run_id"]
        rows, _, _, losses = load_segment(root, key)
        events = read(root / "events" / (key + ".json"))
        first_rise = next((float(row["time"]) for row, loss in zip(rows, losses)
                           if loss[0] - c["values"][0] > 1e-4), None)
        first_wall = next((e["time"] for e in events if e["kind"] == "wall" and e["i"] == 0), None)
        observations.append({
            "seed": c["seed"], "source_prefix": key, "first_observed_deviation": first_rise,
            "first_terrain_reflection": first_wall,
            "observed_deviation_before_first_reflection": (
                first_rise is not None and first_wall is not None and first_rise < first_wall
            ),
        })
    assert len(observations) == 20
    create(root / "prefix_timing.json", {
        "new_oracle_calls": 0, "audit_sha256": sha(root / "audit.json"),
        "command": "PYTHONPATH=src:. python3 docs/research/e27_prefix_timing.py",
        "observations": sorted(observations, key=lambda r: r["seed"]),
        "deviation_before_first_reflection": sum(r["observed_deviation_before_first_reflection"] for r in observations),
        "prefix_deviations": sum(r["first_observed_deviation"] is not None for r in observations),
        "prefix_reflections": sum(r["first_terrain_reflection"] is not None for r in observations),
    })


if __name__ == "__main__":
    timing(Path("output/collision_shared_prefix_transfer"))
