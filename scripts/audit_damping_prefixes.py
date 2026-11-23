"""Check that each control matches its paired baseline before its first action."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys


def audit(output):
    output = Path(output); runs = json.loads((output/"runs.json").read_text())
    lookup = {(r["seed"], r["method"]): r for r in runs}
    prefix_rows = current_losses = controls = movement_roundoffs = 0
    max_movement_difference = 0.
    # Norm statistics use a loop in the canonical trainer and built-in sum in
    # the routed trainer. Allow summation roundoff, not changed trajectories.
    movement_relative_tolerance = 64*sys.float_info.epsilon
    for r in runs:
        if r["method"] not in {"momentum_restart", "momentum_damp", "momentum_periodic"}: continue
        seed, method = r["seed"], r["method"]
        def trace(name):
            with gzip.open(output/"trajectories"/f"s{seed}__{name}.csv.gz", "rt") as stream:
                return list(csv.DictReader(stream))
        base, controlled = trace("momentum"), trace(method)
        actions = json.loads((output/"actions"/f"s{seed}__{method}.json").read_text())
        first = min((a["step"] for a in actions), default=len(controlled))
        for t in range(first):
            for field in ("train_loss", "gradient_norm_before_clip"):
                assert controlled[t][field] == base[t][field], (seed, method, t, field)
            left, right = float(controlled[t]["relative_movement"]), float(base[t]["relative_movement"])
            assert math.isclose(left, right, rel_tol=movement_relative_tolerance, abs_tol=0.), (seed, method, t, "movement")
            difference = abs(left-right)
            max_movement_difference = max(max_movement_difference, difference)
            movement_roundoffs += difference != 0
            prefix_rows += 1
        if first < len(controlled):
            # The action changes inherited momentum, not current parameters:
            # the loss/gradient at the first action still match the baseline.
            for field in ("train_loss", "gradient_norm_before_clip"):
                assert controlled[first][field] == base[first][field], (seed, method, first, field)
            current_losses += 1
        else:
            assert r["final_hash"] == lookup[seed, "momentum"]["final_hash"]
        controls += 1
    result = {"passed": True, "controlled_runs": controls, "exact_loss_and_gradient_norm_pre_action_rows": prefix_rows,
              "exact_current_loss_and_gradient_norm_at_first_action": current_losses,
              "movement_roundoff_rows": movement_roundoffs, "movement_max_abs_difference": max_movement_difference,
              "movement_relative_tolerance": movement_relative_tolerance}
    (output/"prefix_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output", type=Path)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
