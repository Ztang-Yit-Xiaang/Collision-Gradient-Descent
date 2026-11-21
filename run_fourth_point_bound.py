"""E54: extend E52 only when the fourth saved-point bound warrants it."""

import argparse
from pathlib import Path

from run_second_contact_reflection import run
from run_third_point_bound import contact_safe_bounds
from run_third_point_bound_replication import BASE as E53

BASE = Path("output/collision_fourth_point_bound")
SPEC = Path("docs/superpowers/specs/2026-11-15-fourth-point-bound-design.md")
PREFLIGHT = {f"mixed__s{seed}__{name}" for seed,name in (
    (3301,"energy_safe"),(3301,"late_low"),(3305,"early_low"),(3308,"early_high"),
    (3312,"late_low"),(3320,"early_low"))}
FORECAST_CAP = 9


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight","formal"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if (args.stage=="formal") != bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args, phase="E54", spec=SPEC, preflight=PREFLIGHT, forecast_cap=FORECAST_CAP,
        parent=E53, fourth_saved_point=True, reflection_loss_bounds=contact_safe_bounds)
