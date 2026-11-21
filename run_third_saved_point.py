"""E51: forecast one more saved point with natural reflection."""

import argparse
from pathlib import Path

from run_second_contact_reflection import BASE as E50, run

BASE = Path("output/collision_third_saved_point")
SPEC = Path("docs/superpowers/specs/2026-11-15-third-saved-point-design.md")
PREFLIGHT = {f"mixed__s{seed}__{stratum}" for seed,stratum in (
    (3200,"energy_safe"), (3200,"early_high"), (3200,"late_low"),
    (3202,"early_low"), (3202,"late_mid"), (3209,"early_low"))}
FORECAST_CAP = 7


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight","formal"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if (args.stage=="formal") != bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args, phase="E51", spec=SPEC, preflight=PREFLIGHT, forecast_cap=FORECAST_CAP,
        comparison=E50, third_saved_point=True)
