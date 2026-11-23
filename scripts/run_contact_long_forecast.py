"""E33: compare short and long warning windows on fresh first reflections."""

import argparse
from pathlib import Path

from collision_optim.contact_long_forecast import forecast_from_reflection
from run_contact_warning_replication import BOUNDARY_STEP, END_STEP, LIMIT, run

SPEC = Path("docs/superpowers/specs/2026-11-11-contact-long-forecast-design.md")
PARENT = Path("output/collision_contact_warning_replication/config.json")
SEEDS = {"preflight": [2370], "formal": list(range(2380, 2420))}


def extended_forecast(oracle, particle, remaining, config, initial_loss):
    return forecast_from_reflection(oracle, particle, remaining, config, initial_loss, LIMIT, BOUNDARY_STEP, END_STEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.stage == "preflight") == bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args, phase="E33", spec=SPEC, parent=PARENT, seed_sets=SEEDS, forecast_extension=extended_forecast)
