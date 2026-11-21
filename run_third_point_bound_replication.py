"""E53: compare the sealed E51 and E52 rules on fresh paired states."""

import argparse
from pathlib import Path

from run_mixed_post_contact_curvature import run
from run_post_contact_positive_stress import read, sha
from run_third_point_bound import BASE as E52, contact_safe_bounds
from run_third_saved_point import BASE as E51, FORECAST_CAP

BASE = Path("output/collision_third_point_bound_replication")
SPEC = Path("docs/superpowers/specs/2026-11-15-third-point-bound-replication-design.md")
SEEDS = {"preflight": (3290,), "formal": tuple(range(3300,3324))}
FORECAST_OPTIONS = {"e51":{"third_saved_point":True},
                    "e52":{"third_saved_point":True,"reflection_loss_bounds":contact_safe_bounds}}
SEALED_POLICY_SOURCES = ("src/collision_optim/reflection_window.py","src/collision_optim/gravity.py",
                         "src/collision_optim/core.py","run_post_contact_curvature_screen.py",
                         "run_third_point_bound.py")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight","formal"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if (args.stage=="formal") != bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    source = read(E52/"config.json")["source_sha256"]
    for path in SEALED_POLICY_SOURCES:
        assert sha(Path(path)) == source[path], path
    run(args, phase="E53", spec=SPEC, seeds=SEEDS, parents=(E51,E52),
        forecast_options=FORECAST_OPTIONS, forecast_cap=FORECAST_CAP)
