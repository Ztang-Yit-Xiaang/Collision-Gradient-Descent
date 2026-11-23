"""E45 fresh starts for the full and selectively extended saved-point rules."""

import argparse
from pathlib import Path

from run_third_reflection_replication import PARENT_CONFIGS
from run_third_reflection_saved_window import BASE as E44, PARENT as E43
from run_two_stage_protection import run

BASE = Path("output/collision_third_reflection_contact_gate")
SPEC = Path("docs/superpowers/specs/2026-11-12-third-reflection-contact-gate-design.md")
SEEDS = {"preflight":[3010],"formal":list(range(3020,3100))}
CONTACT_FRACTION = .8


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=SEEDS,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=="formal")!=bool(args.lock):
        parser.error("Formal requires the audited preflight lock and positive worker count.")
    run(args,phase="E45",spec=SPEC,seed_sets=SEEDS,
        parents=(*PARENT_CONFIGS,E43/"raw_artifact_manifest.json",E44/"raw_artifact_manifest.json"),
        second_stage=True,guard_fraction=.8)
