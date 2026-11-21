"""E37: guard the remaining gap between the fixed forecast and first contact."""

import argparse
from pathlib import Path

from run_two_stage_protection import PARENTS, run

SPEC=Path("docs/superpowers/specs/2026-11-12-contact-window-guard-design.md")
PARENT=Path("output/collision_second_step_protection/config.json")
SEEDS={"preflight":[2610],"formal":list(range(2620,2700))}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=SEEDS,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1 or (args.stage=="preflight")==bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args,phase="E37",spec=SPEC,seed_sets=SEEDS,parents=(*PARENTS,PARENT),second_stage=True,guard_fraction=.8)
