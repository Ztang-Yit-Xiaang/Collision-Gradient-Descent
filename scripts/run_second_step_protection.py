"""E36: test a conditional second-step warning between the two E35 decisions."""

import argparse
from pathlib import Path

from run_two_stage_protection import PARENTS, run

SPEC = Path("docs/superpowers/specs/2026-11-11-second-step-protection-design.md")
PARENT = Path("output/collision_two_stage_protection")
SEEDS = {"preflight":[2550],"formal":list(range(2560,2600))}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=SEEDS,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1 or (args.stage=="preflight")==bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args,phase="E36",spec=SPEC,seed_sets=SEEDS,parents=(
        *PARENTS,PARENT/"config.json",PARENT/"contexts.json",PARENT/"two_stage_recovery_cases.json",
    ),second_stage=True)
