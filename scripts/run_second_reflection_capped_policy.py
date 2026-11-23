"""E40: fresh E37 control trajectories for the locked second-reflection rule."""

import argparse
from pathlib import Path

from run_contact_window_guard import PARENT as E36
from run_two_stage_protection import PARENTS, run

BASE = Path("output/collision_second_reflection_capped_policy")
SPEC = Path("docs/superpowers/specs/2026-11-12-second-reflection-capped-policy-design.md")
SEEDS = {"preflight":[2710],"formal":list(range(2720,2800))}
RULE_PARENTS = tuple(Path(p) for p in (
    "output/collision_contact_window_guard/config.json",
    "output/collision_later_reflection_warning/config.json",
    "output/collision_capped_reflection_transfer/config.json",
))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=SEEDS,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=="formal")!=bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight lock.")
    run(args,phase="E40",spec=SPEC,seed_sets=SEEDS,parents=(*PARENTS,E36,*RULE_PARENTS),
        second_stage=True,guard_fraction=.8)
