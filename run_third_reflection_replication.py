"""Fresh E43 control trajectories through the second-reflection policy."""

import argparse
from pathlib import Path

from run_contact_window_guard import PARENT as E36
from run_second_reflection_capped_policy import RULE_PARENTS
from run_two_stage_protection import PARENTS, run

BASE = Path("output/collision_third_reflection_replication")
SPEC = Path("docs/superpowers/specs/2026-11-12-third-reflection-replication-design.md")
SEEDS = {"preflight":[2810],"formal":list(range(2820,2900))}
PARENT_CONFIGS = (*PARENTS,E36,*RULE_PARENTS,*(Path(p)/"config.json" for p in (
    "output/collision_second_reflection_capped_policy","output/collision_third_reflection_guard")))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=SEEDS,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1 or (args.stage=="formal")!=bool(args.lock):
        parser.error("Formal requires the audited preflight lock and positive worker count.")
    run(args,phase="E43",spec=SPEC,seed_sets=SEEDS,parents=PARENT_CONFIGS,second_stage=True,guard_fraction=.8)
