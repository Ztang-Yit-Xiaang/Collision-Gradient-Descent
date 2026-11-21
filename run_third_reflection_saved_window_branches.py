"""Evaluate both E44 policies at the shared third-reflection checkpoint."""

import argparse
from pathlib import Path

from run_third_reflection_replication_branches import run
from run_third_reflection_saved_window import SEEDS


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    run(args.output,args.workers,phase="E44",seed_sets=SEEDS,saved_window=True)
