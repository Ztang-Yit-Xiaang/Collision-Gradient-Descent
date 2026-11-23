"""Execute the shared retained and full-window branches for the E45 pair."""

import argparse
from pathlib import Path

from run_third_reflection_replication_branches import run
from run_third_reflection_contact_gate import CONTACT_FRACTION, SEEDS


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    run(args.output,args.workers,phase="E45",seed_sets=SEEDS,saved_window=True,contact_fraction=CONTACT_FRACTION)
