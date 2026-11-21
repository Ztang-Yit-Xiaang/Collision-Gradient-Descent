"""E48: extend the E47 decision window by one saved observation."""

import argparse
from pathlib import Path

from run_post_contact_curvature_screen import BASE as E47, run

BASE = Path("output/collision_second_post_contact_curvature")
SPEC = Path("docs/superpowers/specs/2026-11-15-second-post-contact-curvature-design.md")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","formal"),required=True)
    parser.add_argument("--lock",type=Path)
    args = parser.parse_args()
    if (args.stage=="formal")!=bool(args.lock):
        parser.error("Formal requires the audited preflight lock.")
    run(args,phase="E48",spec=SPEC,comparison=E47,second_saved_point=True)
