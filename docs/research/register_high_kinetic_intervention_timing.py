"""Register E34's three timing policies and its fresh-seed raw data."""

import argparse

from register_contact_warning_replication import register


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register(
        {"passed": args.tests_passed, "seconds": args.test_seconds,
         "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"},
        phase="E34", stem="high_kinetic_intervention_timing",
        spec="docs/superpowers/specs/2026-11-11-high-kinetic-intervention-timing-design.md",
        formal_classification="fresh_high_kinetic_three_timing_policies",
        minimum_links=287, extra_sources=("docs/research/register_high_kinetic_intervention_timing.py",
                                         "docs/research/e34_recovery_timing.py"),
        batch_sizes=(1, 40),
    )
