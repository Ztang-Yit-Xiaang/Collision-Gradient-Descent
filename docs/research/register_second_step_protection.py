"""Register E36's matched original and conditional-second-step policies."""

import argparse

from register_contact_warning_replication import register


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args=parser.parse_args()
    register(
        {"passed":args.tests_passed,"seconds":args.test_seconds,
         "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q"},
        phase="E36",stem="second_step_protection",
        spec="docs/superpowers/specs/2026-11-11-second-step-protection-design.md",
        formal_classification="fresh_six_energy_conditional_second_step",
        minimum_links=306,extra_sources=("docs/research/register_second_step_protection.py",),
    )
