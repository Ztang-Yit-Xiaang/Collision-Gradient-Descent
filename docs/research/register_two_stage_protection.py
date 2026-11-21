"""Register E35's four derived policies and actual shared-path executions."""

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
        phase="E35",stem="two_stage_protection",
        spec="docs/superpowers/specs/2026-11-11-two-stage-protection-design.md",
        formal_classification="fresh_six_energy_four_timing_policies",
        minimum_links=297,extra_sources=("docs/research/register_two_stage_protection.py",
                                        "docs/research/e35_selected_recovery.py"),
    )
