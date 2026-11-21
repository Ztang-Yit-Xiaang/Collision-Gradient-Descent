"""Register E37's exact-contact guard and engineering query probe."""

import argparse

from register_contact_warning_replication import register


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args=parser.parse_args()
    register(
        {"passed":args.tests_passed,"seconds":args.test_seconds,
         "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"},
        phase="E37",stem="contact_window_guard",
        spec="docs/superpowers/specs/2026-11-12-contact-window-guard-design.md",
        formal_classification="fresh_contact_window_vs_fixed_second_step",
        minimum_links=315,extra_sources=("docs/research/register_contact_window_guard.py","docs/research/e37_contact_guard_probe.py"),
        batch_sizes=(6,480),run_date="2026-11-12",
    )
