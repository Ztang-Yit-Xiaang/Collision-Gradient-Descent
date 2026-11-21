"""Register the fresh-seed E33 forecast and intervention experiment."""

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
        phase="E33", stem="contact_long_forecast",
        spec="docs/superpowers/specs/2026-11-11-contact-long-forecast-design.md",
        formal_classification="fresh_seed_short_vs_extended_contact_forecasts",
        minimum_links=279, extra_sources=("docs/research/register_contact_long_forecast.py",),
    )
