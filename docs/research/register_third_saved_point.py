"""Seal and register E51 using the paired replay record format."""

import argparse

from register_second_contact_reflection import register
from run_third_saved_point import BASE, SPEC


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"},
        phase="E51",base=BASE,spec=SPEC,stem="third_saved_point",diagnostic="previous_unprotected_cases",
        extra_sources=("docs/research/register_third_saved_point.py","scripts/run_second_contact_reflection.py"))
