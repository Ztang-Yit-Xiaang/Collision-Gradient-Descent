"""Seal and register E44, including its no-query archived analysis."""

import argparse

from register_third_reflection_replication import register
from run_third_reflection_saved_window import BASE, SPEC


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
              "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"},
             base_dir=BASE,spec=SPEC,phase="E44",stem="third_reflection_saved_window",result_file="saved_window_results.json",
             min_links=395,extra_sources=("docs/research/register_third_reflection_saved_window.py",
                                           "docs/research/e44_contact_recovery.py","run_third_reflection_saved_window_branches.py"))
