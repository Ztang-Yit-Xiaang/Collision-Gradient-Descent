"""Seal and register E45's shared branches and archived gate scan."""

import argparse

from register_third_reflection_replication import register
from run_third_reflection_contact_gate import BASE, SPEC


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int,required=True)
    parser.add_argument("--test-seconds",type=float,required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
              "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"},
             base_dir=BASE,spec=SPEC,phase="E45",stem="third_reflection_contact_gate",result_file="contact_gate_results.json",
             min_links=406,extra_sources=("docs/research/register_third_reflection_contact_gate.py",
                "docs/research/e45_archived_gate_scan.py","docs/research/e45_recovery_timelines.py",
                "run_third_reflection_contact_gate_branches.py"))
