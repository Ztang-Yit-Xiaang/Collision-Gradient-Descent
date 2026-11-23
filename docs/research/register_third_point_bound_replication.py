"""Seal and register E53's fresh paired states and unique continuations."""

import argparse

from register_mixed_post_contact_curvature import register
from run_third_point_bound_replication import BASE, SPEC


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    register({"passed":args.tests_passed,"seconds":args.test_seconds,
        "command":"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q"},
        phase="E53",base=BASE,spec=SPEC,stem="third_point_bound_replication",result_key="new_seed_third_point_bound_result",
        extra_sources=("docs/research/register_third_point_bound_replication.py","scripts/run_mixed_post_contact_curvature.py",
                       "scripts/audit_mixed_post_contact_curvature.py","scripts/audit_third_point_bound.py","scripts/audit_third_saved_point.py",
                       "scripts/run_third_point_bound.py","src/collision_optim/reflection_window.py"))
