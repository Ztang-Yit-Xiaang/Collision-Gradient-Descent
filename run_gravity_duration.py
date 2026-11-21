"""Supplement the capped benchmark with a fixed-duration barrier-crossing audit."""

import os
from pathlib import Path
import sys

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from collision_optim.gravity_audit import run_duration_audit

if __name__ == "__main__":
    run_duration_audit(Path("output/gravity_duration"))
