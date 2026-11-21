"""Run finite-energy helper-ball interventions and direct-kick controls."""

import os
from pathlib import Path
import sys

for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key,"1")
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from collision_optim.strategic_experiment import main

if __name__=="__main__":
    raise SystemExit(main())
