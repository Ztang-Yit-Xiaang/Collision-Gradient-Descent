"""Run paid finite-horizon launch selection and matched-impulse controls."""
import os
from pathlib import Path
import sys

for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:os.environ.setdefault(key,"1")
os.environ.setdefault("MPLCONFIGDIR","/tmp/ee5239-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/ee5239-cache")
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from collision_optim.predictive_experiment import main

if __name__=="__main__":raise SystemExit(main())
