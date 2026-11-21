"""Score existing population results at a common oracle-call cap."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from collision_optim.population_cost import cost_matched_report

if __name__=="__main__":
    report=cost_matched_report(Path("output/population"))
    print({"call_cap":report["call_cap"],"comparisons":len(report["aggregate"])})
