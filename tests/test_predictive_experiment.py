import csv
from dataclasses import replace
import gzip
import json
import numpy as np
import pytest

from collision_optim.predictive import PredictiveConfig
from collision_optim.predictive_experiment import initial_case,run_experiments
from collision_optim.strategic import StrategicConfig


def test_saddle_initialization_preserves_exact_unforced_symmetry():
    config=PredictiveConfig()
    for seed in range(100,120):
        objective,minimum,boundary,state=initial_case("saddle",seed,config)
        p=state.particles[0]
        assert p.x[0]==p.v[0]==objective.gradient(p.x)[0]==0
        assert minimum==0 and boundary is None
        assert p.z>objective.value(p.x)


def test_exported_cost_cap_contains_no_virtual_improvement_and_keeps_shared_starts(tmp_path):
    strategy=replace(StrategicConfig(),window=.05,patience=1)
    config=PredictiveConfig(strategy=strategy,horizon=.2,epsilon_gain=100.,max_decisions=1)
    report=run_experiments(tmp_path,seeds=1,steps=12,cases=["bad_well"],config=config,make_plots=False)
    assert report["completed"]==4 and report["cost_covered"]==4
    runs=json.loads((tmp_path/"runs.json").read_text());initials=[]
    for run in runs:
        with gzip.open(tmp_path/"trajectories"/f"bad_well__s100__{run['mode']}.csv.gz","rt") as stream:
            rows=list(csv.DictReader(stream))
        initials.append((rows[0]["positions"],rows[0]["velocities"],rows[0]["energy"]))
        assert run["planning_calls"]+run["dynamics_calls"]==run["total_oracle_calls"]
        eligible=[r for r in rows if int(r["total_oracle_calls"])<=report["cost_cap"]]
        assert int(eligible[-1]["total_oracle_calls"])==report["cost_cap"]
        assert run["cost_time"]==float(eligible[-1]["time"])
        if run["mode"].startswith("predictive"):
            assert run["shots"]==0
            assert run["positive_supply"]==0
    assert all(i==initials[0] for i in initials)
    with pytest.raises(ValueError,match="new output directory"):
        run_experiments(tmp_path,seeds=1,steps=1,make_plots=False)


def test_rejected_forecasts_do_not_change_unforced_saddle_symmetry(tmp_path):
    strategy=replace(StrategicConfig(),window=.05,patience=1)
    config=PredictiveConfig(strategy=strategy,horizon=.2,epsilon_gain=100.,max_decisions=1)
    report=run_experiments(tmp_path,seeds=1,steps=8,cases=["saddle"],config=config,make_plots=False)
    assert report["failures"]==0
    for mode in ["continue","predictive_ball"]:
        with gzip.open(tmp_path/"trajectories"/f"saddle__s100__{mode}.csv.gz","rt") as stream:
            rows=list(csv.DictReader(stream))
        assert all(json.loads(r["target_x"])[0]==0 for r in rows)
        assert all(np.isfinite(float(r["energy_balance_error"])) for r in rows)
