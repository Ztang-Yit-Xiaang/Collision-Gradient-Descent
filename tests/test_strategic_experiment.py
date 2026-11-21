import csv
import gzip
import json

import numpy as np
import pytest

from collision_optim.population import PopulationGravity
from collision_optim.strategic import MODES,StrategicConfig
from collision_optim.strategic_experiment import initial_target,run_experiments


def test_trapped_initial_states_cannot_cross_barrier_without_supply():
    config=StrategicConfig();method=PopulationGravity(config.population)
    for seed in range(20):
        objective,_,boundary,initial=initial_target("bad_well",seed,config)
        barrier=config.population.physics.gravity*objective.value(np.array([boundary,0.]))
        assert method.energy(initial)<barrier
        repeated=initial_target("bad_well",seed,config)[-1]
        np.testing.assert_array_equal(initial.particles[0].x,repeated.particles[0].x)
        np.testing.assert_array_equal(initial.particles[0].velocity,repeated.particles[0].velocity)


def test_exports_keep_paired_target_only_states_and_actual_cost_rows(tmp_path):
    report=run_experiments(tmp_path,seeds=1,steps=45,cases=["bad_well"],plots=False)
    assert report["completed"]==4
    runs=json.loads((tmp_path/"runs.json").read_text())
    assert len(runs)==4
    first=[]
    for mode in MODES:
        with gzip.open(tmp_path/"trajectories"/f"bad_well__s00__{mode}.csv.gz","rt") as stream:
            rows=list(csv.DictReader(stream))
        first.append((rows[0]["positions"],rows[0]["velocities"],rows[0]["energy"]))
        run=next(r for r in runs if r["mode"]==mode)
        eligible=[r for r in rows if int(r["total_oracle_calls"])<=report["cost_cap"]]
        assert run["cost_time"]==float(eligible[-1]["time"])
        assert run["cost_calls"]==int(eligible[-1]["total_oracle_calls"])
        assert max(abs(float(r["energy_balance_error"])) for r in rows)<1e-10
    assert all(f==first[0] for f in first)
    with pytest.raises(ValueError,match="new output directory"):
        run_experiments(tmp_path,seeds=1,steps=1,plots=False)


def test_initial_only_budget_failure_exports_finite_diagnostics(tmp_path):
    report=run_experiments(tmp_path,seeds=1,steps=1,cases=["bad_well"],oracle_budget=1,plots=False)
    assert report["budget_stops"]==4
    assert report["cost_covered"]==0
    assert all(np.isfinite(r["minimum_surface_gap"]) for r in json.loads((tmp_path/"runs.json").read_text()))
