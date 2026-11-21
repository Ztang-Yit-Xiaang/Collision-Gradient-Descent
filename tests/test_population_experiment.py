import csv
import json
import numpy as np
import pytest

from collision_optim.population_experiment import initial_population,make_objective,run_experiments
from collision_optim import population_experiment


@pytest.mark.parametrize("case",["quadratic","double_well","rosenbrock"])
def test_population_initialization_preserves_energy_zero_momentum_and_separation(case):
    objective,_,_=make_objective(case)
    first=initial_population(case,objective,0)
    second=initial_population(case,objective,0)
    assert first.kinetic==pytest.approx(.6,abs=1e-14)
    np.testing.assert_allclose(first.momentum,0,atol=1e-14)
    for p,q in zip(first.particles,second.particles):
        np.testing.assert_array_equal(p.x,q.x)
        np.testing.assert_array_equal(p.velocity,q.velocity)
        assert p.z>objective.value(p.x)
    assert min(np.linalg.norm(np.r_[p.x-q.x,p.z-q.z]) for i,p in enumerate(first.particles)
               for q in first.particles[i+1:])>=.26


def test_paired_export_records_shared_baseline_actual_contacts_and_native_json(tmp_path,monkeypatch):
    monkeypatch.setattr(population_experiment,"_plots",lambda *args:None)
    report=run_experiments(tmp_path,seeds=1,steps=12,cases=["double_well"],geometry_seeds=2)
    assert report["trajectory_count"]==3
    assert report["paired_comparisons"]==2
    assert all(report["checks"].values())
    saved=json.loads((tmp_path/"summary.json").read_text())
    assert saved["maximum_initial_energy_error"]==0
    assert len(saved["mechanisms"])==4
    with (tmp_path/"runs.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert sum(r["method"]=="independent" for r in rows)==1
    assert all(float(r["initial_kinetic"])==pytest.approx(.6) for r in rows)
    assert all(int(r["total_oracle_calls"])<=10000 for r in rows)
    assert (tmp_path/"events"/"mechanism__energy_focusing__collisions.json").exists()
