import csv
import json

import numpy as np
import pytest

from collision_optim import GravityConfig
from collision_optim import gravity_experiment as experiment
from collision_optim.highdim import make_case


@pytest.mark.parametrize("mode", ["airborne", "surface"])
@pytest.mark.parametrize("case", experiment.CASES)
def test_energy_control_matches_gravity_and_surface_speed_is_normalized(mode, case):
    cfg = GravityConfig()
    objective = make_case(case, 100)
    plain, gravity, matched = experiment._initials(objective, case, 0, mode, cfg)
    potential = cfg.gravity*cfg.height_scale*objective.value(plain.x)
    assert gravity.kinetic+cfg.gravity*gravity.z == pytest.approx(matched.kinetic+potential, abs=1e-13)
    np.testing.assert_array_equal(plain.x, gravity.x)
    np.testing.assert_array_equal(plain.v, gravity.v)
    if mode == "surface":
        tangent_speed = cfg.height_scale*objective.gradient(gravity.x) @ gravity.v
        assert gravity.vz == pytest.approx(tangent_speed)
        expected = 0.18 if "wells" in case else 0.045
        assert gravity.kinetic == pytest.approx(expected, abs=1e-13)


def test_export_retains_costs_failures_energy_and_physical_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_plot", lambda *args: None)
    report = experiment.run_gravity(tmp_path, dimensions=[8], cases=["ellipsoid"],
                                    seeds=1, steps=3, gradient_budget=20)
    assert report["total_runs"] == 10
    assert all(report["checks"].values())
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert len(saved["physics"]) == 12
    with (tmp_path / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert all("path_calls" in row and "completed_time" in row for row in rows)
    assert any(row["stop_reason"].startswith("budget:") for row in rows)
