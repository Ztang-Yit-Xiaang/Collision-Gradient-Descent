import json

import numpy as np
import pytest

from collision_optim import stability
from collision_optim import CountedObjective, MomentumSGD, Quadratic, State, run
from collision_optim.core import StepResult


def test_step_refinement_preserves_damping_rate_and_records_different_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(stability, "_plot", lambda *args, **kwargs: None)
    report = stability.run_stability(tmp_path, dimension=8, duration=0.2, h_values=[0.05, 0.025])
    assert len(report["runs"]) == 6
    coarse, fine = report["configurations"]
    assert fine["active"]["beta"]**2 == pytest.approx(coarse["active"]["beta"])
    assert fine["active"]["patience"] == 2*coarse["active"]["patience"]
    assert fine["active"]["epsilon_f"] == 0.5*coarse["active"]["epsilon_f"]
    assert all(row["completed_duration"] == pytest.approx(0.2) for row in report["runs"])
    assert json.loads((tmp_path / "summary.json").read_text())["dimension"] == 8


def test_nonfinite_step_diagnostics_are_not_saved():
    class BadDiagnostics:
        config = MomentumSGD().config
        def step(self, oracle, state):
            return StepResult(state, 1.0, {"energy": np.inf})
    result = run(BadDiagnostics(), CountedObjective(Quadratic()), State([1], [0]), 10)
    assert result.stop_reason.startswith("numerical_error:")
    assert len(result.rows) == 1

