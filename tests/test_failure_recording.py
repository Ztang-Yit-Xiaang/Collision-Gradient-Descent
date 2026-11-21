import json

import numpy as np
import pytest

from collision_optim import CountedObjective, MomentumSGD, State, run
from collision_optim import highdim


def test_numeric_overflow_keeps_last_complete_state_and_actual_cost():
    class Quartic:
        def value(self, x): return float(np.float64(x[0])**4)
        def gradient(self, x): return 4*x**3
    initial = State([1.0], [1e100])
    result = run(MomentumSGD(), CountedObjective(Quartic()), initial, 20)
    assert result.stop_reason.startswith("numerical_error:")
    assert len(result.rows) == 1
    np.testing.assert_array_equal(result.final_state.x, initial.x)
    assert result.counts["value_calls"] == 3
    assert result.counts["gradient_calls"] == 1


def test_programming_shape_errors_are_not_hidden_as_numeric_failures():
    class WrongShape:
        def value(self, x): return float(x@x)
        def gradient(self, x): return np.ones(x.size+1)
    with pytest.raises(ValueError, match="shape"):
        run(MomentumSGD(), CountedObjective(WrongShape()), State([1.0], [0.0]), 3)


def test_resume_reuses_complete_runs_without_reexecuting_them(tmp_path, monkeypatch):
    monkeypatch.setattr(highdim, "_plots", lambda *args, **kwargs: None)
    first = highdim.run_highdim(tmp_path, dimensions=[8], cases=["ellipsoid"], seeds=1, steps=8)
    def should_not_execute(*args, **kwargs):
        pytest.fail("Completed checkpoint was reexecuted")
    monkeypatch.setattr(highdim, "run", should_not_execute)
    second = highdim.run_highdim(tmp_path, dimensions=[8], cases=["ellipsoid"], seeds=1, steps=8, resume=True)
    assert second["resume_reused_runs"] == 5
    assert first["aggregate"] == second["aggregate"]
    assert json.loads((tmp_path / "checkpoint.json").read_text())["runs"]


def test_resume_rejects_changed_experimental_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(highdim, "_plots", lambda *args, **kwargs: None)
    highdim.run_highdim(tmp_path, dimensions=[8], cases=["ellipsoid"], seeds=1, steps=8)
    with pytest.raises(ValueError, match="changed active"):
        highdim.run_highdim(tmp_path, dimensions=[8], cases=["ellipsoid"], seeds=1, steps=8, h=0.02, resume=True)

