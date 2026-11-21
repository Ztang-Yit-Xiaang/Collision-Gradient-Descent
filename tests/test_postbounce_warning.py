import pytest

from collision_optim.core import State
from collision_optim.objectives import Affine
from collision_optim.population import PopulationOracle
from collision_optim.postbounce_warning import forecast_after_bounce


def test_forecast_spends_one_path_when_warning_precedes_next_contact():
    oracle = PopulationOracle(Affine([1.], 0.))
    result = forecast_after_bounce(oracle, State([0.], [.01], .1, .1), 0., 1e-4, 7, 20)
    assert result["complete"] and result["next_contact"] is None
    assert result["observations"][0]["step"] == 8
    assert oracle.counts() == {"value_calls": 0, "gradient_calls": 0, "path_calls": 1}


def test_forecast_tracks_flat_wall_reflection_without_requerying_line_segments():
    oracle = PopulationOracle(Affine([0.], 0.))
    result = forecast_after_bounce(oracle, State([0.], [.01], .001, -.05), 0., 1e-4, 7, 9)
    assert result["complete"]
    assert .35 < result["next_contact"] < .40
    assert [x["loss_rise"] for x in result["observations"]] == pytest.approx([0., 0.])
    assert oracle.counts() == {"value_calls": 0, "gradient_calls": 1, "path_calls": 2}
