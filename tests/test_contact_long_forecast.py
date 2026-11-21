from collision_optim.contact_long_forecast import forecast_from_reflection
from collision_optim.core import State
from collision_optim.objectives import Affine
from collision_optim.population import PopulationConfig, PopulationOracle


def test_extended_forecast_reuses_first_query_beyond_next_observation():
    oracle = PopulationOracle(Affine([1.], 0.))
    value, forecast = forecast_from_reflection(
        oracle, State([0.], [.001], 0., .16), .03, PopulationConfig(), 0., 1e-4, 7, 20)
    assert value < 1e-4
    assert forecast["observations"][-1]["loss_rise"] > 1e-4
    assert forecast["observations"][-1]["step"] > 7
    assert oracle.counts() == {"value_calls": 0, "gradient_calls": 0, "path_calls": 1}


def test_extended_forecast_pays_for_next_reflection_and_new_line():
    oracle = PopulationOracle(Affine([0.], 0.))
    value, forecast = forecast_from_reflection(
        oracle, State([0.], [.001], 0., .16), .03, PopulationConfig(), 0., 1e-4, 7, 20)
    assert value == 0.
    assert forecast["complete"] and forecast["next_contact"] is not None
    assert forecast["observations"][-1]["step"] == 20
    assert oracle.counts() == {"value_calls": 0, "gradient_calls": 1, "path_calls": 2}
