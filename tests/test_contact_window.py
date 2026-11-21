from collision_optim.contact_window import forecast_first_contact
from collision_optim.core import State
from collision_optim.objectives import Affine
from collision_optim.population import PopulationConfig, PopulationOracle


def test_precontact_forecast_reuses_one_line_across_damping_steps():
    oracle = PopulationOracle(Affine([.01], 0.))
    contact = forecast_first_contact(oracle, State([0.], [.01], .024, -.05), PopulationConfig(), 2, 8)
    assert .30 < contact["time"] < .35 and contact["loss"] > 0
    assert oracle.counts() == {"value_calls": 0, "gradient_calls": 0, "path_calls": 1}
