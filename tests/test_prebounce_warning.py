import numpy as np
import pytest

from collision_optim.prebounce_warning import flight_factors, forecast_rise


def test_forecast_recovers_quadratic_when_initial_loss_drops():
    factors = flight_factors()
    speed = .04
    distances = speed * factors
    slope, curvature = -.01, 6.
    increases = slope * distances + .5 * curvature * distances**2
    assert increases[1] < 0
    expected = np.max(increases[3:])
    assert forecast_rise(increases[1:3], speed, factors) == pytest.approx(expected)
    assert forecast_rise(increases[1:2], speed, factors, curvature) == pytest.approx(np.max(increases[2:]))


def test_stationary_particle_forecast_uses_unchanged_loss():
    assert forecast_rise([0.], 0., flight_factors(), 4.) == 0.
