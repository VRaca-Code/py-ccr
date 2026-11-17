"""Regression tests for curve-consistent Hull-White simulation."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.hull_white_paths import simulate_hull_white_short_rates
from pyccr.simulation.time_grid import TimeGrid


@pytest.fixture
def model():
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0),
        zero_rates=(0.03,) * 5,
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.1,
        volatility=0.01,
    )


@pytest.fixture
def config():
    return SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=1000,
        seed=42,
    )


def test_output_shape_and_finiteness(config, model):
    paths = simulate_hull_white_short_rates(config, model)
    assert paths.shape == (1000, 13)
    assert np.all(np.isfinite(paths))


def test_initial_short_rate(config, model):
    paths = simulate_hull_white_short_rates(config, model)
    np.testing.assert_allclose(paths[:, 0], 0.03)


def test_seed_reproducibility(config, model):
    first = simulate_hull_white_short_rates(config, model)
    second = simulate_hull_white_short_rates(config, model)
    np.testing.assert_array_equal(first, second)


def test_zero_volatility_is_deterministic(model):
    deterministic = HullWhiteCurveModel(
        curve=model.curve,
        mean_reversion=model.mean_reversion,
        volatility=0.0,
    )
    config = SimulationConfig(TimeGrid.uniform(1.0, 4), 20, seed=7)
    paths = simulate_hull_white_short_rates(config, deterministic)
    np.testing.assert_allclose(paths, 0.03)


def test_antithetic_symmetry(model):
    config = SimulationConfig(
        TimeGrid.uniform(1.0, 8),
        n_paths=100,
        seed=123,
        antithetic=True,
    )
    paths = simulate_hull_white_short_rates(config, model)

    times = np.asarray(config.time_grid.times)
    a = model.mean_reversion
    sigma = model.volatility

    shift = 0.03 + 0.5 * (sigma / a) ** 2 * (-np.expm1(-a * times)) ** 2

    actual = paths[:50] + paths[50:]
    expected = np.broadcast_to(2.0 * shift, actual.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_external_draws_are_reproducible(config, model):
    draws = np.ones((config.n_paths, config.time_grid.n_steps))
    first = simulate_hull_white_short_rates(config, model, normal_draws=draws)
    second = simulate_hull_white_short_rates(config, model, normal_draws=draws)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "bad_shape",
    [(12,), (1000, 13), (999, 12), (1000, 12, 1)],
)
def test_invalid_external_draw_shapes(config, model, bad_shape):
    with pytest.raises(ValueError, match="shape"):
        simulate_hull_white_short_rates(
            config,
            model,
            normal_draws=np.zeros(bad_shape),
        )


def test_nonfinite_external_draws(config, model):
    draws = np.zeros((1000, 12))
    draws[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulate_hull_white_short_rates(config, model, normal_draws=draws)


def test_curve_horizon_validation(model):
    config = SimulationConfig(
        TimeGrid.uniform(6.0, 12),
        n_paths=10,
        seed=42,
    )
    with pytest.raises(ValueError, match="horizon"):
        simulate_hull_white_short_rates(config, model)


def test_invalid_config(model):
    with pytest.raises(TypeError, match="SimulationConfig"):
        simulate_hull_white_short_rates(None, model)


def test_invalid_model(config):
    with pytest.raises(TypeError, match="HullWhiteCurveModel"):
        simulate_hull_white_short_rates(config, None)
