"""Regression tests for risk-neutral FX spot simulation."""

import numpy as np
import pytest

from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


@pytest.fixture
def config():
    return SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=1000,
        seed=42,
    )


@pytest.fixture
def parameters():
    return {
        "initial_spot": 1.10,
        "domestic_rate": 0.04,
        "foreign_rate": 0.02,
        "volatility": 0.15,
    }


def test_shape_initial_value_and_positivity(config, parameters):
    paths = simulate_fx_spot_paths(config, **parameters)

    assert paths.shape == (1000, 13)
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0)

    np.testing.assert_array_equal(
        paths[:, 0],
        np.full(config.n_paths, 1.10),
    )


def test_seed_reproducibility(config, parameters):
    first = simulate_fx_spot_paths(config, **parameters)
    second = simulate_fx_spot_paths(config, **parameters)

    np.testing.assert_array_equal(first, second)


def test_zero_volatility_deterministic(config, parameters):
    paths = simulate_fx_spot_paths(
        config,
        **{**parameters, "volatility": 0.0},
    )

    times = np.asarray(config.time_grid.times)
    expected = 1.10 * np.exp((0.04 - 0.02) * times)

    np.testing.assert_allclose(
        paths,
        np.broadcast_to(expected, paths.shape),
        rtol=1e-13,
        atol=1e-13,
    )


def test_zero_rates_zero_volatility_constant_spot(config, parameters):
    paths = simulate_fx_spot_paths(
        config,
        **{
            **parameters,
            "domestic_rate": 0.0,
            "foreign_rate": 0.0,
            "volatility": 0.0,
        },
    )

    np.testing.assert_array_equal(
        paths,
        np.full(paths.shape, 1.10),
    )


def test_antithetic_log_innovations(parameters):
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=1000,
        seed=42,
        antithetic=True,
    )

    paths = simulate_fx_spot_paths(config, **parameters)

    dt = np.asarray(config.time_grid.dt)
    drift = (
        parameters["domestic_rate"]
        - parameters["foreign_rate"]
        - 0.5 * parameters["volatility"] ** 2
    ) * dt

    innovations = np.diff(np.log(paths), axis=1) - drift[None, :]

    np.testing.assert_allclose(
        innovations[:500],
        -innovations[500:],
        rtol=0.0,
        atol=1e-13,
    )


def test_external_draws_reproducibility(config, parameters):
    draws = np.ones((config.n_paths, config.time_grid.n_steps))

    first = simulate_fx_spot_paths(config, **parameters, normal_draws=draws)
    second = simulate_fx_spot_paths(config, **parameters, normal_draws=draws)

    np.testing.assert_array_equal(first, second)


def test_external_zero_draws_give_drift_only(config, parameters):
    draws = np.zeros((config.n_paths, config.time_grid.n_steps))

    paths = simulate_fx_spot_paths(config, **parameters, normal_draws=draws)

    times = np.asarray(config.time_grid.times)
    drift = (
        parameters["domestic_rate"]
        - parameters["foreign_rate"]
        - 0.5 * parameters["volatility"] ** 2
    )

    expected = parameters["initial_spot"] * np.exp(drift * times)

    np.testing.assert_allclose(
        paths,
        np.broadcast_to(expected, paths.shape),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1000,),
        (1000, 11),
        (1000, 12, 1),
        (999, 12),
    ],
)
def test_invalid_external_draw_shapes(config, parameters, shape):
    with pytest.raises(ValueError, match="shape"):
        simulate_fx_spot_paths(
            config,
            **parameters,
            normal_draws=np.zeros(shape),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_external_draws(config, parameters, bad_value):
    draws = np.zeros((config.n_paths, config.time_grid.n_steps))
    draws[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        simulate_fx_spot_paths(config, **parameters, normal_draws=draws)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_spot", 0.0),
        ("initial_spot", -1.0),
        ("initial_spot", np.nan),
        ("domestic_rate", np.inf),
        ("foreign_rate", -np.inf),
        ("volatility", -0.1),
        ("volatility", np.nan),
    ],
)
def test_invalid_parameters(config, parameters, field, value):
    invalid = {**parameters, field: value}

    with pytest.raises(ValueError):
        simulate_fx_spot_paths(config, **invalid)


def test_invalid_config(parameters):
    with pytest.raises(TypeError, match="SimulationConfig"):
        simulate_fx_spot_paths(None, **parameters)


def test_risk_neutral_terminal_mean(parameters):
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=50000,
        seed=42,
        antithetic=True,
    )

    paths = simulate_fx_spot_paths(config, **parameters)

    expected = parameters["initial_spot"] * np.exp(
        (parameters["domestic_rate"] - parameters["foreign_rate"])
        * config.time_grid.horizon
    )

    observed = float(np.mean(paths[:, -1]))

    assert abs(observed / expected - 1.0) < 0.005
