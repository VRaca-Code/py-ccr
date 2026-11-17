"""Regression tests for joint FX and Hull-White scenarios."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.brownian import brownian_increments
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.hull_white_paths import (
    simulate_hull_white_short_rates,
)
from pyccr.simulation.joint_fx_hw_paths import (
    JointFXHullWhitePaths,
    simulate_joint_fx_hull_white_paths,
)
from pyccr.simulation.time_grid import TimeGrid


def make_config(
    *,
    n_paths=100,
    seed=42,
    antithetic=False,
):
    return SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )


def make_model():
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0),
        zero_rates=(0.03,) * 5,
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.10,
        volatility=0.01,
    )


def simulate(config=None, **overrides):
    if config is None:
        config = make_config()

    parameters = {
        "initial_spot": 1.10,
        "domestic_rate": 0.03,
        "foreign_rate": 0.02,
        "fx_volatility": 0.15,
        "correlation": 0.45,
    }
    parameters.update(overrides)

    return simulate_joint_fx_hull_white_paths(
        config,
        make_model(),
        **parameters,
    )


def test_returns_expected_container():
    result = simulate()
    assert isinstance(result, JointFXHullWhitePaths)


def test_output_shapes():
    result = simulate()
    assert result.times.shape == (13,)
    assert result.short_rates.shape == (100, 13)
    assert result.fx_spots.shape == (100, 13)


def test_time_grid_preserved():
    config = make_config()
    result = simulate(config)
    np.testing.assert_array_equal(
        result.times,
        np.asarray(config.time_grid.times),
    )


def test_initial_fx_spot():
    result = simulate()
    np.testing.assert_array_equal(
        result.fx_spots[:, 0],
        np.full(100, 1.10),
    )


def test_initial_short_rate():
    result = simulate()
    np.testing.assert_allclose(
        result.short_rates[:, 0],
        0.03,
        rtol=0.0,
        atol=1e-12,
    )


def test_paths_are_finite():
    result = simulate()
    assert np.all(np.isfinite(result.short_rates))
    assert np.all(np.isfinite(result.fx_spots))


def test_fx_spots_are_positive():
    result = simulate()
    assert np.all(result.fx_spots > 0.0)


def test_seeded_reproducibility():
    first = simulate()
    second = simulate()

    np.testing.assert_array_equal(
        first.short_rates,
        second.short_rates,
    )
    np.testing.assert_array_equal(
        first.fx_spots,
        second.fx_spots,
    )


def test_different_seeds_produce_different_paths():
    first = simulate(make_config(seed=42))
    second = simulate(make_config(seed=43))

    assert not np.array_equal(
        first.short_rates,
        second.short_rates,
    )
    assert not np.array_equal(
        first.fx_spots,
        second.fx_spots,
    )


@pytest.mark.parametrize(
    "rho",
    [-0.75, -0.25, 0.0, 0.25, 0.75],
)
def test_valid_correlations(rho):
    result = simulate(correlation=rho)
    assert result.short_rates.shape == (100, 13)
    assert result.fx_spots.shape == (100, 13)


@pytest.mark.parametrize(
    "rho",
    [-1.0, 1.0, -1.01, 1.01, np.nan, np.inf, -np.inf],
)
def test_invalid_correlations(rho):
    with pytest.raises(ValueError, match="correlation"):
        simulate(correlation=rho)


def test_antithetic_shapes():
    result = simulate(make_config(n_paths=200, antithetic=True))
    assert result.short_rates.shape == (200, 13)
    assert result.fx_spots.shape == (200, 13)


def test_antithetic_seeded_reproducibility():
    config = make_config(n_paths=200, antithetic=True)

    first = simulate(config)
    second = simulate(config)

    np.testing.assert_array_equal(
        first.short_rates,
        second.short_rates,
    )
    np.testing.assert_array_equal(
        first.fx_spots,
        second.fx_spots,
    )


def test_antithetic_fx_log_return_pairing():
    config = make_config(n_paths=200, antithetic=True)
    result = simulate(config)

    dt = np.asarray(config.time_grid.dt)
    drift = (0.03 - 0.02 - 0.5 * 0.15**2) * dt

    returns = np.diff(np.log(result.fx_spots), axis=1)
    innovations = returns - drift[None, :]

    half = config.n_paths // 2

    np.testing.assert_allclose(
        innovations[:half] + innovations[half:],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_consistency_with_standalone_simulators():
    config = make_config(n_paths=200, antithetic=True)
    model = make_model()
    rho = 0.45

    result = simulate_joint_fx_hull_white_paths(
        config,
        model,
        initial_spot=1.10,
        domestic_rate=0.03,
        foreign_rate=0.02,
        fx_volatility=0.15,
        correlation=rho,
    )

    correlation_matrix = np.array([[1.0, rho], [rho, 1.0]])

    increments = brownian_increments(
        config,
        correlation_matrix,
    )

    dt = np.asarray(config.time_grid.dt)
    normals = increments / np.sqrt(dt)[None, :, None]

    expected_rates = simulate_hull_white_short_rates(
        config,
        model,
        normal_draws=normals[:, :, 0],
    )

    expected_fx = simulate_fx_spot_paths(
        config,
        initial_spot=1.10,
        domestic_rate=0.03,
        foreign_rate=0.02,
        volatility=0.15,
        normal_draws=normals[:, :, 1],
    )

    np.testing.assert_array_equal(
        result.short_rates,
        expected_rates,
    )
    np.testing.assert_array_equal(
        result.fx_spots,
        expected_fx,
    )


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.6])
def test_empirical_shock_correlation(rho):
    config = make_config(
        n_paths=20_000,
        antithetic=True,
    )
    result = simulate(config, correlation=rho)

    dt = np.asarray(config.time_grid.dt)
    times = np.asarray(config.time_grid.times)

    a = 0.10
    sigma = 0.01

    shift = 0.03 + 0.5 * (sigma / a) ** 2 * (-np.expm1(-a * times)) ** 2

    state = result.short_rates - shift[None, :]

    decay = np.exp(-a * dt)
    variance = sigma**2 * -np.expm1(-2.0 * a * dt) / (2.0 * a)

    rate_normals = (state[:, 1:] - decay[None, :] * state[:, :-1]) / np.sqrt(variance)[
        None, :
    ]

    fx_returns = np.diff(
        np.log(result.fx_spots),
        axis=1,
    )

    fx_drift = (0.03 - 0.02 - 0.5 * 0.15**2) * dt

    fx_normals = (fx_returns - fx_drift[None, :]) / (0.15 * np.sqrt(dt)[None, :])

    observed = np.corrcoef(
        rate_normals.ravel(),
        fx_normals.ravel(),
    )[0, 1]

    assert abs(observed - rho) < 0.02


def test_zero_fx_volatility():
    result = simulate(fx_volatility=0.0)

    expected = 1.10 * np.exp((0.03 - 0.02) * result.times)

    np.testing.assert_allclose(
        result.fx_spots,
        np.broadcast_to(
            expected,
            result.fx_spots.shape,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_invalid_initial_spot():
    with pytest.raises(ValueError):
        simulate(initial_spot=0.0)


def test_invalid_fx_volatility():
    with pytest.raises(ValueError):
        simulate(fx_volatility=-0.1)
