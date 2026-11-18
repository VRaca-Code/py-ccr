"""Analytical cross-asset correlation regression tests.

The current joint FX/Hull-White simulator shares correlated
Gaussian innovations but uses a constant-rate FX drift.

Terminal correlation is tested between log-FX and the
stochastic Hull-White short-rate state, not between FX
spot and the short rate.

The exact OU transition uses normalized step innovations;
their correlation differs from the correlation of continuous
Brownian increments when considering a terminal state.
"""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.joint_fx_hw_paths import (
    simulate_joint_fx_hull_white_paths,
)
from pyccr.simulation.time_grid import TimeGrid


def model(mean_reversion=0.3, volatility=0.012):
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0),
        zero_rates=(0.025, 0.026, 0.027, 0.028, 0.03),
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=mean_reversion,
        volatility=volatility,
    )


def simulate(
    *,
    correlation,
    mean_reversion=0.3,
    antithetic=False,
    grid=None,
    n_paths=8192,
    seed=457001,
):
    if grid is None:
        grid = TimeGrid.uniform(1.0, 24)

    config = SimulationConfig(
        time_grid=grid,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    hw_model = model(mean_reversion=mean_reversion)

    result = simulate_joint_fx_hull_white_paths(
        config,
        hw_model,
        initial_spot=1.1,
        domestic_rate=0.03,
        foreign_rate=0.01,
        fx_volatility=0.2,
        correlation=correlation,
    )

    return result, hw_model


def theoretical_terminal_correlation(
    rho,
    mean_reversion,
    horizon,
):
    a = mean_reversion
    integrated_kernel = -np.expm1(-a * horizon) / a
    integrated_kernel_squared = -np.expm1(-2.0 * a * horizon) / (2.0 * a)

    return rho * integrated_kernel / np.sqrt(horizon * integrated_kernel_squared)


def discrete_terminal_correlation(
    rho,
    mean_reversion,
    times,
):
    """Exact correlation implied by the implemented grid scheme.

    FX uses sqrt(dt)*Z_fx; Hull-White uses exact OU
    transition standard deviations times Z_hw.

    Because the simulator correlates these normalized
    step innovations, its finite-grid terminal covariance
    is a discrete sum, not the continuous-time integral.
    """
    a = mean_reversion
    dt = np.diff(times)
    horizon = times[-1]

    variance_steps = -np.expm1(-2.0 * a * dt) / (2.0 * a)
    rate_weights = np.exp(-a * (horizon - times[1:])) * np.sqrt(variance_steps)
    fx_weights = np.sqrt(dt)

    return (
        rho
        * np.dot(fx_weights, rate_weights)
        / np.sqrt(np.dot(fx_weights, fx_weights) * np.dot(rate_weights, rate_weights))
    )


@pytest.mark.parametrize(
    "rho",
    [-0.8, -0.4, 0.0, 0.4, 0.8],
)
@pytest.mark.parametrize("antithetic", [False, True])
def test_terminal_log_fx_rate_correlation(
    rho,
    antithetic,
):
    result, hw_model = simulate(
        correlation=rho,
        antithetic=antithetic,
        seed=457100 + int((rho + 1) * 100) + (10000 if antithetic else 0),
    )

    observed = float(
        np.corrcoef(
            np.log(result.fx_spots[:, -1]),
            result.short_rates[:, -1],
        )[0, 1]
    )

    expected = discrete_terminal_correlation(
        rho,
        hw_model.mean_reversion,
        result.times,
    )

    assert abs(observed - expected) < 0.045


@pytest.mark.parametrize(
    "mean_reversion",
    [0.05, 0.3, 1.5],
)
def test_mean_reversion_changes_terminal_correlation(
    mean_reversion,
):
    result, hw_model = simulate(
        correlation=0.8,
        mean_reversion=mean_reversion,
        seed=457200 + int(mean_reversion * 100),
    )

    observed = float(
        np.corrcoef(
            np.log(result.fx_spots[:, -1]),
            result.short_rates[:, -1],
        )[0, 1]
    )

    expected = discrete_terminal_correlation(
        0.8,
        hw_model.mean_reversion,
        result.times,
    )

    assert abs(observed - expected) < 0.045


def test_nonuniform_grid_correlation():
    grid = TimeGrid((0.0, 0.02, 0.1, 0.25, 0.55, 1.0))

    result, hw_model = simulate(
        correlation=-0.7,
        mean_reversion=1.2,
        grid=grid,
        seed=457301,
        n_paths=16384,
    )

    observed = float(
        np.corrcoef(
            np.log(result.fx_spots[:, -1]),
            result.short_rates[:, -1],
        )[0, 1]
    )

    expected = discrete_terminal_correlation(
        -0.7,
        hw_model.mean_reversion,
        result.times,
    )

    assert abs(observed - expected) < 0.035


@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.7])
def test_continuous_and_fine_grid_benchmarks_agree(rho):
    times = np.linspace(0.0, 1.0, 1001)

    discrete = discrete_terminal_correlation(
        rho,
        0.4,
        times,
    )

    continuous = theoretical_terminal_correlation(
        rho,
        0.4,
        1.0,
    )

    np.testing.assert_allclose(
        discrete,
        continuous,
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_recovered_shock_correlation(
    antithetic,
):
    rho = -0.65

    result, hw_model = simulate(
        correlation=rho,
        antithetic=antithetic,
        n_paths=4096,
        seed=457400 + int(antithetic),
    )

    times = result.times
    dt = np.diff(times)
    a = hw_model.mean_reversion
    sigma = hw_model.volatility

    forward = np.asarray(
        hw_model.curve.instantaneous_forward(times),
        dtype=float,
    )
    shift = forward + 0.5 * (sigma / a) ** 2 * (-np.expm1(-a * times)) ** 2

    state = result.short_rates - shift[None, :]
    decay = np.exp(-a * dt)
    variance = sigma**2 * -np.expm1(-2.0 * a * dt) / (2.0 * a)

    hw_normals = (state[:, 1:] - decay[None, :] * state[:, :-1]) / np.sqrt(variance)[
        None, :
    ]

    fx_returns = np.diff(
        np.log(result.fx_spots),
        axis=1,
    )

    fx_drift = (0.03 - 0.01 - 0.5 * 0.2**2) * dt

    fx_normals = (fx_returns - fx_drift[None, :]) / (0.2 * np.sqrt(dt)[None, :])

    observed = float(
        np.corrcoef(
            hw_normals.ravel(),
            fx_normals.ravel(),
        )[0, 1]
    )

    assert abs(observed - rho) < 0.025

    if antithetic:
        half = hw_normals.shape[0] // 2

        np.testing.assert_allclose(
            hw_normals[:half] + hw_normals[half:],
            0.0,
            atol=1e-11,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            fx_normals[:half] + fx_normals[half:],
            0.0,
            atol=1e-11,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "rho",
    [-1.0, 1.0, -1.01, 1.01, np.nan, np.inf],
)
def test_invalid_joint_correlations_rejected(rho):
    with pytest.raises(ValueError, match="correlation"):
        simulate(correlation=rho)


def test_zero_rate_volatility_is_deterministic():
    grid = TimeGrid.uniform(1.0, 12)
    config = SimulationConfig(
        time_grid=grid,
        n_paths=256,
        seed=457500,
    )

    hw_model = model(volatility=0.0)

    result = simulate_joint_fx_hull_white_paths(
        config,
        hw_model,
        initial_spot=1.1,
        domestic_rate=0.03,
        foreign_rate=0.01,
        fx_volatility=0.2,
        correlation=0.7,
    )

    expected_rates = hw_model.curve.instantaneous_forward(np.asarray(grid.times))

    np.testing.assert_allclose(
        result.short_rates,
        np.broadcast_to(
            expected_rates,
            result.short_rates.shape,
        ),
        rtol=0.0,
        atol=1e-12,
    )

    # Terminal correlation is undefined when the rate
    # factor has zero variance. Do not calculate it.
    assert np.all(result.fx_spots > 0.0)
