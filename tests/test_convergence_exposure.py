"""Tests for Monte Carlo EE and EPE uncertainty."""

import numpy as np
import pytest

from pyccr.convergence.exposure import exposure_mean_intervals
from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


def test_ee_and_epe_match_existing_exposure_profiles():
    times = np.array([0.0, 0.25, 1.0, 2.0])
    values = np.array(
        [
            [-2.0, 1.0, 3.0, 4.0],
            [1.0, -1.0, 0.0, 5.0],
            [2.0, 3.0, -1.0, 1.0],
            [-1.0, 2.0, 4.0, -2.0],
        ]
    )

    profiles = calculate_exposure_profiles(times, values)
    result = exposure_mean_intervals(times, values)

    np.testing.assert_allclose(result.times, times)
    np.testing.assert_allclose(result.ee, profiles.ee, rtol=0, atol=1e-14)
    assert result.epe.estimate == pytest.approx(profiles.epe, abs=1e-14)
    assert result.n_paths == 4
    assert result.n_independent_units == 4
    assert result.ee_standard_errors.shape == times.shape


def test_each_ee_interval_matches_independent_mean_estimator():
    times = [0.0, 0.5, 1.0]
    values = np.array(
        [
            [0.0, -2.0, 4.0],
            [1.0, 3.0, -1.0],
            [2.0, 1.0, 5.0],
            [-3.0, 4.0, 2.0],
        ]
    )
    result = exposure_mean_intervals(times, values)

    for index in range(len(times)):
        expected = monte_carlo_mean_interval(np.maximum(values[:, index], 0.0))
        assert result.ee[index] == pytest.approx(expected.estimate)
        assert result.ee_standard_errors[index] == pytest.approx(
            expected.standard_error
        )
        assert result.ee_lower_bounds[index] == pytest.approx(expected.lower_bound)
        assert result.ee_upper_bounds[index] == pytest.approx(expected.upper_bound)


def test_epe_uses_pathwise_integration_not_average_of_errors():
    times = np.array([0.0, 0.5, 1.0])
    values = np.array(
        [
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 2.0, 2.0],
            [0.0, 1.0, 3.0],
        ]
    )
    result = exposure_mean_intervals(times, values)

    # Trapezoidal weights are [0.25, 0.50, 0.25].
    pathwise = np.maximum(values, 0.0) @ np.array([0.25, 0.50, 0.25])
    expected = monte_carlo_mean_interval(pathwise)

    assert result.epe.estimate == pytest.approx(expected.estimate)
    assert result.epe.standard_error == pytest.approx(expected.standard_error)


def test_single_time_grid_uses_ee_as_epe():
    times = [0.0]
    values = np.array([[-2.0], [1.0], [3.0], [0.0]])

    result = exposure_mean_intervals(times, values)

    assert result.ee.shape == (1,)
    assert result.epe.estimate == pytest.approx(result.ee[0])
    assert result.epe.standard_error == pytest.approx(result.ee_standard_errors[0])


def test_constant_positive_exposure_has_zero_uncertainty():
    times = [0.0, 0.5, 1.0]
    values = np.full((20, 3), 5.0)

    result = exposure_mean_intervals(times, values)

    np.testing.assert_array_equal(result.ee, np.full(3, 5.0))
    np.testing.assert_array_equal(result.ee_standard_errors, np.zeros(3))
    assert result.epe.estimate == pytest.approx(5.0)
    assert result.epe.standard_error == pytest.approx(0.0)


def test_antithetic_pairs_are_used_for_ee_and_epe():
    times = [0.0, 0.5, 1.0]
    values = np.array(
        [
            [0.0, 2.0, 4.0],
            [0.0, 6.0, 8.0],
            [0.0, 4.0, 2.0],
            [0.0, 8.0, 6.0],
        ]
    )

    result = exposure_mean_intervals(times, values, antithetic=True)

    assert result.n_paths == 4
    assert result.n_independent_units == 2
    assert result.epe.n_independent_units == 2

    expected_terminal = monte_carlo_mean_interval(values[:, -1], antithetic=True)
    assert result.ee_standard_errors[-1] == pytest.approx(
        expected_terminal.standard_error
    )


def test_antithetic_requires_valid_pair_count():
    with pytest.raises(ValueError, match="antithetic"):
        exposure_mean_intervals(
            [0.0, 1.0],
            np.ones((3, 2)),
            antithetic=True,
        )


def test_rejects_one_path_for_uncertainty_estimation():
    with pytest.raises(ValueError, match="at least two"):
        exposure_mean_intervals(
            [0.0, 1.0],
            np.ones((1, 2)),
        )


def test_rejects_invalid_portfolio_shape():
    with pytest.raises(ValueError, match="shape"):
        exposure_mean_intervals(
            [0.0, 1.0],
            np.ones((4, 3)),
        )


def test_rejects_nonfinite_portfolio_values():
    values = np.ones((4, 2))
    values[0, 1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        exposure_mean_intervals([0.0, 1.0], values)


def test_fx_forward_like_exposure_uses_actual_simulator():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=2048,
        seed=20261118,
    )
    spots = simulate_fx_spot_paths(
        config,
        initial_spot=1.10,
        domestic_rate=0.04,
        foreign_rate=0.02,
        volatility=0.15,
    )

    # Illustrative signed mark-to-market, not a full
    # collateralized or discounted FX-forward valuation.
    signed_values = 1_000_000.0 * (spots - 1.10)

    result = exposure_mean_intervals(
        config.time_grid.times,
        signed_values,
    )
    profiles = calculate_exposure_profiles(
        config.time_grid.times,
        signed_values,
    )

    np.testing.assert_allclose(result.ee, profiles.ee, rtol=1e-13, atol=1e-8)
    assert result.epe.estimate == pytest.approx(profiles.epe, rel=1e-13)
    assert result.ee[0] == 0.0
    assert result.ee_standard_errors[0] == 0.0
    assert result.ee_standard_errors[-1] > 0
    assert result.epe.standard_error > 0


def test_fx_antithetic_exposure_units():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=1024,
        seed=42,
        antithetic=True,
    )
    spots = simulate_fx_spot_paths(
        config,
        initial_spot=1.10,
        domestic_rate=0.04,
        foreign_rate=0.02,
        volatility=0.15,
    )

    result = exposure_mean_intervals(
        config.time_grid.times,
        spots - 1.10,
        antithetic=True,
    )

    assert result.n_paths == 1024
    assert result.n_independent_units == 512
    assert result.epe.n_independent_units == 512
    assert result.epe.standard_error >= 0
    assert np.all(result.ee_lower_bounds <= result.ee_upper_bounds)
