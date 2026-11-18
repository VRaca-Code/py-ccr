"""Tests for Monte Carlo mean confidence intervals and FX paths."""

import numpy as np
import pytest
from scipy.stats import t

from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


def test_ordinary_mean_and_standard_error():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = monte_carlo_mean_interval(values)

    expected_se = np.std(values, ddof=1) / np.sqrt(4)
    critical = t.ppf(0.975, df=3)

    assert result.estimate == pytest.approx(2.5)
    assert result.standard_error == pytest.approx(expected_se)
    assert result.lower_bound == pytest.approx(2.5 - critical * expected_se)
    assert result.upper_bound == pytest.approx(2.5 + critical * expected_se)
    assert result.n_independent_units == 4
    assert not result.antithetic


def test_antithetic_pairs_are_independent_units():
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 7.0, 5.0, 3.0])
    result = monte_carlo_mean_interval(values, antithetic=True)

    assert result.estimate == pytest.approx(5.0)
    assert result.standard_error == pytest.approx(0.0)
    assert result.n_paths == 8
    assert result.n_independent_units == 4
    assert result.antithetic


def test_antithetic_pair_ordering_is_first_half_second_half():
    values = np.array([1.0, 4.0, 2.0, 8.0])
    result = monte_carlo_mean_interval(values, antithetic=True)
    pair_means = np.array([1.5, 6.0])

    assert result.estimate == pytest.approx(np.mean(pair_means))
    assert result.standard_error == pytest.approx(
        np.std(pair_means, ddof=1) / np.sqrt(2)
    )


def test_constant_observations_have_zero_width():
    result = monte_carlo_mean_interval(np.full(20, 2.5))
    assert result.estimate == 2.5
    assert result.standard_error == 0.0
    assert result.lower_bound == 2.5
    assert result.upper_bound == 2.5


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [1.0],
        [[1.0, 2.0]],
        [1.0, float("nan")],
        [1.0, float("inf")],
    ],
)
def test_invalid_observations(invalid):
    with pytest.raises(ValueError):
        monte_carlo_mean_interval(invalid)


@pytest.mark.parametrize(
    "invalid",
    [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_confidence_levels(invalid):
    with pytest.raises(ValueError, match="confidence_level"):
        monte_carlo_mean_interval([1.0, 2.0], confidence_level=invalid)


@pytest.mark.parametrize(
    "invalid",
    [True, False, "abc"],
)
def test_invalid_confidence_level_types(invalid):
    with pytest.raises(TypeError, match="confidence_level"):
        monte_carlo_mean_interval([1.0, 2.0], confidence_level=invalid)


def test_odd_antithetic_path_count_rejected():
    with pytest.raises(ValueError, match="antithetic"):
        monte_carlo_mean_interval(
            [1.0, 2.0, 3.0],
            antithetic=True,
        )


def test_two_antithetic_paths_rejected():
    with pytest.raises(ValueError, match="antithetic"):
        monte_carlo_mean_interval(
            [1.0, 2.0],
            antithetic=True,
        )


def test_nonboolean_antithetic_flag_rejected():
    with pytest.raises(TypeError, match="antithetic"):
        monte_carlo_mean_interval(
            [1.0, 2.0, 3.0, 4.0],
            antithetic="yes",
        )


@pytest.mark.parametrize(
    "n_paths",
    [512, 2048, 8192],
)
def test_fx_terminal_mean_matches_analytic_expectation(n_paths):
    initial_spot = 1.10
    domestic_rate = 0.04
    foreign_rate = 0.02
    volatility = 0.15
    horizon = 1.0

    config = SimulationConfig(
        time_grid=TimeGrid.uniform(horizon, 12),
        n_paths=n_paths,
        seed=20261118,
    )
    terminal = simulate_fx_spot_paths(
        config,
        initial_spot=initial_spot,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        volatility=volatility,
    )[:, -1]

    result = monte_carlo_mean_interval(terminal)
    analytic = initial_spot * np.exp((domestic_rate - foreign_rate) * horizon)

    assert result.n_paths == n_paths
    assert result.n_independent_units == n_paths
    assert result.standard_error > 0

    # A wide, pre-specified stochastic tolerance prevents
    # asserting monotonic convergence for individual seeds.
    assert abs(result.estimate - analytic) < (6.0 * result.standard_error)


def test_fx_seed_reproducibility_of_uncertainty():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=1024,
        seed=314159,
    )
    params = {
        "initial_spot": 1.10,
        "domestic_rate": 0.04,
        "foreign_rate": 0.02,
        "volatility": 0.15,
    }

    first = monte_carlo_mean_interval(simulate_fx_spot_paths(config, **params)[:, -1])
    second = monte_carlo_mean_interval(simulate_fx_spot_paths(config, **params)[:, -1])

    assert first == second


def test_fx_antithetic_pairs_use_half_as_many_units():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=2048,
        seed=42,
        antithetic=True,
    )
    terminal = simulate_fx_spot_paths(
        config,
        initial_spot=1.10,
        domestic_rate=0.04,
        foreign_rate=0.02,
        volatility=0.15,
    )[:, -1]

    result = monte_carlo_mean_interval(terminal, antithetic=True)

    assert result.n_paths == 2048
    assert result.n_independent_units == 1024
    assert result.standard_error > 0
    assert result.lower_bound < result.upper_bound
