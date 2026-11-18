"""Tests for pointwise PFE bootstrap uncertainty."""

import numpy as np
import pytest

from pyccr.convergence.pfe import pfe_bootstrap_intervals
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


@pytest.fixture
def sample():
    rng = np.random.default_rng(314159)
    times = np.array([0.0, 0.25, 0.75, 1.0])
    values = rng.normal(size=(128, 4))
    values[:, 0] = 0.0
    return times, values


def test_pfe_matches_existing_exposure_profiles(sample):
    times, values = sample
    result = pfe_bootstrap_intervals(times, values, n_resamples=120, seed=11)
    profiles = calculate_exposure_profiles(times, values, confidence_level=0.95)

    np.testing.assert_array_equal(result.pfe, profiles.pfe)
    np.testing.assert_array_equal(result.times, times)
    assert result.bootstrap_quantiles.shape == (120, 4)
    assert result.lower_bounds.shape == (4,)
    assert result.upper_bounds.shape == (4,)
    assert result.n_paths == 128
    assert result.n_independent_units == 128


def test_reproducibility_with_fixed_seed(sample):
    times, values = sample
    first = pfe_bootstrap_intervals(times, values, n_resamples=120, seed=7)
    second = pfe_bootstrap_intervals(times, values, n_resamples=120, seed=7)

    np.testing.assert_array_equal(
        first.bootstrap_quantiles,
        second.bootstrap_quantiles,
    )
    np.testing.assert_array_equal(first.lower_bounds, second.lower_bounds)


def test_percentile_bounds_match_bootstrap_distribution(sample):
    times, values = sample
    result = pfe_bootstrap_intervals(
        times,
        values,
        interval_confidence_level=0.90,
        n_resamples=120,
    )

    np.testing.assert_allclose(
        result.lower_bounds,
        np.quantile(result.bootstrap_quantiles, 0.05, axis=0),
    )
    np.testing.assert_allclose(
        result.upper_bounds,
        np.quantile(result.bootstrap_quantiles, 0.95, axis=0),
    )


def test_zero_exposure_has_degenerate_intervals():
    result = pfe_bootstrap_intervals(
        [0.0, 1.0],
        np.full((32, 2), -5.0),
        n_resamples=100,
    )
    np.testing.assert_array_equal(result.pfe, np.zeros(2))
    np.testing.assert_array_equal(result.lower_bounds, np.zeros(2))
    np.testing.assert_array_equal(result.upper_bounds, np.zeros(2))


def test_single_observation_time():
    result = pfe_bootstrap_intervals(
        [0.0],
        np.arange(20.0).reshape(20, 1),
        n_resamples=100,
    )
    assert result.pfe.shape == (1,)
    assert result.bootstrap_quantiles.shape == (100, 1)


def test_antithetic_resamples_whole_pairs():
    # Every matched pair has the same average, but PFE depends
    # on the complete distribution of both path values.
    values = np.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 9.0],
            [0.0, 8.0],
            [0.0, 7.0],
            [0.0, 6.0],
        ]
    )
    result = pfe_bootstrap_intervals(
        [0.0, 1.0],
        values,
        pfe_confidence_level=0.75,
        n_resamples=100,
        seed=9,
        antithetic=True,
    )

    assert result.n_paths == 8
    assert result.n_independent_units == 4
    assert result.antithetic

    # Reconstruct the bootstrap's first replicate independently.
    rng = np.random.default_rng(9)
    selected = rng.integers(0, 4, size=4)
    indices = np.concatenate((selected, selected + 4))
    expected = np.quantile(
        np.maximum(values[indices], 0.0),
        0.75,
        axis=0,
    )
    np.testing.assert_array_equal(result.bootstrap_quantiles[0], expected)


@pytest.mark.parametrize(
    "bad_count",
    [1, 3],
)
def test_invalid_antithetic_path_counts(bad_count):
    with pytest.raises(ValueError, match="antithetic"):
        pfe_bootstrap_intervals(
            [0.0],
            np.ones((bad_count, 1)),
            n_resamples=100,
            antithetic=True,
        )


def test_single_path_rejected():
    with pytest.raises(ValueError, match="at least two"):
        pfe_bootstrap_intervals([0.0], [[1.0]], n_resamples=100)


@pytest.mark.parametrize(
    "value",
    [0, 1, -0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_pfe_confidence_level(value):
    with pytest.raises(ValueError, match="pfe_confidence_level"):
        pfe_bootstrap_intervals(
            [0.0],
            [[1.0], [2.0]],
            pfe_confidence_level=value,
            n_resamples=100,
        )


@pytest.mark.parametrize(
    "value",
    [0, 1, -0.1, 1.1, float("nan")],
)
def test_invalid_interval_confidence_level(value):
    with pytest.raises(ValueError, match="interval_confidence_level"):
        pfe_bootstrap_intervals(
            [0.0],
            [[1.0], [2.0]],
            interval_confidence_level=value,
            n_resamples=100,
        )


@pytest.mark.parametrize(
    "name",
    ["pfe_confidence_level", "interval_confidence_level"],
)
def test_boolean_confidence_level_rejected(name):
    with pytest.raises(TypeError, match=name):
        pfe_bootstrap_intervals(
            [0.0],
            [[1.0], [2.0]],
            n_resamples=100,
            **{name: True},
        )


@pytest.mark.parametrize(
    "value",
    [0, 99, -1, 100.0, True],
)
def test_invalid_resample_count(value):
    with pytest.raises(ValueError, match="n_resamples"):
        pfe_bootstrap_intervals(
            [0.0],
            [[1.0], [2.0]],
            n_resamples=value,
        )


@pytest.mark.parametrize(
    "value",
    [-1, 1.5, True],
)
def test_invalid_seed(value):
    with pytest.raises(ValueError, match="seed"):
        pfe_bootstrap_intervals(
            [0.0],
            [[1.0], [2.0]],
            n_resamples=100,
            seed=value,
        )


def test_nonboolean_antithetic_rejected():
    with pytest.raises(TypeError, match="antithetic"):
        pfe_bootstrap_intervals(
            [0.0],
            [[1.0], [2.0]],
            n_resamples=100,
            antithetic="yes",
        )


def test_fx_simulation_pfe_integration():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=512,
        seed=20261118,
    )
    spots = simulate_fx_spot_paths(
        config,
        initial_spot=1.10,
        domestic_rate=0.04,
        foreign_rate=0.02,
        volatility=0.15,
    )
    signed_values = 1_000_000.0 * (spots - 1.10)

    result = pfe_bootstrap_intervals(
        config.time_grid.times,
        signed_values,
        n_resamples=150,
        seed=42,
    )
    profiles = calculate_exposure_profiles(
        config.time_grid.times,
        signed_values,
    )

    np.testing.assert_array_equal(result.pfe, profiles.pfe)
    assert result.n_paths == 512
    assert result.bootstrap_quantiles.shape == (150, 9)
    assert result.pfe[0] == 0.0
    assert result.lower_bounds[0] == 0.0
    assert result.upper_bounds[0] == 0.0
    assert result.pfe[-1] > 0.0
    assert np.all(result.lower_bounds <= result.upper_bounds)


def test_fx_antithetic_pfe_integration():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=512,
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
    result = pfe_bootstrap_intervals(
        config.time_grid.times,
        spots - 1.10,
        n_resamples=120,
        seed=7,
        antithetic=True,
    )

    assert result.n_paths == 512
    assert result.n_independent_units == 256
    assert result.bootstrap_quantiles.shape == (120, 9)
    assert np.all(np.isfinite(result.bootstrap_quantiles))
