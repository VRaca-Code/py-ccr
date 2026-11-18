"""Tests for nested Monte Carlo path-count diagnostics."""

import numpy as np
import pytest

from pyccr.convergence.exposure import exposure_mean_intervals
from pyccr.convergence.path_count import path_count_convergence
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


def test_matches_individual_exposure_estimators():
    times = [0.0, 0.5, 1.0]
    values = np.array(
        [
            [0.0, 1.0, 3.0],
            [0.0, 2.0, -1.0],
            [0.0, 4.0, 2.0],
            [0.0, -2.0, 5.0],
            [0.0, 3.0, 1.0],
            [0.0, 5.0, 4.0],
        ]
    )
    result = path_count_convergence(times, values, [2, 4, 6])

    for i, count in enumerate((2, 4, 6)):
        means = exposure_mean_intervals(times, values[:count])
        profiles = calculate_exposure_profiles(times, values[:count])
        np.testing.assert_allclose(result.ee[i], means.ee)
        np.testing.assert_allclose(
            result.ee_standard_errors[i],
            means.ee_standard_errors,
        )
        np.testing.assert_allclose(result.pfe[i], profiles.pfe)
        assert result.epe[i] == pytest.approx(profiles.epe)
        assert result.epe_standard_errors[i] == pytest.approx(means.epe.standard_error)

    np.testing.assert_array_equal(result.independent_unit_counts, [2, 4, 6])
    assert result.ee.shape == (3, 3)


def test_antithetic_prefix_selects_complete_pairs():
    # Matched pairs are (0,4), (1,5), (2,6), (3,7).
    times = [0.0, 1.0]
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
    result = path_count_convergence(times, values, [4, 6, 8], antithetic=True)

    for i, count in enumerate((4, 6, 8)):
        half = count // 2
        selected = np.concatenate((values[:half], values[4 : 4 + half]))
        expected = exposure_mean_intervals(times, selected, antithetic=True)
        np.testing.assert_allclose(result.ee[i], expected.ee)
        np.testing.assert_allclose(
            result.ee_standard_errors[i],
            expected.ee_standard_errors,
        )

    np.testing.assert_array_equal(result.independent_unit_counts, [2, 3, 4])


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [[2, 4]],
        [2, 2],
        [4, 2],
        [1, 4],
        [2, 100],
    ],
)
def test_invalid_path_counts(counts):
    with pytest.raises(ValueError, match="path_counts"):
        path_count_convergence(
            [0.0, 1.0],
            np.ones((8, 2)),
            counts,
        )


@pytest.mark.parametrize(
    "counts",
    [
        [2.0, 4.0],
        [2.5, 4.5],
        [True, False],
    ],
)
def test_noninteger_counts_rejected(counts):
    with pytest.raises(TypeError, match="integers"):
        path_count_convergence(
            [0.0, 1.0],
            np.ones((8, 2)),
            counts,
        )


@pytest.mark.parametrize(
    "counts",
    [[3, 4], [4, 5], [2, 8]],
)
def test_invalid_antithetic_counts(counts):
    with pytest.raises(ValueError, match="antithetic"):
        path_count_convergence(
            [0.0, 1.0],
            np.ones((8, 2)),
            counts,
            antithetic=True,
        )


def test_antithetic_total_must_be_even():
    with pytest.raises(ValueError, match="antithetic"):
        path_count_convergence(
            [0.0, 1.0],
            np.ones((7, 2)),
            [4, 6],
            antithetic=True,
        )


def test_nonboolean_antithetic_rejected():
    with pytest.raises(TypeError, match="antithetic"):
        path_count_convergence(
            [0.0, 1.0],
            np.ones((8, 2)),
            [4, 8],
            antithetic="yes",
        )


def test_nonfinite_portfolio_values_rejected():
    values = np.ones((8, 2))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        path_count_convergence([0.0, 1.0], values, [4, 8])


def test_input_not_modified():
    values = np.arange(24.0).reshape(8, 3)
    original = values.copy()
    path_count_convergence([0.0, 0.5, 1.0], values, [4, 8])
    np.testing.assert_array_equal(values, original)


def test_fx_simulator_path_count_convergence():
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
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
    values = 1_000_000.0 * (spots - 1.10)
    result = path_count_convergence(
        config.time_grid.times,
        values,
        [128, 512, 2048],
    )

    assert result.ee.shape == (3, 9)
    assert result.pfe.shape == (3, 9)
    assert result.epe.shape == (3,)
    assert np.all(result.ee[:, 0] == 0.0)
    assert np.all(result.ee_standard_errors[:, 0] == 0.0)
    assert np.all(result.ee_standard_errors[:, -1] > 0.0)
    assert np.all(np.isfinite(result.pfe))


def test_fx_antithetic_path_count_convergence():
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
    result = path_count_convergence(
        config.time_grid.times,
        spots - 1.10,
        [128, 512, 1024],
        antithetic=True,
    )

    np.testing.assert_array_equal(
        result.independent_unit_counts,
        [64, 256, 512],
    )
    assert result.ee.shape == (3, 9)
    assert np.all(np.isfinite(result.epe))
    assert np.all(result.epe_standard_errors >= 0)
