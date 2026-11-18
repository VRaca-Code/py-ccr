"""Extreme-tail PFE validation for the standalone FX model.

Analytical targets are undiscounted positive exposures of a
long FX call under deterministic-rate lognormal dynamics.

Bootstrap tests verify implementation mechanics, not nominal
frequentist coverage at extreme quantiles. In particular,
percentile bootstrap cannot extrapolate beyond the sample.
"""

import numpy as np
import pytest
from scipy.stats import norm

from pyccr.convergence.pfe import pfe_bootstrap_intervals
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid

INITIAL_SPOT = 1.1
STRIKE = 1.1
DOMESTIC_RATE = 0.03
FOREIGN_RATE = 0.01
VOLATILITY = 0.2


def analytical_pfe(quantile, horizon=1.0):
    spot_quantile = INITIAL_SPOT * np.exp(
        (DOMESTIC_RATE - FOREIGN_RATE - 0.5 * VOLATILITY**2) * horizon
        + VOLATILITY * np.sqrt(horizon) * norm.ppf(quantile)
    )
    return max(float(spot_quantile - STRIKE), 0.0)


def fx_values(
    *,
    n_paths=8192,
    antithetic=False,
    seed=459001,
    volatility=VOLATILITY,
    normal_draws=None,
):
    grid = TimeGrid.uniform(horizon=1.0, n_steps=1)
    config = SimulationConfig(
        time_grid=grid,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    spots = simulate_fx_spot_paths(
        config,
        initial_spot=INITIAL_SPOT,
        domestic_rate=DOMESTIC_RATE,
        foreign_rate=FOREIGN_RATE,
        volatility=volatility,
        normal_draws=normal_draws,
    )
    return np.asarray(grid.times), spots - STRIKE


@pytest.mark.parametrize("quantile", [0.95, 0.99, 0.995])
@pytest.mark.parametrize("antithetic", [False, True])
def test_seeded_terminal_pfe_against_analytical_target(
    quantile,
    antithetic,
):
    times, values = fx_values(
        n_paths=16384,
        antithetic=antithetic,
        seed=459100 + int(quantile * 1000) + (10000 if antithetic else 0),
    )

    result = calculate_exposure_profiles(
        times,
        values,
        confidence_level=quantile,
    )

    target = analytical_pfe(quantile)

    # Fixed-seed regression tolerance, not a confidence
    # interval or a claim of exact tail calibration.
    assert abs(result.pfe[-1] - target) < 0.065
    assert result.pfe[0] == 0.0


@pytest.mark.parametrize("quantile", [0.99, 0.995])
@pytest.mark.parametrize("antithetic", [False, True])
def test_deterministic_normal_quantile_grid(
    quantile,
    antithetic,
):
    n_paths = 8192

    # Deterministic midpoint normal quantiles remove Monte
    # Carlo noise and independently check the FX transform.
    probabilities = (np.arange(n_paths, dtype=float) + 0.5) / n_paths

    normals = norm.ppf(probabilities)

    if antithetic:
        half = n_paths // 2
        first_half = normals[:half]
        normals = np.concatenate((first_half, -first_half))

    times, values = fx_values(
        n_paths=n_paths,
        antithetic=antithetic,
        normal_draws=normals[:, None],
    )

    result = calculate_exposure_profiles(
        times,
        values,
        confidence_level=quantile,
    )

    assert abs(result.pfe[-1] - analytical_pfe(quantile)) < 0.002

    assert result.pfe[0] == 0.0


@pytest.mark.parametrize("antithetic", [False, True])
def test_tail_quantiles_are_ordered(antithetic):
    times, values = fx_values(
        n_paths=4096,
        antithetic=antithetic,
    )

    estimates = [
        calculate_exposure_profiles(
            times,
            values,
            confidence_level=quantile,
        ).pfe[-1]
        for quantile in (0.95, 0.99, 0.995)
    ]

    assert estimates[0] <= estimates[1] <= estimates[2]


@pytest.mark.parametrize("quantile", [0.99, 0.995])
@pytest.mark.parametrize("antithetic", [False, True])
def test_bootstrap_matches_original_tail_estimator(
    quantile,
    antithetic,
):
    times, values = fx_values(
        n_paths=2048,
        antithetic=antithetic,
        seed=459200 + int(antithetic),
    )

    result = pfe_bootstrap_intervals(
        times,
        values,
        pfe_confidence_level=quantile,
        n_resamples=120,
        seed=459201,
        antithetic=antithetic,
    )

    expected = calculate_exposure_profiles(
        times,
        values,
        confidence_level=quantile,
    )

    np.testing.assert_array_equal(
        result.pfe,
        expected.pfe,
    )

    assert result.n_paths == 2048
    assert result.n_independent_units == (1024 if antithetic else 2048)
    assert result.bootstrap_quantiles.shape == (120, 2)
    assert np.all(np.isfinite(result.bootstrap_quantiles))
    assert np.all(result.lower_bounds <= result.upper_bounds)

    np.testing.assert_allclose(
        result.lower_bounds,
        np.quantile(
            result.bootstrap_quantiles,
            0.025,
            axis=0,
        ),
    )
    np.testing.assert_allclose(
        result.upper_bounds,
        np.quantile(
            result.bootstrap_quantiles,
            0.975,
            axis=0,
        ),
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_extreme_tail_bootstrap_is_reproducible(antithetic):
    times, values = fx_values(
        n_paths=512,
        antithetic=antithetic,
    )

    kwargs = {
        "pfe_confidence_level": 0.995,
        "n_resamples": 100,
        "seed": 459300,
        "antithetic": antithetic,
    }

    first = pfe_bootstrap_intervals(
        times,
        values,
        **kwargs,
    )
    second = pfe_bootstrap_intervals(
        times,
        values,
        **kwargs,
    )

    np.testing.assert_array_equal(
        first.bootstrap_quantiles,
        second.bootstrap_quantiles,
    )
    np.testing.assert_array_equal(
        first.lower_bounds,
        second.lower_bounds,
    )
    np.testing.assert_array_equal(
        first.upper_bounds,
        second.upper_bounds,
    )


@pytest.mark.parametrize("quantile", [0.99, 0.995])
def test_antithetic_first_bootstrap_replicate_uses_pairs(
    quantile,
):
    times, values = fx_values(
        n_paths=512,
        antithetic=True,
        seed=459400,
    )

    result = pfe_bootstrap_intervals(
        times,
        values,
        pfe_confidence_level=quantile,
        n_resamples=100,
        seed=459401,
        antithetic=True,
    )

    rng = np.random.default_rng(459401)
    half = 256
    selected = rng.integers(0, half, size=half)
    selected_paths = np.concatenate((selected, selected + half))

    expected = np.quantile(
        np.maximum(values[selected_paths], 0.0),
        quantile,
        axis=0,
    )

    np.testing.assert_array_equal(
        result.bootstrap_quantiles[0],
        expected,
    )


@pytest.mark.parametrize("quantile", [0.99, 0.995])
@pytest.mark.parametrize("antithetic", [False, True])
def test_bootstrap_cannot_extrapolate_past_sample_maximum(
    quantile,
    antithetic,
):
    times, values = fx_values(
        n_paths=256,
        antithetic=antithetic,
        seed=459500,
    )

    result = pfe_bootstrap_intervals(
        times,
        values,
        pfe_confidence_level=quantile,
        n_resamples=100,
        seed=459501,
        antithetic=antithetic,
    )

    sample_max = np.max(
        np.maximum(values, 0.0),
        axis=0,
    )

    assert np.all(result.bootstrap_quantiles <= sample_max[None, :] + 1e-12)
    assert np.all(result.upper_bounds <= sample_max + 1e-12)


@pytest.mark.parametrize("antithetic", [False, True])
def test_zero_volatility_tail_is_deterministic(antithetic):
    times, values = fx_values(
        n_paths=128,
        antithetic=antithetic,
        volatility=0.0,
    )

    result = pfe_bootstrap_intervals(
        times,
        values,
        pfe_confidence_level=0.995,
        n_resamples=100,
        seed=459600,
        antithetic=antithetic,
    )

    terminal = max(
        INITIAL_SPOT * np.exp(DOMESTIC_RATE - FOREIGN_RATE) - STRIKE,
        0.0,
    )

    np.testing.assert_allclose(
        result.pfe,
        [0.0, terminal],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.lower_bounds,
        result.pfe,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.upper_bounds,
        result.pfe,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("quantile", [0.99, 0.995])
def test_sparse_sample_can_miss_true_tail(quantile):
    # Construct a valid finite sample entirely below the
    # analytical quantile. A percentile bootstrap cannot
    # repair this missing upper tail.
    n_paths = 256
    normals = norm.ppf((np.arange(n_paths) + 0.5) / (n_paths + 1.0))

    # Keep all observations below the chosen true quantile.
    normals = np.minimum(
        normals,
        norm.ppf(quantile) - 0.1,
    )

    times, values = fx_values(
        n_paths=n_paths,
        normal_draws=normals[:, None],
    )

    result = pfe_bootstrap_intervals(
        times,
        values,
        pfe_confidence_level=quantile,
        n_resamples=100,
        seed=459700,
    )

    target = analytical_pfe(quantile)

    assert result.pfe[-1] < target
    assert result.upper_bounds[-1] < target
