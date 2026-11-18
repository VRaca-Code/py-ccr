"""Tests for paired FX EPE grid-difference inference."""

import numpy as np
import pytest

from pyccr.convergence.paired_epe_grid import (
    paired_fx_epe_grid_differences,
)
from pyccr.convergence.simulated_epe_grid import (
    simulated_fx_epe_grid_convergence,
)


def parameters(**overrides):
    values = {
        "horizon": 1.0,
        "step_counts": (4, 8, 16),
        "n_paths": 256,
        "seed": 20261118,
        "initial_spot": 1.1,
        "domestic_rate": 0.03,
        "foreign_rate": 0.01,
        "volatility": 0.2,
        "strike": 1.1,
    }
    values.update(overrides)
    return values


def test_result_shapes():
    result = paired_fx_epe_grid_differences(**parameters())
    assert result.coarse_step_counts.tolist() == [4, 8]
    assert result.fine_step_counts.tolist() == [8, 16]
    assert result.paired_differences.shape == (2,)
    assert result.paired_standard_errors.shape == (2,)
    assert result.paired_lower_bounds.shape == (2,)
    assert result.paired_upper_bounds.shape == (2,)
    assert result.deterministic_differences.shape == (2,)
    assert result.n_paths == 256
    assert result.n_independent_units == 256


def test_matches_existing_simulated_grid_differences():
    inputs = parameters()
    paired = paired_fx_epe_grid_differences(**inputs)
    baseline = simulated_fx_epe_grid_convergence(**inputs)
    np.testing.assert_allclose(
        paired.paired_differences,
        np.diff(baseline.simulated_epe),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        paired.deterministic_differences,
        np.diff(baseline.deterministic_grid_epe),
    )


def test_error_decomposition():
    result = paired_fx_epe_grid_differences(**parameters())
    np.testing.assert_allclose(
        result.paired_differences,
        result.deterministic_differences + result.monte_carlo_difference_errors,
        atol=1e-14,
    )


def test_reproducibility():
    first = paired_fx_epe_grid_differences(**parameters())
    second = paired_fx_epe_grid_differences(**parameters())
    np.testing.assert_array_equal(
        first.paired_differences,
        second.paired_differences,
    )
    np.testing.assert_array_equal(
        first.paired_standard_errors,
        second.paired_standard_errors,
    )


def test_paired_se_lower_than_naive_for_reference_case():
    result = paired_fx_epe_grid_differences(
        **parameters(
            step_counts=(8, 64),
            n_paths=4096,
        )
    )
    assert np.all(
        result.paired_standard_errors < result.naive_independence_standard_errors
    )


def test_antithetic_sampling_units():
    result = paired_fx_epe_grid_differences(**parameters(antithetic=True))
    assert result.antithetic
    assert result.n_independent_units == 128
    assert np.all(result.paired_standard_errors >= 0)


def test_antithetic_matches_baseline():
    inputs = parameters(antithetic=True)
    result = paired_fx_epe_grid_differences(**inputs)
    baseline = simulated_fx_epe_grid_convergence(**inputs)
    np.testing.assert_allclose(
        result.paired_differences,
        np.diff(baseline.simulated_epe),
        rtol=1e-11,
        atol=1e-12,
    )


def test_zero_volatility_zero_sampling_error():
    result = paired_fx_epe_grid_differences(
        **parameters(
            volatility=0.0,
            strike=1.0,
        )
    )
    np.testing.assert_allclose(
        result.monte_carlo_difference_errors,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.paired_standard_errors,
        0.0,
        atol=1e-12,
    )


def test_zero_volatility_antithetic():
    result = paired_fx_epe_grid_differences(
        **parameters(
            volatility=0.0,
            strike=1.0,
            antithetic=True,
        )
    )
    np.testing.assert_allclose(
        result.paired_standard_errors,
        0.0,
        atol=1e-12,
    )


def test_confidence_intervals_contain_estimates():
    result = paired_fx_epe_grid_differences(**parameters())
    assert np.all(result.paired_lower_bounds <= result.paired_differences)
    assert np.all(result.paired_differences <= result.paired_upper_bounds)


def test_two_grids_supported():
    result = paired_fx_epe_grid_differences(**parameters(step_counts=(4, 16)))
    assert result.paired_differences.shape == (1,)


@pytest.mark.parametrize(
    "counts",
    [
        (),
        (4,),
        (8, 4),
        (3, 8),
        (4, 4),
    ],
)
def test_invalid_grids(counts):
    with pytest.raises(ValueError):
        paired_fx_epe_grid_differences(**parameters(step_counts=counts))


def test_invalid_confidence_level():
    with pytest.raises(ValueError, match="confidence_level"):
        paired_fx_epe_grid_differences(**parameters(confidence_level=1.0))


def test_invalid_antithetic_path_count():
    with pytest.raises(ValueError, match="n_paths"):
        paired_fx_epe_grid_differences(
            **parameters(
                antithetic=True,
                n_paths=3,
            )
        )


def test_invalid_seed():
    with pytest.raises(ValueError, match="seed"):
        paired_fx_epe_grid_differences(**parameters(seed=-1))


def test_invalid_strike():
    with pytest.raises(ValueError, match="strike"):
        paired_fx_epe_grid_differences(**parameters(strike=0.0))
