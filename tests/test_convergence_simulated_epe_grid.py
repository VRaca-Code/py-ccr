"""Tests for coupled-grid FX simulated EPE convergence."""

import numpy as np
import pytest

from pyccr.convergence.epe_grid import (
    fx_epe_grid_convergence,
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


def test_result_shapes_and_metadata():
    result = simulated_fx_epe_grid_convergence(**parameters())
    assert result.step_counts.tolist() == [4, 8, 16]
    assert result.simulated_epe.shape == (3,)
    assert result.monte_carlo_standard_errors.shape == (3,)
    assert result.monte_carlo_lower_bounds.shape == (3,)
    assert result.monte_carlo_upper_bounds.shape == (3,)
    assert result.deterministic_grid_epe.shape == (3,)
    assert result.n_paths == 256
    assert result.n_independent_units == 256
    assert not result.antithetic
    assert result.seed == 20261118


def test_deterministic_benchmark_matches_existing_module():
    result = simulated_fx_epe_grid_convergence(**parameters())
    inputs = parameters()
    benchmark = fx_epe_grid_convergence(
        **{
            key: value
            for key, value in inputs.items()
            if key not in {"n_paths", "seed"}
        }
    )
    np.testing.assert_array_equal(
        result.deterministic_grid_epe,
        benchmark.epe_estimates,
    )
    assert result.continuous_reference_epe == (benchmark.reference_epe)
    np.testing.assert_array_equal(
        result.deterministic_grid_errors,
        benchmark.signed_errors,
    )


def test_error_decomposition():
    result = simulated_fx_epe_grid_convergence(**parameters())
    np.testing.assert_allclose(
        result.monte_carlo_deviations + result.deterministic_grid_errors,
        result.total_reference_deviations,
        atol=1e-14,
    )


def test_reproducibility():
    first = simulated_fx_epe_grid_convergence(**parameters())
    second = simulated_fx_epe_grid_convergence(**parameters())
    np.testing.assert_array_equal(first.simulated_epe, second.simulated_epe)
    np.testing.assert_array_equal(
        first.monte_carlo_standard_errors,
        second.monte_carlo_standard_errors,
    )


def test_different_seeds_change_simulated_estimates():
    first = simulated_fx_epe_grid_convergence(**parameters(seed=11))
    second = simulated_fx_epe_grid_convergence(**parameters(seed=22))
    assert not np.array_equal(first.simulated_epe, second.simulated_epe)
    np.testing.assert_array_equal(
        first.deterministic_grid_epe,
        second.deterministic_grid_epe,
    )


def test_antithetic_independent_unit_count():
    result = simulated_fx_epe_grid_convergence(**parameters(antithetic=True))
    assert result.antithetic
    assert result.n_paths == 256
    assert result.n_independent_units == 128
    assert np.all(result.monte_carlo_standard_errors >= 0)


def test_zero_volatility_deterministic_study():
    result = simulated_fx_epe_grid_convergence(
        **parameters(
            volatility=0.0,
            strike=1.0,
        )
    )
    np.testing.assert_allclose(
        result.monte_carlo_deviations,
        0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.monte_carlo_standard_errors,
        0.0,
        atol=1e-13,
    )


def test_zero_volatility_antithetic_study():
    result = simulated_fx_epe_grid_convergence(
        **parameters(
            volatility=0.0,
            strike=1.0,
            antithetic=True,
        )
    )
    np.testing.assert_allclose(
        result.monte_carlo_deviations,
        0.0,
        atol=1e-13,
    )


def test_single_grid_is_supported():
    result = simulated_fx_epe_grid_convergence(**parameters(step_counts=(8,)))
    assert result.simulated_epe.shape == (1,)


def test_constant_positive_exposure():
    result = simulated_fx_epe_grid_convergence(
        **parameters(
            domestic_rate=0.0,
            foreign_rate=0.0,
            volatility=0.0,
            strike=1.0,
        )
    )
    np.testing.assert_allclose(
        result.simulated_epe,
        0.1,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.deterministic_grid_epe,
        0.1,
        atol=1e-13,
    )


def test_confidence_interval_contains_estimate():
    result = simulated_fx_epe_grid_convergence(**parameters())
    assert np.all(result.monte_carlo_lower_bounds <= result.simulated_epe)
    assert np.all(result.simulated_epe <= result.monte_carlo_upper_bounds)


@pytest.mark.parametrize(
    "counts",
    [
        (3, 8),
        (5, 8, 16),
        (7, 16),
    ],
)
def test_non_nested_grid_rejected(counts):
    with pytest.raises(ValueError, match="divide"):
        simulated_fx_epe_grid_convergence(**parameters(step_counts=counts))


@pytest.mark.parametrize(
    "count",
    [0, 1, True, 1.5],
)
def test_invalid_path_counts(count):
    with pytest.raises(ValueError, match="n_paths"):
        simulated_fx_epe_grid_convergence(**parameters(n_paths=count))


@pytest.mark.parametrize(
    "count",
    [2, 3, 5],
)
def test_invalid_antithetic_path_counts(count):
    with pytest.raises(ValueError, match="n_paths"):
        simulated_fx_epe_grid_convergence(
            **parameters(
                n_paths=count,
                antithetic=True,
            )
        )


@pytest.mark.parametrize(
    "seed",
    [-1, True, 1.5, None],
)
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        simulated_fx_epe_grid_convergence(**parameters(seed=seed))


def test_nonboolean_antithetic_rejected():
    with pytest.raises(TypeError, match="antithetic"):
        simulated_fx_epe_grid_convergence(**parameters(antithetic="yes"))


def test_invalid_confidence_level_propagates():
    with pytest.raises(ValueError, match="confidence_level"):
        simulated_fx_epe_grid_convergence(**parameters(confidence_level=1.0))


def test_invalid_strike_propagates():
    with pytest.raises(ValueError, match="strike"):
        simulated_fx_epe_grid_convergence(**parameters(strike=0.0))


def test_invalid_step_counts_propagate():
    with pytest.raises(ValueError, match="step_counts"):
        simulated_fx_epe_grid_convergence(**parameters(step_counts=(8, 4)))
