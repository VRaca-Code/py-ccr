"""Tests for independent-seed paired EPE convergence."""

import numpy as np
import pytest

from pyccr.convergence.paired_epe_grid import (
    paired_fx_epe_grid_differences,
)
from pyccr.convergence.seeded_paired_epe import (
    independent_seed_paired_fx_epe_study,
)


def parameters(**overrides):
    values = {
        "horizon": 1.0,
        "step_counts": (4, 8, 16),
        "seeds": (101, 202, 303, 404, 505),
        "n_paths": 128,
        "initial_spot": 1.1,
        "domestic_rate": 0.03,
        "foreign_rate": 0.01,
        "volatility": 0.2,
        "strike": 1.1,
    }
    values.update(overrides)
    return values


def test_shapes_and_metadata():
    result = independent_seed_paired_fx_epe_study(**parameters())
    assert result.seeds.tolist() == [101, 202, 303, 404, 505]
    assert result.coarse_step_counts.tolist() == [4, 8]
    assert result.fine_step_counts.tolist() == [8, 16]
    assert result.run_differences.shape == (5, 2)
    assert result.run_standard_errors.shape == (5, 2)
    assert result.run_lower_bounds.shape == (5, 2)
    assert result.run_upper_bounds.shape == (5, 2)
    assert result.across_run_means.shape == (2,)
    assert result.across_run_standard_errors.shape == (2,)
    assert result.n_runs == 5
    assert result.n_paths_per_run == 128
    assert result.n_independent_units_per_run == 128


def test_each_seed_matches_individual_study():
    inputs = parameters()
    result = independent_seed_paired_fx_epe_study(**inputs)
    for index, seed in enumerate(inputs["seeds"]):
        individual = paired_fx_epe_grid_differences(
            **{key: value for key, value in inputs.items() if key != "seeds"},
            seed=seed,
        )
        np.testing.assert_allclose(
            result.run_differences[index],
            individual.paired_differences,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.run_standard_errors[index],
            individual.paired_standard_errors,
            rtol=0.0,
            atol=0.0,
        )


def test_across_run_mean_and_standard_error():
    result = independent_seed_paired_fx_epe_study(**parameters())
    np.testing.assert_allclose(
        result.across_run_means,
        np.mean(result.run_differences, axis=0),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.across_run_standard_errors,
        np.std(result.run_differences, axis=0, ddof=1) / np.sqrt(result.n_runs),
        rtol=1e-13,
        atol=1e-14,
    )


def test_across_run_intervals_contain_estimates():
    result = independent_seed_paired_fx_epe_study(**parameters())
    assert np.all(result.across_run_lower_bounds <= result.across_run_means)
    assert np.all(result.across_run_means <= result.across_run_upper_bounds)


def test_error_decomposition():
    result = independent_seed_paired_fx_epe_study(**parameters())
    np.testing.assert_allclose(
        result.across_run_means,
        result.deterministic_differences + result.across_run_sampling_errors,
        atol=1e-14,
    )


def test_reproducibility():
    first = independent_seed_paired_fx_epe_study(**parameters())
    second = independent_seed_paired_fx_epe_study(**parameters())
    np.testing.assert_array_equal(
        first.run_differences,
        second.run_differences,
    )
    np.testing.assert_array_equal(
        first.across_run_standard_errors,
        second.across_run_standard_errors,
    )


def test_antithetic_independent_units():
    result = independent_seed_paired_fx_epe_study(**parameters(antithetic=True))
    assert result.antithetic
    assert result.n_independent_units_per_run == 64
    assert result.n_runs == 5
    assert np.all(result.run_standard_errors >= 0)


def test_zero_volatility():
    result = independent_seed_paired_fx_epe_study(
        **parameters(volatility=0.0, strike=1.0)
    )
    np.testing.assert_allclose(
        result.run_standard_errors,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.across_run_standard_errors,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.across_run_sampling_errors,
        0.0,
        atol=1e-12,
    )


def test_deterministic_benchmark_seed_invariance():
    first = independent_seed_paired_fx_epe_study(**parameters())
    second = independent_seed_paired_fx_epe_study(**parameters(seeds=(11, 22, 33)))
    np.testing.assert_array_equal(
        first.deterministic_differences,
        second.deterministic_differences,
    )


def test_minimum_three_seeds():
    result = independent_seed_paired_fx_epe_study(**parameters(seeds=(1, 2, 3)))
    assert result.n_runs == 3


@pytest.mark.parametrize(
    "seeds",
    [
        (),
        (1,),
        (1, 2),
        [1, 2, 3],
        (1, 1, 2),
        (1, -2, 3),
        (True, 2, 3),
        (1.0, 2, 3),
        (1, 2, 2**63),
    ],
)
def test_invalid_seeds(seeds):
    with pytest.raises(ValueError, match="seeds"):
        independent_seed_paired_fx_epe_study(**parameters(seeds=seeds))


@pytest.mark.parametrize(
    "counts",
    [
        (4,),
        (8, 4),
        (3, 8),
        (4, 4),
    ],
)
def test_invalid_grids(counts):
    with pytest.raises(ValueError):
        independent_seed_paired_fx_epe_study(**parameters(step_counts=counts))


def test_invalid_confidence_level():
    with pytest.raises(ValueError, match="confidence_level"):
        independent_seed_paired_fx_epe_study(**parameters(confidence_level=1.0))


def test_invalid_antithetic_path_count():
    with pytest.raises(ValueError, match="n_paths"):
        independent_seed_paired_fx_epe_study(**parameters(antithetic=True, n_paths=3))


def test_invalid_seed_independence_not_inferred():
    # Duplicate seeds must never be silently treated
    # as independent runs.
    with pytest.raises(ValueError, match="distinct"):
        independent_seed_paired_fx_epe_study(**parameters(seeds=(101, 202, 101)))
