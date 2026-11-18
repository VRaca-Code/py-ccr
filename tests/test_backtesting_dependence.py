"""Tests for descriptive breach serial-dependence diagnostics."""

import numpy as np
import pytest

from pyccr.backtesting.dependence import (
    breach_dependence_diagnostics,
)


def test_transition_counts_and_breach_pairs():
    breaches = np.array(
        [False, True, True, False, False, True],
        dtype=bool,
    )
    result = breach_dependence_diagnostics(breaches, max_lag=2)

    np.testing.assert_array_equal(
        result.transition_counts,
        [[1, 2], [1, 1]],
    )
    assert result.consecutive_breach_pairs == 1
    assert result.n_breaches == 3
    assert result.breach_frequency == pytest.approx(0.5)
    assert result.transition_counts.sum() == 5


def test_alternating_sequence_has_negative_lag_one_correlation():
    breaches = np.array([False, True] * 20, dtype=bool)
    result = breach_dependence_diagnostics(breaches, max_lag=3)

    assert result.autocorrelations[0] == pytest.approx(-1.0)
    assert result.autocorrelations[1] == pytest.approx(1.0)
    assert result.autocorrelations[2] == pytest.approx(-1.0)


def test_clustered_sequence_has_positive_lag_one_correlation():
    breaches = np.array(
        [False] * 20 + [True] * 20,
        dtype=bool,
    )
    result = breach_dependence_diagnostics(breaches, max_lag=2)

    assert result.autocorrelations[0] > 0.8
    assert result.consecutive_breach_pairs == 19


@pytest.mark.parametrize("value", [False, True])
def test_constant_sequence_reports_undefined_correlation(value):
    result = breach_dependence_diagnostics(
        np.full(20, value, dtype=bool),
        max_lag=3,
    )

    assert np.all(np.isnan(result.autocorrelations))
    assert result.variance_inflation_estimate is None
    assert result.effective_sample_size_estimate is None


def test_lag_lengths_and_finite_results():
    breaches = np.array(
        [False, False, True, False, True] * 10,
        dtype=bool,
    )
    result = breach_dependence_diagnostics(breaches, max_lag=5)

    np.testing.assert_array_equal(result.lags, [1, 2, 3, 4, 5])
    assert result.autocorrelations.shape == (5,)
    assert np.all(np.isfinite(result.autocorrelations))
    assert result.transition_counts.sum() == 49


def test_effective_sample_size_matches_inflation_formula():
    breaches = np.array(
        [False] * 15 + [True] * 15,
        dtype=bool,
    )
    result = breach_dependence_diagnostics(breaches, max_lag=3)

    lags = result.lags
    expected = 1 + 2 * np.sum((1 - lags / len(breaches)) * result.autocorrelations)

    assert result.variance_inflation_estimate == pytest.approx(expected)
    assert result.effective_sample_size_estimate == pytest.approx(
        len(breaches) / expected
    )


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [True],
        [0, 1, 0],
        [[True, False]],
        [0.0, 1.0],
    ],
)
def test_invalid_breach_arrays_rejected(invalid):
    with pytest.raises(ValueError, match="boolean"):
        breach_dependence_diagnostics(invalid)


@pytest.mark.parametrize(
    "invalid",
    [0, -1, 5, 1.5, True],
)
def test_invalid_max_lag_rejected(invalid):
    breaches = np.array(
        [True, False, True, False, False],
        dtype=bool,
    )
    with pytest.raises(ValueError, match="max_lag"):
        breach_dependence_diagnostics(breaches, max_lag=invalid)


def test_single_lag_supported():
    breaches = np.array(
        [False, True, False, False],
        dtype=bool,
    )
    result = breach_dependence_diagnostics(breaches, max_lag=1)
    assert result.lags.tolist() == [1]
    assert len(result.autocorrelations) == 1
