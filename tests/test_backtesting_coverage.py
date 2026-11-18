"""Tests for historical quantile coverage and rolling breach statistics."""

import numpy as np
import pytest
from scipy.stats import binomtest

from pyccr.backtesting.coverage import (
    breach_indicators,
    exact_binomial_p_value,
    rolling_coverage,
)
from pyccr.backtesting.coverage import (
    test_coverage as evaluate_coverage,
)


def test_upper_tail_breaches():
    result = breach_indicators(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 2.0, 2.0, 4.0],
    )
    assert result.tolist() == [False, False, True, False]


def test_lower_tail_breaches():
    result = breach_indicators(
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0],
        tail="lower",
    )
    assert result.tolist() == [True, False, False]


def test_exact_upper_p_value_matches_scipy():
    observed = exact_binomial_p_value(7, 100, 0.01)
    expected = binomtest(7, 100, 0.01, alternative="greater").pvalue
    assert observed == pytest.approx(expected)


def test_exact_lower_p_value_matches_scipy():
    observed = exact_binomial_p_value(0, 100, 0.05, alternative="less")
    expected = binomtest(0, 100, 0.05, alternative="less").pvalue
    assert observed == pytest.approx(expected)


def test_exact_two_sided_p_value_matches_scipy():
    observed = exact_binomial_p_value(7, 100, 0.01, alternative="two-sided")
    expected = binomtest(7, 100, 0.01, alternative="two-sided").pvalue
    assert observed == pytest.approx(expected)


def test_coverage_statistics():
    result = evaluate_coverage(
        [0.0, 2.0, 3.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        nominal_probability=0.25,
    )
    assert result.n_observations == 4
    assert result.n_breaches == 2
    assert result.expected_breaches == pytest.approx(1.0)
    assert result.observed_frequency == pytest.approx(0.5)
    assert result.breaches.tolist() == [False, True, True, False]


def test_significance_is_distinct_from_breach_probability():
    result = evaluate_coverage(
        [2.0] * 10,
        [1.0] * 10,
        nominal_probability=0.01,
        significance_level=0.05,
    )
    assert result.nominal_probability == 0.01
    assert result.significance_level == 0.05
    assert result.reject_null


def test_rolling_counts_and_frequencies():
    result = rolling_coverage(
        np.array([True, False, True, True, False]),
        window=3,
        nominal_probability=0.1,
    )
    assert result.endpoints.tolist() == [2, 3, 4]
    assert result.counts.tolist() == [2, 2, 2]
    assert result.frequencies.tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3])
    assert result.expected_counts.tolist() == pytest.approx([0.3, 0.3, 0.3])


@pytest.mark.parametrize(
    "realized,quantiles",
    [
        ([], []),
        ([1.0, 2.0], [1.0]),
        ([1.0, float("nan")], [1.0, 1.0]),
        ([1.0], [float("inf")]),
        ([[1.0]], [[1.0]]),
    ],
)
def test_invalid_observations(realized, quantiles):
    with pytest.raises(ValueError):
        breach_indicators(realized, quantiles)


@pytest.mark.parametrize("tail", ["middle", "", None])
def test_invalid_tail(tail):
    with pytest.raises(ValueError):
        breach_indicators([1.0], [1.0], tail=tail)


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0, -0.1, float("nan"), float("inf")],
)
def test_invalid_nominal_probability(value):
    with pytest.raises(ValueError):
        exact_binomial_p_value(1, 10, value)


def test_boolean_nominal_probability_raises_type_error():
    with pytest.raises(TypeError):
        exact_binomial_p_value(1, 10, True)


@pytest.mark.parametrize(
    "count,n",
    [(-1, 10), (11, 10), (1, 0), (True, 10), (1, True)],
)
def test_invalid_counts(count, n):
    with pytest.raises(ValueError):
        exact_binomial_p_value(count, n, 0.01)


@pytest.mark.parametrize("window", [0, 6, True, 1.5])
def test_invalid_rolling_window(window):
    with pytest.raises(ValueError):
        rolling_coverage(
            np.array([True, False, True, False, True]),
            window=window,
        )


def test_rolling_requires_boolean_indicators():
    with pytest.raises(ValueError):
        rolling_coverage([0, 1, 0], window=2)


def test_rolling_single_observation():
    result = rolling_coverage(np.array([True]), window=1)
    assert result.counts.tolist() == [1]
    assert result.frequencies.tolist() == [1.0]


def test_no_breaches_is_not_upper_tail_rejection():
    result = evaluate_coverage(
        np.zeros(100),
        np.ones(100),
        nominal_probability=0.01,
        alternative="greater",
    )
    assert result.n_breaches == 0
    assert result.p_value == pytest.approx(1.0)
    assert not result.reject_null
