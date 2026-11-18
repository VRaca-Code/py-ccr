"""Tests for Plotly backtesting diagnostics."""

import numpy as np
import pytest

from app.backtesting_charts import (
    binomial_null_chart,
    bootstrap_coverage_chart,
    breach_autocorrelation_chart,
    breach_timeline_chart,
    rolling_coverage_chart,
)
from pyccr.backtesting.bootstrap import bootstrap_coverage_interval
from pyccr.backtesting.coverage import test_coverage as evaluate_coverage
from pyccr.backtesting.dependence import breach_dependence_diagnostics


@pytest.fixture
def sample():
    realized = np.zeros(100)
    realized[[5, 18, 47, 79]] = 2.0
    quantiles = np.ones(100)
    result = evaluate_coverage(realized, quantiles, nominal_probability=0.01)
    return realized, quantiles, result


def test_timeline_has_three_traces(sample):
    realized, quantiles, result = sample
    fig = breach_timeline_chart(realized, quantiles, result)
    assert len(fig.data) == 3
    assert len(fig.data[2].x) == 4
    assert list(fig.data[2].x) == [5, 18, 47, 79]


def test_timeline_rejects_misaligned_values(sample):
    realized, quantiles, result = sample
    with pytest.raises(ValueError):
        breach_timeline_chart(realized + 10.0, quantiles, result)


def test_timeline_supports_lower_tail():
    realized = np.array([2.0, 0.0, 2.0, -1.0])
    quantiles = np.ones(4)
    result = evaluate_coverage(
        realized,
        quantiles,
        tail="lower",
        nominal_probability=0.1,
    )
    fig = breach_timeline_chart(realized, quantiles, result)
    assert list(fig.data[2].x) == [1, 3]


def test_rolling_chart_has_requested_windows(sample):
    _, _, result = sample
    fig = rolling_coverage_chart(result, windows=(13, 26, 52))
    assert len(fig.data) == 3
    assert [len(trace.x) for trace in fig.data] == [88, 75, 49]


def test_rolling_chart_rejects_invalid_window(sample):
    _, _, result = sample
    with pytest.raises(ValueError):
        rolling_coverage_chart(result, windows=(101,))


def test_rolling_chart_rejects_empty_windows(sample):
    _, _, result = sample
    with pytest.raises(ValueError):
        rolling_coverage_chart(result, windows=())


def test_binomial_chart_marks_observed_count(sample):
    _, _, result = sample
    fig = binomial_null_chart(result)
    assert len(fig.data) == 1
    assert result.n_breaches in list(fig.data[0].x)
    assert np.sum(fig.data[0].y) <= 1.000001


def test_binomial_chart_rejects_small_limit(sample):
    _, _, result = sample
    with pytest.raises(ValueError):
        binomial_null_chart(result, max_points=2)


def test_binomial_chart_rejects_truncated_support(sample):
    _, _, result = sample
    with pytest.raises(ValueError):
        binomial_null_chart(result, max_points=3)


def test_chart_serialization(sample):
    realized, quantiles, result = sample
    figures = (
        breach_timeline_chart(realized, quantiles, result),
        rolling_coverage_chart(result),
        binomial_null_chart(result),
    )
    for figure in figures:
        assert figure.to_plotly_json()["data"]


def test_bootstrap_chart_contains_two_distributions(sample):
    _, _, coverage = sample

    iid = bootstrap_coverage_interval(
        coverage.breaches,
        n_resamples=200,
        seed=11,
    )
    block = bootstrap_coverage_interval(
        coverage.breaches,
        method="circular_block",
        block_length=5,
        n_resamples=200,
        seed=11,
    )

    fig = bootstrap_coverage_chart(
        iid,
        block,
        nominal_probability=coverage.nominal_probability,
    )

    assert len(fig.data) == 2
    assert fig.data[0].name == "IID bootstrap"
    assert fig.data[1].name == "Circular block bootstrap"
    assert len(fig.layout.shapes) == 6
    assert fig.to_plotly_json()["data"]


def test_bootstrap_chart_rejects_mismatched_samples(sample):
    _, _, coverage = sample

    iid = bootstrap_coverage_interval(
        coverage.breaches,
        n_resamples=100,
    )
    different = np.zeros(100, dtype=bool)
    block = bootstrap_coverage_interval(
        different,
        method="circular_block",
        block_length=5,
        n_resamples=100,
    )

    with pytest.raises(ValueError, match="same sample"):
        bootstrap_coverage_chart(
            iid,
            block,
            nominal_probability=0.01,
        )


def test_bootstrap_chart_rejects_invalid_probability(sample):
    _, _, coverage = sample

    iid = bootstrap_coverage_interval(
        coverage.breaches,
        n_resamples=100,
    )
    block = bootstrap_coverage_interval(
        coverage.breaches,
        method="circular_block",
        block_length=5,
        n_resamples=100,
    )

    with pytest.raises(ValueError, match="nominal_probability"):
        bootstrap_coverage_chart(
            iid,
            block,
            nominal_probability=0.0,
        )


def test_autocorrelation_chart_has_expected_lags(sample):
    _, _, coverage = sample

    result = breach_dependence_diagnostics(
        coverage.breaches,
        max_lag=5,
    )

    fig = breach_autocorrelation_chart(result)

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [1, 2, 3, 4, 5]
    assert len(fig.data[0].y) == 5
    assert fig.to_plotly_json()["data"]


@pytest.mark.parametrize("value", [False, True])
def test_autocorrelation_chart_handles_constant_series(value):
    result = breach_dependence_diagnostics(
        np.full(20, value, dtype=bool),
        max_lag=3,
    )

    fig = breach_autocorrelation_chart(result)

    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 0
    assert len(fig.data[0].y) == 0
    assert fig.to_plotly_json()["data"]


def test_autocorrelation_chart_has_zero_reference(sample):
    _, _, coverage = sample
    result = breach_dependence_diagnostics(
        coverage.breaches,
        max_lag=3,
    )

    fig = breach_autocorrelation_chart(result)

    assert len(fig.layout.shapes) == 1
