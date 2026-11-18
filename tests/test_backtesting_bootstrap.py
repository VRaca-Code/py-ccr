"""Tests for IID and circular-block breach-frequency bootstrap."""

import numpy as np
import pytest

from pyccr.backtesting.bootstrap import bootstrap_coverage_interval


@pytest.fixture
def breaches():
    return np.array([False] * 90 + [True] * 10, dtype=bool)


@pytest.mark.parametrize("method", ["iid", "circular_block"])
def test_result_shape_and_bounds(breaches, method):
    options = {"block_length": 5} if method == "circular_block" else {}
    result = bootstrap_coverage_interval(
        breaches,
        method=method,
        n_resamples=300,
        seed=17,
        **options,
    )

    assert result.n_observations == 100
    assert result.n_breaches == 10
    assert result.observed_frequency == pytest.approx(0.1)
    assert result.resampled_frequencies.shape == (300,)
    assert 0 <= result.lower_bound <= result.upper_bound <= 1
    assert np.all(
        (result.resampled_frequencies >= 0) & (result.resampled_frequencies <= 1)
    )


@pytest.mark.parametrize("method", ["iid", "circular_block"])
def test_reproducible_seed(breaches, method):
    options = {"block_length": 7} if method == "circular_block" else {}
    first = bootstrap_coverage_interval(breaches, method=method, seed=11, **options)
    second = bootstrap_coverage_interval(breaches, method=method, seed=11, **options)
    np.testing.assert_array_equal(
        first.resampled_frequencies,
        second.resampled_frequencies,
    )


def test_all_zero_breaches_produce_degenerate_interval():
    result = bootstrap_coverage_interval(
        np.zeros(50, dtype=bool),
        n_resamples=100,
    )
    assert result.lower_bound == 0
    assert result.upper_bound == 0


def test_all_one_breaches_produce_degenerate_interval():
    result = bootstrap_coverage_interval(
        np.ones(50, dtype=bool),
        method="circular_block",
        block_length=5,
        n_resamples=100,
    )
    assert result.lower_bound == 1
    assert result.upper_bound == 1


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [0, 1, 0],
        [[True, False]],
        [0.0, 1.0],
    ],
)
def test_reject_nonboolean_or_invalid_shape(invalid):
    with pytest.raises(ValueError, match="boolean"):
        bootstrap_coverage_interval(invalid)


@pytest.mark.parametrize(
    "value",
    [0, -1, 1.0, True, 101],
)
def test_invalid_block_length(breaches, value):
    with pytest.raises(ValueError, match="block_length"):
        bootstrap_coverage_interval(
            breaches,
            method="circular_block",
            block_length=value,
            n_resamples=100,
        )


def test_missing_block_length_rejected(breaches):
    with pytest.raises(ValueError, match="block_length"):
        bootstrap_coverage_interval(
            breaches,
            method="circular_block",
        )


def test_block_length_rejected_for_iid(breaches):
    with pytest.raises(ValueError, match="block_length"):
        bootstrap_coverage_interval(breaches, block_length=2)


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, float("nan")])
def test_invalid_confidence_level(breaches, value):
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_coverage_interval(breaches, confidence_level=value)


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_boolean_confidence_level_rejected(breaches, value):
    with pytest.raises(TypeError, match="confidence_level"):
        bootstrap_coverage_interval(breaches, confidence_level=value)


@pytest.mark.parametrize("value", [0, 99, 1.5, True])
def test_invalid_resamples(breaches, value):
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_coverage_interval(breaches, n_resamples=value)


@pytest.mark.parametrize("value", [-1, 1.5, True, 2**32])
def test_invalid_seed(breaches, value):
    with pytest.raises(ValueError, match="seed"):
        bootstrap_coverage_interval(breaches, seed=value)


def test_invalid_method(breaches):
    with pytest.raises(ValueError, match="method"):
        bootstrap_coverage_interval(breaches, method="unsupported")


def test_block_length_one_matches_iid_with_same_seed(breaches):
    iid = bootstrap_coverage_interval(breaches, n_resamples=200, seed=13)
    block = bootstrap_coverage_interval(
        breaches,
        method="circular_block",
        block_length=1,
        n_resamples=200,
        seed=13,
    )
    np.testing.assert_array_equal(
        iid.resampled_frequencies,
        block.resampled_frequencies,
    )


def test_block_bootstrap_preserves_sample_size(breaches):
    result = bootstrap_coverage_interval(
        breaches,
        method="circular_block",
        block_length=33,
        n_resamples=200,
    )
    assert result.n_observations == len(breaches)
    assert result.block_length == 33
