"""Tests for Christoffersen breach-coverage diagnostics."""

import numpy as np
import pytest
from scipy.stats import chi2

from pyccr.backtesting.christoffersen import (
    christoffersen_coverage_test,
)


def test_transition_counts_and_conditional_identity():
    breaches = np.array(
        [False, True, True, False, False, True],
        dtype=bool,
    )
    result = christoffersen_coverage_test(breaches, nominal_probability=0.2)

    np.testing.assert_array_equal(
        result.transition_counts,
        [[1, 2], [1, 1]],
    )
    assert result.independence_identifiable
    assert result.lr_independence is not None
    assert result.lr_conditional == pytest.approx(
        result.lr_unconditional + result.lr_independence
    )
    assert result.p_conditional == pytest.approx(chi2.sf(result.lr_conditional, 2))


def test_unconditional_statistic_zero_at_empirical_frequency():
    breaches = np.array(
        [False, True] * 10,
        dtype=bool,
    )
    result = christoffersen_coverage_test(breaches, nominal_probability=0.5)

    assert result.lr_unconditional == pytest.approx(0.0)
    assert result.p_unconditional == pytest.approx(1.0)


def test_clustered_breaches_reject_independence_asymptotically():
    breaches = np.array(
        [False] * 100 + [True] * 100,
        dtype=bool,
    )
    result = christoffersen_coverage_test(breaches, nominal_probability=0.5)

    assert result.lr_independence > 10
    assert result.p_independence < 0.01
    assert result.probability_after_breach > (result.probability_after_no_breach)


@pytest.mark.parametrize("value", [False, True])
def test_constant_sequences_have_unidentified_independence(value):
    breaches = np.full(30, value, dtype=bool)
    result = christoffersen_coverage_test(breaches, nominal_probability=0.05)

    assert not result.independence_identifiable
    assert result.lr_independence is None
    assert result.p_independence is None
    assert result.lr_conditional is None
    assert result.p_conditional is None
    assert result.sparse_transitions
    assert np.isfinite(result.lr_unconditional)


def test_zero_transition_cells_do_not_create_nan():
    breaches = np.array(
        [False, True] * 15,
        dtype=bool,
    )
    result = christoffersen_coverage_test(breaches, nominal_probability=0.5)

    assert result.independence_identifiable
    assert np.isfinite(result.lr_independence)
    assert np.isfinite(result.lr_conditional)
    assert result.sparse_transitions


def test_sparse_transition_flag():
    breaches = np.array(
        [False] * 99 + [True],
        dtype=bool,
    )
    result = christoffersen_coverage_test(breaches, nominal_probability=0.01)

    assert result.sparse_transitions
    assert result.independence_identifiable is False


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
def test_invalid_breach_sequences(invalid):
    with pytest.raises(ValueError, match="boolean"):
        christoffersen_coverage_test(invalid, nominal_probability=0.05)


@pytest.mark.parametrize(
    "invalid",
    [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_nominal_probabilities(invalid):
    with pytest.raises(ValueError, match="nominal_probability"):
        christoffersen_coverage_test([True, False], nominal_probability=invalid)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), "abc"])
def test_invalid_nominal_probability_types(invalid):
    with pytest.raises(TypeError, match="nominal_probability"):
        christoffersen_coverage_test([True, False], nominal_probability=invalid)


def test_transition_total_is_n_minus_one():
    rng = np.random.default_rng(123)
    breaches = rng.random(200) < 0.1

    result = christoffersen_coverage_test(breaches, nominal_probability=0.1)

    assert result.transition_counts.sum() == 199
    assert result.n_observations == 200
    assert 0.0 <= result.p_unconditional <= 1.0
    if result.independence_identifiable:
        assert 0.0 <= result.p_independence <= 1.0
        assert 0.0 <= result.p_conditional <= 1.0
