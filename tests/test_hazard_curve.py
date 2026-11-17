"""Tests for the constant-hazard counterparty credit curve."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve


def test_zero_hazard_gives_certain_survival():
    curve = ConstantHazardCurve(0.0)
    times = np.array([0.0, 0.5, 1.0, 5.0, 30.0])

    np.testing.assert_array_equal(
        curve.survival_probability(times),
        np.ones_like(times),
    )
    np.testing.assert_array_equal(
        curve.default_probability(times),
        np.zeros_like(times),
    )
    np.testing.assert_array_equal(
        curve.marginal_default_probabilities(times),
        np.zeros(times.size - 1),
    )


def test_exponential_survival_formula():
    curve = ConstantHazardCurve(0.03)
    times = np.array([0.0, 0.25, 1.0, 2.0, 5.0])

    np.testing.assert_allclose(
        curve.survival_probability(times),
        np.exp(-0.03 * times),
        rtol=1e-14,
        atol=0.0,
    )


def test_cumulative_default_probability_complements_survival():
    curve = ConstantHazardCurve(0.025)
    times = np.linspace(0.0, 10.0, 101)

    np.testing.assert_allclose(
        curve.default_probability(times) + curve.survival_probability(times),
        1.0,
        rtol=1e-14,
        atol=1e-14,
    )


def test_survival_is_monotonically_decreasing():
    curve = ConstantHazardCurve(0.04)
    times = np.linspace(0.0, 30.0, 301)

    survival = curve.survival_probability(times)

    assert np.all(np.diff(survival) <= 0.0)
    assert np.all((survival >= 0.0) & (survival <= 1.0))


def test_default_probability_is_monotonically_increasing():
    curve = ConstantHazardCurve(0.04)
    times = np.linspace(0.0, 30.0, 301)

    probabilities = curve.default_probability(times)

    assert np.all(np.diff(probabilities) >= 0.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_marginal_default_probabilities_telescope():
    curve = ConstantHazardCurve(0.035)
    times = np.array([0.0, 0.1, 0.5, 1.0, 2.5, 5.0])

    marginal = curve.marginal_default_probabilities(times)

    assert marginal.shape == (times.size - 1,)
    assert np.all(marginal >= 0.0)

    np.testing.assert_allclose(
        marginal.sum(),
        curve.default_probability(times[-1]),
        rtol=1e-13,
        atol=1e-14,
    )


def test_marginal_default_probabilities_match_survival_differences():
    curve = ConstantHazardCurve(0.02)
    times = np.array([0.0, 0.25, 0.75, 2.0, 4.0])

    survival = curve.survival_probability(times)

    np.testing.assert_allclose(
        curve.marginal_default_probabilities(times),
        survival[:-1] - survival[1:],
        rtol=1e-13,
        atol=1e-14,
    )


def test_scalar_inputs():
    curve = ConstantHazardCurve(0.02)

    np.testing.assert_allclose(
        curve.survival_probability(2.0),
        np.exp(-0.04),
    )
    np.testing.assert_allclose(
        curve.default_probability(2.0),
        1.0 - np.exp(-0.04),
    )


def test_array_inputs_preserve_shape():
    curve = ConstantHazardCurve(0.02)
    times = np.array([[0.0, 1.0], [2.0, 3.0]])

    assert curve.survival_probability(times).shape == (2, 2)
    assert curve.default_probability(times).shape == (2, 2)


@pytest.mark.parametrize(
    "hazard_rate",
    [-0.01, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_hazard_rate_rejected(hazard_rate):
    with pytest.raises(ValueError, match="hazard_rate"):
        ConstantHazardCurve(hazard_rate)


@pytest.mark.parametrize(
    "times",
    [
        [-0.1, 1.0],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [0.0, -float("inf")],
    ],
)
def test_invalid_times_rejected(times):
    curve = ConstantHazardCurve(0.02)

    with pytest.raises(ValueError, match="times"):
        curve.survival_probability(times)

    with pytest.raises(ValueError, match="times"):
        curve.default_probability(times)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [0.0],
        [0.1, 1.0],
        [0.0, 0.5, 0.5],
        [0.0, 1.0, 0.5],
        [[0.0, 1.0], [2.0, 3.0]],
    ],
)
def test_invalid_marginal_default_grid_rejected(times):
    curve = ConstantHazardCurve(0.02)

    with pytest.raises(ValueError, match="times"):
        curve.marginal_default_probabilities(times)


def test_large_hazard_remains_numerically_stable():
    curve = ConstantHazardCurve(100.0)

    np.testing.assert_allclose(
        curve.default_probability(10.0),
        1.0,
        rtol=0.0,
        atol=1e-15,
    )

    marginal = curve.marginal_default_probabilities([0.0, 1.0, 10.0])

    assert np.all(np.isfinite(marginal))
    assert np.all(marginal >= 0.0)
    np.testing.assert_allclose(marginal.sum(), 1.0)


def test_curve_is_immutable():
    curve = ConstantHazardCurve(0.02)

    with pytest.raises(AttributeError):
        curve.hazard_rate = 0.05
