"""Tests for smooth OIS discount curves."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve


@pytest.fixture
def curve():
    return SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        zero_rates=(0.025, 0.027, 0.029, 0.030, 0.032, 0.034, 0.035, 0.036),
    )


def test_interpolation_reproduces_market_nodes(curve):
    np.testing.assert_allclose(
        curve.zero_rate(curve.tenors),
        curve.zero_rates,
        rtol=0.0,
        atol=1e-12,
    )


def test_discount_factors(curve):
    maturities = np.array([1.0, 5.0, 10.0])
    expected = np.exp(-maturities * curve.zero_rate(maturities))
    np.testing.assert_allclose(
        curve.discount_factor(maturities),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_instantaneous_forward_matches_numerical_derivative(curve):
    maturities = np.array([0.75, 1.5, 3.0, 7.0, 15.0, 25.0])
    h = 1e-5
    numerical = -(
        np.log(curve.discount_factor(maturities + h))
        - np.log(curve.discount_factor(maturities - h))
    ) / (2.0 * h)

    np.testing.assert_allclose(
        curve.instantaneous_forward(maturities),
        numerical,
        rtol=1e-7,
        atol=1e-9,
    )


def test_forward_derivative_matches_finite_difference(curve):
    maturities = np.array([0.75, 1.5, 3.0, 7.0, 15.0, 25.0])
    h = 1e-4
    numerical = (
        curve.instantaneous_forward(maturities + h)
        - curve.instantaneous_forward(maturities - h)
    ) / (2.0 * h)

    np.testing.assert_allclose(
        curve.forward_derivative(maturities),
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "invalid_time",
    [-1.0, 31.0, np.nan, np.inf],
)
def test_invalid_times_rejected(curve, invalid_time):
    with pytest.raises(ValueError):
        curve.discount_factor(invalid_time)


def test_initial_discount_factor_is_one(curve):
    assert float(curve.discount_factor(0.0)) == pytest.approx(1.0)


def test_discount_factors_are_positive(curve):
    maturities = np.linspace(0.0, 30.0, 301)
    assert np.all(curve.discount_factor(maturities) > 0.0)
