"""Tests for discount-curve construction and interpolation."""

import numpy as np
import pytest

from pyccr.market.discount_curve import DiscountCurve


@pytest.fixture
def curve() -> DiscountCurve:
    return DiscountCurve(
        tenors=(1.0, 2.0, 5.0, 10.0),
        zero_rates=(0.025, 0.028, 0.032, 0.035),
    )


def test_zero_rates_at_market_tenors(curve: DiscountCurve) -> None:
    np.testing.assert_allclose(
        curve.zero_rate([1.0, 2.0, 5.0, 10.0]),
        [0.025, 0.028, 0.032, 0.035],
    )


def test_linear_interpolation(curve: DiscountCurve) -> None:
    expected = 0.028 + (0.032 - 0.028) / 3.0
    np.testing.assert_allclose(curve.zero_rate(3.0), expected)


def test_flat_extrapolation(curve: DiscountCurve) -> None:
    np.testing.assert_allclose(
        curve.zero_rate([0.0, 0.5, 15.0]),
        [0.025, 0.025, 0.035],
    )


def test_discount_factor_identity(curve: DiscountCurve) -> None:
    maturities = np.array([0.0, 1.0, 3.0, 5.0, 10.0])

    expected = np.exp(-curve.zero_rate(maturities) * maturities)

    np.testing.assert_allclose(
        curve.discount_factor(maturities),
        expected,
    )


def test_discount_factor_at_zero(curve: DiscountCurve) -> None:
    np.testing.assert_allclose(curve.discount_factor(0.0), 1.0)


def test_forward_rate_between_knots(curve: DiscountCurve) -> None:
    t = 3.0
    slope = (0.032 - 0.028) / (5.0 - 2.0)

    expected = curve.zero_rate(t) + t * slope

    np.testing.assert_allclose(
        curve.instantaneous_forward(t),
        expected,
    )


def test_forward_rate_outside_knots(curve: DiscountCurve) -> None:
    np.testing.assert_allclose(
        curve.instantaneous_forward([0.5, 12.0]),
        [0.025, 0.035],
    )


def test_vectorized_outputs(curve: DiscountCurve) -> None:
    maturities = np.array([1.0, 3.0, 5.0])

    assert curve.zero_rate(maturities).shape == (3,)
    assert curve.discount_factor(maturities).shape == (3,)
    assert curve.instantaneous_forward(maturities).shape == (3,)


@pytest.mark.parametrize(
    "tenors, rates",
    [
        ((), ()),
        ((1.0, 2.0), (0.02,)),
        ((0.0, 1.0), (0.02, 0.03)),
        ((2.0, 1.0), (0.02, 0.03)),
        ((1.0, 1.0), (0.02, 0.03)),
        ((float("nan"), 2.0), (0.02, 0.03)),
        ((1.0, 2.0), (0.02, float("inf"))),
    ],
)
def test_invalid_curve_inputs(tenors, rates) -> None:
    with pytest.raises(ValueError):
        DiscountCurve(tenors=tenors, zero_rates=rates)


@pytest.mark.parametrize(
    "maturity",
    [-1.0, float("nan"), float("inf")],
)
def test_invalid_maturity(curve: DiscountCurve, maturity: float) -> None:
    with pytest.raises(ValueError):
        curve.zero_rate(maturity)

    with pytest.raises(ValueError):
        curve.discount_factor(maturity)

    with pytest.raises(ValueError):
        curve.instantaneous_forward(maturity)
