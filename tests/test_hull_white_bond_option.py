"""Tests for Hull-White zero-coupon bond options."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_bond_option import (
    bond_option_price,
    bond_option_volatility,
)


@pytest.fixture
def model():
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        zero_rates=(0.025, 0.027, 0.029, 0.030, 0.032, 0.034, 0.035, 0.036),
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=0.01,
    )


def test_bond_option_volatility_positive(model):
    assert bond_option_volatility(model, 2.0, 5.0) > 0.0


def test_volatility_zero_at_zero_expiry(model):
    assert bond_option_volatility(model, 0.0, 5.0) == pytest.approx(0.0)


def test_call_price_nonnegative(model):
    assert bond_option_price(model, 2.0, 5.0, 0.9, "call") >= 0.0


def test_put_price_nonnegative(model):
    assert bond_option_price(model, 2.0, 5.0, 0.9, "put") >= 0.0


def test_put_call_parity(model):
    expiry = 2.0
    maturity = 5.0
    strike = 0.9

    call = bond_option_price(model, expiry, maturity, strike, "call")
    put = bond_option_price(model, expiry, maturity, strike, "put")

    expected = model.curve.discount_factor(
        maturity
    ) - strike * model.curve.discount_factor(expiry)

    assert call - put == pytest.approx(float(expected), abs=1e-12)


def test_call_decreases_with_strike(model):
    low = bond_option_price(model, 2.0, 5.0, 0.85, "call")
    high = bond_option_price(model, 2.0, 5.0, 0.95, "call")
    assert low >= high


def test_put_increases_with_strike(model):
    low = bond_option_price(model, 2.0, 5.0, 0.85, "put")
    high = bond_option_price(model, 2.0, 5.0, 0.95, "put")
    assert high >= low


def test_zero_volatility_limit(model):
    deterministic = HullWhiteCurveModel(
        curve=model.curve,
        mean_reversion=model.mean_reversion,
        volatility=0.0,
    )

    expiry = 2.0
    maturity = 5.0
    strike = 0.9

    forward = float(
        model.curve.discount_factor(maturity) / model.curve.discount_factor(expiry)
    )
    discount = float(model.curve.discount_factor(expiry))

    expected = discount * max(forward - strike, 0.0)

    assert bond_option_price(
        deterministic, expiry, maturity, strike, "call"
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expiry", "maturity"),
    [
        (-1.0, 5.0),
        (5.0, 2.0),
        (2.0, 2.0),
        (2.0, 35.0),
    ],
)
def test_invalid_expiry_or_maturity(model, expiry, maturity):
    with pytest.raises(ValueError):
        bond_option_volatility(model, expiry, maturity)


@pytest.mark.parametrize("strike", [0.0, -0.1, np.nan])
def test_invalid_strike(model, strike):
    with pytest.raises(ValueError):
        bond_option_price(model, 2.0, 5.0, strike)


def test_invalid_option_type(model):
    with pytest.raises(ValueError):
        bond_option_price(model, 2.0, 5.0, 0.9, "invalid")
