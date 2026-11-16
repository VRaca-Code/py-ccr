"""Regression tests for Garman-Kohlhagen FX option pricing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import norm

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.pricing.fx_option import FXOption


def make_curves(domestic_rate=0.04, foreign_rate=0.02):
    return CurveSet(
        discount_curves={
            "USD": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(domestic_rate, domestic_rate),
            ),
            "EUR": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(foreign_rate, foreign_rate),
            ),
        },
        projection_curves={},
    )


def make_option(option_type="call", **overrides):
    parameters = {
        "domestic_currency": "USD",
        "foreign_currency": "EUR",
        "spot": 1.10,
        "strike": 1.12,
        "maturity": 2.0,
        "volatility": 0.15,
        "option_type": option_type,
    }
    parameters.update(overrides)
    return FXOption(**parameters)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_garman_kohlhagen_independent_formula(option_type):
    option = make_option(option_type)
    curves = make_curves()

    spot = 1.10
    strike = 1.12
    maturity = 2.0
    volatility = 0.15
    rd = 0.04
    rf = 0.02

    forward = spot * np.exp((rd - rf) * maturity)
    total_vol = volatility * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    domestic_df = np.exp(-rd * maturity)

    if option_type == "call":
        expected = domestic_df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        expected = domestic_df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))

    assert option.price(curves) == pytest.approx(expected, rel=1e-12)


def test_fx_option_put_call_parity():
    curves = make_curves()
    call = make_option("call")
    put = make_option("put")

    domestic_df = np.exp(-0.04 * 2.0)
    foreign_df = np.exp(-0.02 * 2.0)

    expected = 1.10 * foreign_df - 1.12 * domestic_df

    assert call.price(curves) - put.price(curves) == pytest.approx(
        expected,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expected"),
    [
        ("call", 1.20, 1.10, 0.10),
        ("call", 1.00, 1.10, 0.00),
        ("put", 1.00, 1.10, 0.10),
        ("put", 1.20, 1.10, 0.00),
    ],
)
def test_fx_option_zero_maturity(option_type, spot, strike, expected):
    option = make_option(
        option_type,
        spot=spot,
        strike=strike,
        maturity=0.0,
    )

    assert option.price(make_curves()) == pytest.approx(expected)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_option_zero_volatility(option_type):
    option = make_option(option_type, volatility=0.0)
    curves = make_curves()

    forward = 1.10 * np.exp((0.04 - 0.02) * 2.0)
    discount = np.exp(-0.04 * 2.0)

    intrinsic = (
        max(forward - 1.12, 0.0) if option_type == "call" else max(1.12 - forward, 0.0)
    )

    assert option.price(curves) == pytest.approx(discount * intrinsic)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_option_value_increases_with_volatility(option_type):
    curves = make_curves()

    low = make_option(option_type, volatility=0.10)
    high = make_option(option_type, volatility=0.30)

    assert high.price(curves) > low.price(curves)


def test_fx_call_increases_with_domestic_rate():
    call = make_option("call")

    low = make_curves(domestic_rate=0.02)
    high = make_curves(domestic_rate=0.06)

    assert call.price(high) > call.price(low)


def test_fx_call_decreases_with_foreign_rate():
    call = make_option("call")

    low = make_curves(foreign_rate=0.01)
    high = make_curves(foreign_rate=0.05)

    assert call.price(high) < call.price(low)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("strike", 0.0),
        ("strike", -1.0),
        ("strike", np.nan),
        ("strike", np.inf),
        ("maturity", -1.0),
        ("maturity", np.nan),
        ("maturity", np.inf),
        ("volatility", -0.01),
        ("volatility", np.nan),
        ("volatility", np.inf),
        ("option_type", "digital"),
        ("option_type", ""),
    ],
)
def test_fx_option_invalid_inputs(field, invalid_value):
    with pytest.raises(ValueError):
        make_option(**{field: invalid_value})


def test_fx_option_invalid_option_type_type():
    with pytest.raises(TypeError, match="option_type"):
        make_option(option_type=None)


def test_fx_option_normalizes_inputs():
    option = FXOption(
        domestic_currency=" usd ",
        foreign_currency=" eur ",
        spot=1.10,
        strike=1.12,
        maturity=2.0,
        volatility=0.15,
        option_type=" CALL ",
    )

    assert option.domestic_currency == "USD"
    assert option.foreign_currency == "EUR"
    assert option.option_type == "call"


def test_fx_option_is_immutable():
    option = make_option()

    with pytest.raises(FrozenInstanceError):
        option.strike = 1.20


def test_fx_option_missing_discount_curve():
    curves = CurveSet(
        discount_curves={
            "USD": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(0.04, 0.04),
            )
        },
        projection_curves={},
    )

    with pytest.raises(KeyError):
        make_option().price(curves)
