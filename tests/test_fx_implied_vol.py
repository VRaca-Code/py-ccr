"""Regression tests for Garman-Kohlhagen FX implied volatility."""

from dataclasses import replace

import numpy as np
import pytest

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.pricing.fx_implied_vol import implied_volatility
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


def make_option(**overrides):
    parameters = {
        "domestic_currency": "USD",
        "foreign_currency": "EUR",
        "spot": 1.10,
        "strike": 1.12,
        "maturity": 2.0,
        "volatility": 0.15,
        "option_type": "call",
    }
    parameters.update(overrides)
    return FXOption(**parameters)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [0.90, 1.10, 1.30])
@pytest.mark.parametrize("maturity", [0.5, 2.0, 5.0])
@pytest.mark.parametrize("volatility", [0.10, 0.20, 0.40])
def test_implied_volatility_recovers_input(option_type, strike, maturity, volatility):
    curves = make_curves()
    option = make_option(
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
    )

    market_price = option.price(curves)
    recovered = implied_volatility(option, curves, market_price)

    assert recovered == pytest.approx(volatility, abs=1e-9)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_zero_volatility_price_returns_zero(option_type):
    curves = make_curves()
    option = make_option(option_type=option_type, volatility=0.0)

    recovered = implied_volatility(option, curves, option.price(curves))

    assert recovered == 0.0


def test_put_call_implied_volatility_consistency():
    curves = make_curves()
    call = make_option(option_type="call", volatility=0.23)
    put = replace(call, option_type="put")

    call_vol = implied_volatility(call, curves, call.price(curves))
    put_vol = implied_volatility(put, curves, put.price(curves))

    assert call_vol == pytest.approx(0.23, abs=1e-10)
    assert put_vol == pytest.approx(call_vol, abs=1e-10)


@pytest.mark.parametrize("market_price", [-0.01, np.nan, np.inf])
def test_invalid_market_price_rejected(market_price):
    with pytest.raises(ValueError, match="market_price"):
        implied_volatility(make_option(), make_curves(), market_price)


@pytest.mark.parametrize("max_volatility", [0.0, -1.0, np.nan, np.inf])
def test_invalid_max_volatility_rejected(max_volatility):
    with pytest.raises(ValueError, match="max_volatility"):
        implied_volatility(
            make_option(),
            make_curves(),
            0.10,
            max_volatility=max_volatility,
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan, np.inf])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        implied_volatility(
            make_option(),
            make_curves(),
            0.10,
            tolerance=tolerance,
        )


def test_zero_maturity_rejected():
    option = make_option(maturity=0.0)

    with pytest.raises(ValueError, match="zero maturity"):
        implied_volatility(option, make_curves(), option.price(make_curves()))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_below_no_arbitrage_bound_rejected(option_type):
    curves = make_curves()

    # Choose an in-the-money option with a strictly positive lower bound.
    strike = 0.90 if option_type == "call" else 1.30
    option = make_option(option_type=option_type, strike=strike)

    domestic_df = np.exp(-0.04 * option.maturity)
    foreign_df = np.exp(-0.02 * option.maturity)

    discounted_spot = option.spot * foreign_df
    discounted_strike = option.strike * domestic_df

    if option_type == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
    else:
        lower = max(discounted_strike - discounted_spot, 0.0)

    with pytest.raises(ValueError, match="lower bound"):
        implied_volatility(option, curves, lower - 0.01)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_above_no_arbitrage_bound_rejected(option_type):
    curves = make_curves()
    option = make_option(option_type=option_type)

    domestic_df = np.exp(-0.04 * option.maturity)
    foreign_df = np.exp(-0.02 * option.maturity)

    upper = (
        option.spot * foreign_df
        if option_type == "call"
        else option.strike * domestic_df
    )

    with pytest.raises(ValueError, match="upper bound"):
        implied_volatility(option, curves, upper + 0.01)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_exact_upper_bound_has_no_finite_volatility(option_type):
    curves = make_curves()
    option = make_option(option_type=option_type)

    domestic_df = np.exp(-0.04 * option.maturity)
    foreign_df = np.exp(-0.02 * option.maturity)

    upper = (
        option.spot * foreign_df
        if option_type == "call"
        else option.strike * domestic_df
    )

    with pytest.raises(ValueError, match="no finite implied volatility"):
        implied_volatility(option, curves, upper)


def test_maximum_volatility_bracket_rejected():
    curves = make_curves()
    option = make_option(volatility=0.80)
    market_price = option.price(curves)

    with pytest.raises(ValueError, match="above max_volatility"):
        implied_volatility(
            option,
            curves,
            market_price,
            max_volatility=0.20,
        )


def test_custom_maximum_volatility_recovers_input():
    curves = make_curves()
    option = make_option(volatility=0.80)
    market_price = option.price(curves)

    recovered = implied_volatility(
        option,
        curves,
        market_price,
        max_volatility=1.0,
    )

    assert recovered == pytest.approx(0.80, abs=1e-10)


def test_solver_does_not_modify_option():
    curves = make_curves()
    option = make_option(volatility=0.25)
    original = option

    recovered = implied_volatility(option, curves, option.price(curves))

    assert recovered == pytest.approx(0.25, abs=1e-10)
    assert option == original
