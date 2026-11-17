"""Regression tests for smile-aware European FX option pricing."""

import math

import pytest

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.market.fx_smile_interpolation import fx_25delta_smile_surface
from pyccr.market.fx_smile_quotes import FXSmileQuotes
from pyccr.market.fx_vol_surface import FXVolSurface
from pyccr.pricing.fx_forward import FXForward
from pyccr.pricing.fx_option import FXOption
from pyccr.pricing.fx_smile_option import price_fx_option_with_smile


@pytest.fixture
def curves():
    return CurveSet(
        discount_curves={
            "USD": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(0.04, 0.04),
            ),
            "EUR": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(0.02, 0.02),
            ),
        },
        projection_curves={},
    )


@pytest.fixture
def surface(curves):
    maturity = 2.0
    forward = FXForward("USD", "EUR", 1.10).forward_rate(curves, maturity)

    return fx_25delta_smile_surface(
        forward=forward,
        maturity=maturity,
        quotes=FXSmileQuotes(
            atm_vol=0.12,
            risk_reversal=0.02,
            butterfly=0.005,
        ),
    )


def smile_price(curves, surface, strike, option_type="call", **overrides):
    parameters = {
        "domestic_currency": "USD",
        "foreign_currency": "EUR",
        "spot": 1.10,
        "strike": strike,
        "maturity": 2.0,
        "option_type": option_type,
        "curves": curves,
        "surface": surface,
    }
    parameters.update(overrides)
    return price_fx_option_with_smile(**parameters)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("node_index", [0, 1, 2])
def test_market_node_price_matches_direct_pricer(
    curves, surface, option_type, node_index
):
    strike = surface.strikes[node_index]
    volatility = surface.volatilities[0][node_index]

    expected = FXOption(
        domestic_currency="USD",
        foreign_currency="EUR",
        spot=1.10,
        strike=strike,
        maturity=2.0,
        volatility=volatility,
        option_type=option_type,
    ).price(curves)

    assert smile_price(curves, surface, strike, option_type) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75])
def test_interpolated_price_matches_direct_pricer(curves, surface, option_type, weight):
    left, right = surface.strikes[:2]
    strike = left + weight * (right - left)

    volatility = surface.volatility(2.0, strike)

    expected = FXOption(
        domestic_currency="USD",
        foreign_currency="EUR",
        spot=1.10,
        strike=strike,
        maturity=2.0,
        volatility=volatility,
        option_type=option_type,
    ).price(curves)

    assert smile_price(curves, surface, strike, option_type) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("node_index", [0, 1, 2])
def test_put_call_parity(curves, surface, node_index):
    strike = surface.strikes[node_index]

    call = smile_price(curves, surface, strike, "call")
    put = smile_price(curves, surface, strike, "put")

    domestic_df = curves.discount_curve("USD").discount_factor(2.0)
    foreign_df = curves.discount_curve("EUR").discount_factor(2.0)

    expected = 1.10 * foreign_df - strike * domestic_df

    assert call - put == pytest.approx(expected, abs=1e-12)


def test_strike_specific_volatility_changes_price(curves, surface):
    strike = surface.strikes[0]
    smile_volatility = surface.volatility(2.0, strike)

    smile_value = smile_price(curves, surface, strike)

    atm_value = FXOption(
        domestic_currency="USD",
        foreign_currency="EUR",
        spot=1.10,
        strike=strike,
        maturity=2.0,
        volatility=surface.volatilities[0][1],
        option_type="call",
    ).price(curves)

    assert smile_volatility != surface.volatilities[0][1]
    assert smile_value != pytest.approx(atm_value, abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_flat_smile_matches_constant_volatility(curves, option_type):
    flat_surface = FXVolSurface(
        maturities=(2.0,),
        strikes=(0.9, 1.1, 1.3),
        volatilities=((0.15, 0.15, 0.15),),
    )

    strike = 1.12

    expected = FXOption(
        domestic_currency="USD",
        foreign_currency="EUR",
        spot=1.10,
        strike=strike,
        maturity=2.0,
        volatility=0.15,
        option_type=option_type,
    ).price(curves)

    assert smile_price(curves, flat_surface, strike, option_type) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("side", ["below", "above"])
def test_strike_extrapolation_rejected(curves, surface, side):
    strike = (
        surface.strikes[0] * 0.99 if side == "below" else surface.strikes[-1] * 1.01
    )

    with pytest.raises(ValueError, match="strike outside"):
        smile_price(curves, surface, strike)


@pytest.mark.parametrize("maturity", [1.0, 3.0])
def test_maturity_extrapolation_rejected(curves, surface, maturity):
    with pytest.raises(ValueError, match="maturity outside"):
        smile_price(
            curves,
            surface,
            surface.strikes[1],
            maturity=maturity,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spot", 0.0),
        ("spot", -1.0),
        ("spot", math.nan),
        ("strike", 0.0),
        ("strike", -1.0),
        ("strike", math.inf),
        ("maturity", -1.0),
        ("maturity", math.nan),
        ("option_type", "digital"),
    ],
)
def test_invalid_option_inputs(curves, surface, field, value):
    with pytest.raises(ValueError):
        parameters = {field: value}
        strike = parameters.pop("strike", surface.strikes[1])
        smile_price(
            curves,
            surface,
            strike,
            **parameters,
        )


def test_invalid_option_type_type(curves, surface):
    with pytest.raises(TypeError, match="option_type"):
        smile_price(
            curves,
            surface,
            surface.strikes[1],
            option_type=None,
        )


def test_invalid_surface_type(curves):
    with pytest.raises(TypeError, match="FXVolSurface"):
        smile_price(curves, None, 1.10)


def test_invalid_curve_set_type(surface):
    with pytest.raises(TypeError, match="CurveSet"):
        smile_price(None, surface, surface.strikes[1])


def test_missing_discount_curve(surface):
    incomplete_curves = CurveSet(
        discount_curves={
            "USD": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(0.04, 0.04),
            )
        },
        projection_curves={},
    )

    with pytest.raises(KeyError):
        smile_price(
            incomplete_curves,
            surface,
            surface.strikes[1],
        )


def test_currency_and_option_type_normalization(curves, surface):
    strike = surface.strikes[1]

    normalized = smile_price(curves, surface, strike, option_type="call")

    unnormalized = smile_price(
        curves,
        surface,
        strike,
        option_type=" CALL ",
        domestic_currency=" usd ",
        foreign_currency=" eur ",
    )

    assert unnormalized == pytest.approx(normalized)


def test_zero_volatility_surface_uses_intrinsic_value(curves):
    zero_surface = FXVolSurface(
        maturities=(2.0,),
        strikes=(1.0, 1.1, 1.2),
        volatilities=((0.0, 0.0, 0.0),),
    )

    strike = 1.1
    forward = FXForward("USD", "EUR", 1.10).forward_rate(curves, 2.0)
    domestic_df = curves.discount_curve("USD").discount_factor(2.0)

    expected = domestic_df * max(forward - strike, 0.0)

    assert smile_price(curves, zero_surface, strike) == pytest.approx(expected)
