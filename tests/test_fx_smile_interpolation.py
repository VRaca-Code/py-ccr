"""Regression tests for FX 25-delta smile interpolation."""

import math
from dataclasses import FrozenInstanceError

import pytest

from pyccr.market.fx_smile import fx_25delta_smile_nodes
from pyccr.market.fx_smile_interpolation import fx_25delta_smile_surface
from pyccr.market.fx_smile_quotes import FXSmileQuotes
from pyccr.market.fx_vol_surface import FXVolSurface


@pytest.fixture
def quotes():
    return FXSmileQuotes(
        atm_vol=0.12,
        risk_reversal=0.02,
        butterfly=0.005,
    )


@pytest.fixture
def surface(quotes):
    return fx_25delta_smile_surface(
        forward=1.10,
        maturity=1.0,
        quotes=quotes,
    )


def test_returns_existing_surface_type(surface):
    assert isinstance(surface, FXVolSurface)


def test_single_maturity(surface):
    assert surface.maturities == (1.0,)


def test_three_strike_nodes(surface):
    assert len(surface.strikes) == 3
    assert surface.strikes[0] < surface.strikes[1] < surface.strikes[2]


def test_atm_forward_node(surface):
    assert surface.strikes[1] == 1.10


def test_quote_volatilities_recovered(surface):
    assert surface.volatilities[0] == pytest.approx((0.115, 0.12, 0.135))


@pytest.mark.parametrize("node_index", [0, 1, 2])
def test_exact_market_node_recovery(surface, node_index):
    strike = surface.strikes[node_index]
    quoted_vol = surface.volatilities[0][node_index]

    assert surface.volatility(1.0, strike) == pytest.approx(quoted_vol, abs=1e-14)


@pytest.mark.parametrize("left_index", [0, 1])
def test_midpoint_interpolates_squared_volatility(surface, left_index):
    left_strike = surface.strikes[left_index]
    right_strike = surface.strikes[left_index + 1]

    left_vol = surface.volatilities[0][left_index]
    right_vol = surface.volatilities[0][left_index + 1]

    midpoint = (left_strike + right_strike) / 2

    expected_variance = (left_vol**2 + right_vol**2) / 2

    assert surface.volatility(1.0, midpoint) == pytest.approx(
        math.sqrt(expected_variance), abs=1e-14
    )


@pytest.mark.parametrize("weight", [0.1, 0.25, 0.75, 0.9])
def test_weighted_variance_interpolation(surface, weight):
    left_strike, right_strike = surface.strikes[:2]
    left_vol, right_vol = surface.volatilities[0][:2]

    strike = left_strike + weight * (right_strike - left_strike)

    expected_total_variance = (1 - weight) * left_vol**2 + weight * right_vol**2

    assert surface.total_variance(1.0, strike) == pytest.approx(expected_total_variance)


def test_matches_direct_smile_node_construction(quotes):
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=quotes,
    )
    surface = fx_25delta_smile_surface(
        forward=1.10,
        maturity=1.0,
        quotes=quotes,
    )

    assert surface.strikes == nodes.strikes
    assert surface.volatilities == (nodes.volatilities,)


def test_forward_scaling(quotes):
    first = fx_25delta_smile_surface(
        forward=1.0,
        maturity=1.0,
        quotes=quotes,
    )
    second = fx_25delta_smile_surface(
        forward=2.0,
        maturity=1.0,
        quotes=quotes,
    )

    assert second.strikes == pytest.approx(
        tuple(2 * strike for strike in first.strikes)
    )
    assert second.volatilities == first.volatilities


@pytest.mark.parametrize("side", ["below", "above"])
def test_strike_extrapolation_rejected(surface, side):
    if side == "below":
        strike = surface.strikes[0] * 0.99
    else:
        strike = surface.strikes[-1] * 1.01

    with pytest.raises(ValueError, match="strike outside"):
        surface.volatility(1.0, strike)


@pytest.mark.parametrize("maturity", [0.5, 1.5])
def test_maturity_extrapolation_rejected(surface, maturity):
    with pytest.raises(ValueError, match="maturity outside"):
        surface.volatility(maturity, surface.strikes[1])


@pytest.mark.parametrize(
    ("forward", "maturity"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (math.nan, 1.0),
        (math.inf, 1.0),
        (1.10, 0.0),
        (1.10, -1.0),
        (1.10, math.nan),
        (1.10, math.inf),
    ],
)
def test_invalid_inputs_rejected(quotes, forward, maturity):
    with pytest.raises(ValueError):
        fx_25delta_smile_surface(
            forward=forward,
            maturity=maturity,
            quotes=quotes,
        )


def test_invalid_quote_type_rejected():
    with pytest.raises(TypeError, match="FXSmileQuotes"):
        fx_25delta_smile_surface(
            forward=1.10,
            maturity=1.0,
            quotes=None,
        )


def test_surface_is_immutable(surface):
    with pytest.raises(FrozenInstanceError):
        surface.strikes = (1.0, 1.1, 1.2)


def test_zero_wing_volatility_rejected():
    with pytest.raises(ValueError, match="Positive wing"):
        fx_25delta_smile_surface(
            forward=1.10,
            maturity=1.0,
            quotes=FXSmileQuotes(
                atm_vol=0.0,
                risk_reversal=0.0,
                butterfly=0.0,
            ),
        )


def test_positive_total_variance_inside_domain(surface):
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        strike = surface.strikes[0] + weight * (
            surface.strikes[-1] - surface.strikes[0]
        )
        assert surface.total_variance(1.0, strike) > 0


def test_negative_risk_reversal_supported():
    surface = fx_25delta_smile_surface(
        forward=1.10,
        maturity=1.0,
        quotes=FXSmileQuotes(
            atm_vol=0.12,
            risk_reversal=-0.02,
            butterfly=0.005,
        ),
    )

    assert surface.volatilities[0] == pytest.approx((0.135, 0.12, 0.115))
