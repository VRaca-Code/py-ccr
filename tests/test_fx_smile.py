"""Regression tests for FX 25-delta smile construction."""

import math
from dataclasses import FrozenInstanceError

import pytest
from scipy.stats import norm

from pyccr.market.fx_smile import FXSmileNodes, fx_25delta_smile_nodes
from pyccr.market.fx_smile_quotes import FXSmileQuotes


def make_quotes():
    return FXSmileQuotes(
        atm_vol=0.12,
        risk_reversal=0.02,
        butterfly=0.005,
    )


def forward_delta(forward, strike, maturity, volatility, option_type):
    sigma_sqrt_t = volatility * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * maturity) / sigma_sqrt_t
    return norm.cdf(d1) - (option_type == "put")


def test_smile_strikes_are_ordered():
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=make_quotes(),
    )
    assert nodes.strikes[0] < nodes.strikes[1] < nodes.strikes[2]


def test_atm_forward_strike():
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=make_quotes(),
    )
    assert nodes.strikes[1] == 1.10


def test_smile_recovers_market_volatilities():
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=make_quotes(),
    )
    assert nodes.volatilities == pytest.approx((0.115, 0.12, 0.135))


def test_smile_recovers_risk_reversal_and_butterfly():
    quotes = make_quotes()
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=quotes,
    )

    put_vol, atm_vol, call_vol = nodes.volatilities
    assert call_vol - put_vol == pytest.approx(quotes.risk_reversal)
    assert (put_vol + call_vol) / 2 - atm_vol == pytest.approx(quotes.butterfly)


@pytest.mark.parametrize(
    ("forward", "maturity", "atm", "rr", "bf"),
    [
        (1.10, 1.0, 0.12, 0.02, 0.005),
        (1.25, 0.5, 0.18, -0.03, 0.01),
        (150.0, 2.0, 0.15, 0.01, 0.005),
        (0.75, 5.0, 0.25, -0.02, 0.015),
    ],
)
def test_wing_strikes_recover_25delta(forward, maturity, atm, rr, bf):
    nodes = fx_25delta_smile_nodes(
        forward=forward,
        maturity=maturity,
        quotes=FXSmileQuotes(
            atm_vol=atm,
            risk_reversal=rr,
            butterfly=bf,
        ),
    )

    put_delta = forward_delta(
        forward,
        nodes.strikes[0],
        maturity,
        nodes.volatilities[0],
        "put",
    )
    call_delta = forward_delta(
        forward,
        nodes.strikes[2],
        maturity,
        nodes.volatilities[2],
        "call",
    )

    assert put_delta == pytest.approx(-0.25, abs=1e-12)
    assert call_delta == pytest.approx(0.25, abs=1e-12)


def test_forward_scaling():
    quotes = make_quotes()
    nodes_a = fx_25delta_smile_nodes(forward=1.0, maturity=1.0, quotes=quotes)
    nodes_b = fx_25delta_smile_nodes(forward=2.0, maturity=1.0, quotes=quotes)

    assert nodes_b.strikes == pytest.approx(
        tuple(2 * strike for strike in nodes_a.strikes)
    )
    assert nodes_b.volatilities == nodes_a.volatilities


def test_negative_risk_reversal():
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=FXSmileQuotes(
            atm_vol=0.12,
            risk_reversal=-0.02,
            butterfly=0.005,
        ),
    )
    assert nodes.volatilities == pytest.approx((0.135, 0.12, 0.115))


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
def test_invalid_forward_or_maturity(forward, maturity):
    with pytest.raises(ValueError):
        fx_25delta_smile_nodes(
            forward=forward,
            maturity=maturity,
            quotes=make_quotes(),
        )


def test_invalid_quotes_type():
    with pytest.raises(TypeError, match="FXSmileQuotes"):
        fx_25delta_smile_nodes(
            forward=1.10,
            maturity=1.0,
            quotes=None,
        )


@pytest.mark.parametrize(
    ("atm", "rr", "bf"),
    [
        (0.0, 0.0, 0.0),
        (0.10, 0.20, 0.0),
        (0.10, -0.20, 0.0),
    ],
)
def test_zero_wing_volatility_rejected(atm, rr, bf):
    with pytest.raises(ValueError, match="Positive wing"):
        fx_25delta_smile_nodes(
            forward=1.10,
            maturity=1.0,
            quotes=FXSmileQuotes(
                atm_vol=atm,
                risk_reversal=rr,
                butterfly=bf,
            ),
        )


def test_smile_nodes_are_immutable():
    nodes = fx_25delta_smile_nodes(
        forward=1.10,
        maturity=1.0,
        quotes=make_quotes(),
    )
    with pytest.raises(FrozenInstanceError):
        nodes.strikes = (1.0, 1.1, 1.2)


@pytest.mark.parametrize(
    "strikes",
    [
        (1.0, 1.0, 1.2),
        (1.2, 1.1, 1.0),
        (0.0, 1.1, 1.2),
        (math.nan, 1.1, 1.2),
    ],
)
def test_invalid_smile_node_strikes(strikes):
    with pytest.raises(ValueError):
        FXSmileNodes(
            strikes=strikes,
            volatilities=(0.11, 0.12, 0.13),
        )


@pytest.mark.parametrize(
    "volatilities",
    [
        (-0.01, 0.12, 0.13),
        (0.11, math.nan, 0.13),
        (0.11, 0.12, math.inf),
    ],
)
def test_invalid_smile_node_volatilities(volatilities):
    with pytest.raises(ValueError):
        FXSmileNodes(
            strikes=(1.0, 1.1, 1.2),
            volatilities=volatilities,
        )
