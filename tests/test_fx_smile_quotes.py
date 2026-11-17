"""Regression tests for FX ATM/RR/BF smile quote conventions."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyccr.market.fx_smile_quotes import FXSmileQuotes


def test_atm_rr_bf_to_wing_volatilities():
    quotes = FXSmileQuotes(
        atm_vol=0.12,
        risk_reversal=0.02,
        butterfly=0.005,
    )

    put_vol, call_vol = quotes.wing_volatilities()

    assert put_vol == pytest.approx(0.115)
    assert call_vol == pytest.approx(0.135)


def test_negative_risk_reversal():
    quotes = FXSmileQuotes(
        atm_vol=0.12,
        risk_reversal=-0.02,
        butterfly=0.005,
    )

    put_vol, call_vol = quotes.wing_volatilities()

    assert put_vol == pytest.approx(0.135)
    assert call_vol == pytest.approx(0.115)


def test_zero_risk_reversal():
    quotes = FXSmileQuotes(
        atm_vol=0.15,
        risk_reversal=0.0,
        butterfly=0.01,
    )

    assert quotes.wing_volatilities() == pytest.approx((0.16, 0.16))


def test_zero_butterfly():
    quotes = FXSmileQuotes(
        atm_vol=0.15,
        risk_reversal=0.02,
        butterfly=0.0,
    )

    assert quotes.wing_volatilities() == pytest.approx((0.14, 0.16))


def test_from_wing_volatilities():
    quotes = FXSmileQuotes.from_wing_volatilities(
        atm_vol=0.12,
        put_vol=0.115,
        call_vol=0.135,
    )

    assert quotes.atm_vol == pytest.approx(0.12)
    assert quotes.risk_reversal == pytest.approx(0.02)
    assert quotes.butterfly == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("atm_vol", "put_vol", "call_vol"),
    [
        (0.10, 0.09, 0.11),
        (0.20, 0.22, 0.18),
        (0.15, 0.15, 0.15),
        (0.0, 0.0, 0.0),
        (0.30, 0.10, 0.50),
    ],
)
def test_wing_quote_round_trip(atm_vol, put_vol, call_vol):
    quotes = FXSmileQuotes.from_wing_volatilities(
        atm_vol=atm_vol,
        put_vol=put_vol,
        call_vol=call_vol,
    )

    recovered_put, recovered_call = quotes.wing_volatilities()

    assert recovered_put == pytest.approx(put_vol)
    assert recovered_call == pytest.approx(call_vol)
    assert quotes.atm_vol == pytest.approx(atm_vol)


@pytest.mark.parametrize(
    ("atm_vol", "risk_reversal", "butterfly"),
    [
        (0.12, 0.02, 0.005),
        (0.12, -0.02, 0.005),
        (0.15, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.20, 0.10, -0.02),
    ],
)
def test_market_quote_round_trip(
    atm_vol,
    risk_reversal,
    butterfly,
):
    original = FXSmileQuotes(
        atm_vol=atm_vol,
        risk_reversal=risk_reversal,
        butterfly=butterfly,
    )

    put_vol, call_vol = original.wing_volatilities()

    recovered = FXSmileQuotes.from_wing_volatilities(
        atm_vol=atm_vol,
        put_vol=put_vol,
        call_vol=call_vol,
    )

    assert recovered.atm_vol == pytest.approx(atm_vol)
    assert recovered.risk_reversal == pytest.approx(risk_reversal)
    assert recovered.butterfly == pytest.approx(butterfly)


def test_negative_atm_rejected():
    with pytest.raises(ValueError, match="ATM"):
        FXSmileQuotes(
            atm_vol=-0.01,
            risk_reversal=0.0,
            butterfly=0.0,
        )


@pytest.mark.parametrize(
    ("atm_vol", "risk_reversal", "butterfly"),
    [
        (np.nan, 0.0, 0.0),
        (np.inf, 0.0, 0.0),
        (0.12, np.nan, 0.0),
        (0.12, -np.inf, 0.0),
        (0.12, 0.0, np.inf),
        (0.12, 0.0, np.nan),
    ],
)
def test_nonfinite_market_quotes_rejected(
    atm_vol,
    risk_reversal,
    butterfly,
):
    with pytest.raises(ValueError, match="finite"):
        FXSmileQuotes(
            atm_vol=atm_vol,
            risk_reversal=risk_reversal,
            butterfly=butterfly,
        )


@pytest.mark.parametrize(
    ("atm_vol", "risk_reversal", "butterfly"),
    [
        (0.10, 0.30, 0.0),
        (0.10, -0.30, 0.0),
        (0.10, 0.0, -0.20),
    ],
)
def test_negative_wing_volatility_rejected(
    atm_vol,
    risk_reversal,
    butterfly,
):
    with pytest.raises(ValueError, match="Wing"):
        FXSmileQuotes(
            atm_vol=atm_vol,
            risk_reversal=risk_reversal,
            butterfly=butterfly,
        )


@pytest.mark.parametrize(
    ("atm_vol", "put_vol", "call_vol"),
    [
        (0.10, -0.01, 0.12),
        (0.10, 0.12, -0.01),
        (-0.01, 0.10, 0.12),
    ],
)
def test_negative_input_volatility_rejected(
    atm_vol,
    put_vol,
    call_vol,
):
    with pytest.raises(ValueError, match="non-negative"):
        FXSmileQuotes.from_wing_volatilities(
            atm_vol=atm_vol,
            put_vol=put_vol,
            call_vol=call_vol,
        )


@pytest.mark.parametrize(
    ("atm_vol", "put_vol", "call_vol"),
    [
        (np.nan, 0.10, 0.12),
        (0.10, np.inf, 0.12),
        (0.10, 0.12, -np.inf),
    ],
)
def test_nonfinite_wing_inputs_rejected(
    atm_vol,
    put_vol,
    call_vol,
):
    with pytest.raises(ValueError, match="finite"):
        FXSmileQuotes.from_wing_volatilities(
            atm_vol=atm_vol,
            put_vol=put_vol,
            call_vol=call_vol,
        )


def test_quotes_are_immutable():
    quotes = FXSmileQuotes(
        atm_vol=0.12,
        risk_reversal=0.02,
        butterfly=0.005,
    )

    with pytest.raises(FrozenInstanceError):
        quotes.atm_vol = 0.15


def test_zero_volatilities_allowed():
    quotes = FXSmileQuotes(
        atm_vol=0.0,
        risk_reversal=0.0,
        butterfly=0.0,
    )

    assert quotes.wing_volatilities() == (0.0, 0.0)
