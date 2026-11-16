"""Regression tests for Bachelier implied normal volatility."""

import numpy as np
import pytest

from pyccr.pricing.bachelier_swaption import (
    bachelier_swaption_price,
    implied_normal_volatility,
)


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
@pytest.mark.parametrize("strike", [0.02, 0.03, 0.04])
def test_implied_volatility_round_trip(option_type, strike):
    original_vol = 0.0075

    kwargs = {
        "forward": 0.03,
        "strike": strike,
        "expiry": 2.0,
        "annuity": 4.5,
        "option_type": option_type,
    }

    price = bachelier_swaption_price(
        **kwargs,
        normal_volatility=original_vol,
    )

    recovered = implied_normal_volatility(
        price=price,
        **kwargs,
    )

    np.testing.assert_allclose(
        recovered,
        original_vol,
        rtol=1e-9,
        atol=1e-12,
    )


def test_implied_volatility_rejects_price_below_intrinsic():
    with pytest.raises(ValueError):
        implied_normal_volatility(
            price=0.001,
            forward=0.04,
            strike=0.03,
            expiry=2.0,
            annuity=4.0,
        )


def test_zero_price_atm_implies_zero_volatility():
    result = implied_normal_volatility(
        price=0.0,
        forward=0.03,
        strike=0.03,
        expiry=2.0,
        annuity=4.0,
    )

    assert result == 0.0
