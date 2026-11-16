"""Regression tests for no-basis FX forward pricing."""

import numpy as np
import pytest

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.pricing.fx_forward import FXForward


def make_curves(usd_rate=0.04, eur_rate=0.02):
    usd = DiscountCurve(
        tenors=(1.0, 5.0),
        zero_rates=(usd_rate, usd_rate),
    )
    eur = DiscountCurve(
        tenors=(1.0, 5.0),
        zero_rates=(eur_rate, eur_rate),
    )
    return CurveSet(
        discount_curves={"USD": usd, "EUR": eur},
        projection_curves={},
    )


def test_fx_forward_covered_interest_parity():
    fx = FXForward("USD", "EUR", 1.10)
    curves = make_curves()

    maturity = 2.0
    expected = 1.10 * np.exp((0.04 - 0.02) * maturity)

    assert fx.forward_rate(curves, maturity) == pytest.approx(expected)


def test_fx_forward_zero_maturity_equals_spot():
    fx = FXForward("USD", "EUR", 1.10)

    assert fx.forward_rate(make_curves(), 0.0) == pytest.approx(1.10)


def test_fx_forward_equal_rates_equals_spot():
    fx = FXForward("USD", "EUR", 1.10)

    assert fx.forward_rate(
        make_curves(usd_rate=0.03, eur_rate=0.03),
        3.0,
    ) == pytest.approx(1.10)


def test_fx_forward_reciprocal_currency_quotes():
    curves = make_curves()

    eurusd = FXForward("USD", "EUR", 1.10)
    usdeur = FXForward("EUR", "USD", 1.0 / 1.10)

    assert (
        eurusd.forward_rate(curves, 2.0) * usdeur.forward_rate(curves, 2.0)
    ) == pytest.approx(1.0)


def test_fx_forward_increases_with_domestic_rate():
    fx = FXForward("USD", "EUR", 1.10)

    low = make_curves(usd_rate=0.03)
    high = make_curves(usd_rate=0.05)

    assert fx.forward_rate(high, 2.0) > fx.forward_rate(low, 2.0)


def test_fx_forward_decreases_with_foreign_rate():
    fx = FXForward("USD", "EUR", 1.10)

    low = make_curves(eur_rate=0.01)
    high = make_curves(eur_rate=0.03)

    assert fx.forward_rate(high, 2.0) < fx.forward_rate(low, 2.0)


@pytest.mark.parametrize(
    ("domestic", "foreign", "spot"),
    [
        ("USD", "USD", 1.10),
        ("", "EUR", 1.10),
        ("USD", "", 1.10),
        ("USD", "EUR", 0.0),
        ("USD", "EUR", -1.0),
        ("USD", "EUR", np.nan),
        ("USD", "EUR", np.inf),
    ],
)
def test_fx_forward_invalid_inputs(domestic, foreign, spot):
    with pytest.raises(ValueError):
        FXForward(domestic, foreign, spot)


@pytest.mark.parametrize("maturity", [-1.0, np.nan, np.inf])
def test_fx_forward_invalid_maturity(maturity):
    fx = FXForward("USD", "EUR", 1.10)

    with pytest.raises(ValueError):
        fx.forward_rate(make_curves(), maturity)


def test_fx_forward_missing_currency_curve():
    fx = FXForward("USD", "EUR", 1.10)

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
        fx.forward_rate(curves, 2.0)


def test_fx_forward_normalizes_currency_codes():
    fx = FXForward(" usd ", " eur ", 1.10)

    assert fx.domestic_currency == "USD"
    assert fx.foreign_currency == "EUR"
