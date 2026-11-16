"""Tests for Hull-White one-factor swaption calibration."""

import numpy as np
import pytest

from pyccr.calibration.hull_white_calibration import (
    SwaptionQuote,
    calibrate_hull_white,
)
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.bachelier_swaption import implied_normal_volatility
from pyccr.pricing.hull_white_swaption import hull_white_swaption_price


@pytest.fixture
def curve():
    return SmoothDiscountCurve(
        tenors=np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
        zero_rates=np.array([0.02, 0.0205, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026]),
    )


def synthetic_quotes(curve, mean_reversion=0.08, volatility=0.012):
    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=mean_reversion,
        volatility=volatility,
    )

    schedules = (
        (1.0, (1.5, 2.0, 2.5, 3.0)),
        (2.0, (2.5, 3.0, 3.5, 4.0)),
        (3.0, (3.5, 4.0, 4.5, 5.0)),
        (5.0, (5.5, 6.0, 6.5, 7.0)),
    )

    quotes = []

    for expiry, payment_times in schedules:
        times = np.asarray(payment_times)
        accruals = np.diff(np.concatenate(([expiry], times)))
        discounts = np.asarray(curve.discount_factor(times))

        annuity = float(np.dot(accruals, discounts))
        forward = float(
            (curve.discount_factor(expiry) - curve.discount_factor(times[-1])) / annuity
        )

        price = hull_white_swaption_price(
            model=model,
            expiry=expiry,
            payment_times=payment_times,
            fixed_rate=forward,
        )

        normal_volatility = implied_normal_volatility(
            price=price,
            forward=forward,
            strike=forward,
            expiry=expiry,
            annuity=annuity,
        )

        quotes.append(
            SwaptionQuote(
                expiry=expiry,
                payment_times=payment_times,
                normal_volatility=normal_volatility,
            )
        )

    return tuple(quotes)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expiry": 0.0}, "expiry"),
        ({"normal_volatility": -0.01}, "normal_volatility"),
        ({"strike": -0.01}, "strike"),
        ({"option_type": "invalid"}, "option_type"),
        ({"payment_times": (1.0,)}, "payment_times"),
        ({"payment_times": (2.0, 1.5)}, "payment_times"),
    ],
)
def test_invalid_swaption_quotes(kwargs, message):
    arguments = {
        "expiry": 1.0,
        "payment_times": (1.5, 2.0),
        "normal_volatility": 0.01,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SwaptionQuote(**arguments)


def test_calibration_requires_quotes(curve):
    with pytest.raises(ValueError, match="at least one"):
        calibrate_hull_white(curve, ())


def test_calibration_rejects_invalid_initial_parameters(curve):
    quotes = synthetic_quotes(curve)

    with pytest.raises(ValueError, match="initial parameters"):
        calibrate_hull_white(
            curve,
            quotes,
            initial_mean_reversion=-0.1,
        )


def test_calibration_rejects_maturity_beyond_curve(curve):
    quote = SwaptionQuote(
        expiry=9.0,
        payment_times=(10.5,),
        normal_volatility=0.01,
    )

    with pytest.raises(ValueError, match="curve horizon"):
        calibrate_hull_white(curve, (quote,))


def test_synthetic_parameter_recovery(curve):
    true_mean_reversion = 0.08
    true_volatility = 0.012

    quotes = synthetic_quotes(
        curve,
        mean_reversion=true_mean_reversion,
        volatility=true_volatility,
    )

    result = calibrate_hull_white(
        curve,
        quotes,
        initial_mean_reversion=0.05,
        initial_volatility=0.01,
    )

    assert result.success
    assert result.rmse < 1e-7
    assert result.mean_reversion == pytest.approx(true_mean_reversion, abs=0.01)
    assert result.volatility == pytest.approx(true_volatility, abs=0.001)


def test_calibrated_model_reprices_synthetic_quotes(curve):
    quotes = synthetic_quotes(curve)
    result = calibrate_hull_white(curve, quotes)

    assert result.success

    for quote in quotes:
        times = np.asarray(quote.payment_times)
        accruals = np.diff(np.concatenate(([quote.expiry], times)))
        discounts = np.asarray(curve.discount_factor(times))

        annuity = float(np.dot(accruals, discounts))
        forward = float(
            (curve.discount_factor(quote.expiry) - curve.discount_factor(times[-1]))
            / annuity
        )

        model_price = hull_white_swaption_price(
            model=result.model,
            expiry=quote.expiry,
            payment_times=quote.payment_times,
            fixed_rate=forward,
        )

        implied_volatility = implied_normal_volatility(
            price=model_price,
            forward=forward,
            strike=forward,
            expiry=quote.expiry,
            annuity=annuity,
        )

        assert implied_volatility == pytest.approx(
            quote.normal_volatility,
            abs=1e-5,
        )
