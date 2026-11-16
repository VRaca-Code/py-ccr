"""Tests for Hull-White European swaption pricing."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_swaption import hull_white_swaption_price


@pytest.fixture
def curve() -> SmoothDiscountCurve:
    return SmoothDiscountCurve(
        tenors=(0.0, 1.0, 2.0, 5.0, 10.0),
        zero_rates=(0.025, 0.026, 0.027, 0.029, 0.031),
    )


@pytest.fixture
def model(curve: SmoothDiscountCurve) -> HullWhiteCurveModel:
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.05,
        volatility=0.01,
    )


@pytest.fixture
def schedule() -> tuple[float, ...]:
    return (1.5, 2.0, 2.5, 3.0)


def test_payer_and_receiver_nonnegative(model, schedule) -> None:
    for option_type in ("payer", "receiver"):
        price = hull_white_swaption_price(
            model,
            1.0,
            schedule,
            0.03,
            option_type=option_type,
        )
        assert np.isfinite(price)
        assert price >= 0.0


def test_payer_receiver_parity(model, schedule) -> None:
    expiry = 1.0
    fixed_rate = 0.03

    payer = hull_white_swaption_price(
        model,
        expiry,
        schedule,
        fixed_rate,
        option_type="payer",
    )
    receiver = hull_white_swaption_price(
        model,
        expiry,
        schedule,
        fixed_rate,
        option_type="receiver",
    )

    times = np.asarray(schedule)
    accruals = np.diff(np.concatenate(([expiry], times)))
    discounts = np.asarray(model.curve.discount_factor(times))

    annuity = float(np.dot(accruals, discounts))
    floating_leg = float(
        model.curve.discount_factor(expiry) - model.curve.discount_factor(times[-1])
    )

    expected = floating_leg - fixed_rate * annuity

    assert payer - receiver == pytest.approx(expected, abs=1e-10)


def test_notional_scaling(model, schedule) -> None:
    base = hull_white_swaption_price(
        model,
        1.0,
        schedule,
        0.03,
    )
    scaled = hull_white_swaption_price(
        model,
        1.0,
        schedule,
        0.03,
        notional=5_000_000,
    )

    assert scaled == pytest.approx(5_000_000 * base)


def test_payer_decreases_with_strike(model, schedule) -> None:
    low = hull_white_swaption_price(
        model,
        1.0,
        schedule,
        0.02,
    )
    high = hull_white_swaption_price(
        model,
        1.0,
        schedule,
        0.04,
    )

    assert low >= high


def test_receiver_increases_with_strike(model, schedule) -> None:
    low = hull_white_swaption_price(
        model,
        1.0,
        schedule,
        0.02,
        option_type="receiver",
    )
    high = hull_white_swaption_price(
        model,
        1.0,
        schedule,
        0.04,
        option_type="receiver",
    )

    assert high >= low


def test_zero_volatility_intrinsic(curve, schedule) -> None:
    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.05,
        volatility=0.0,
    )

    expiry = 1.0
    strike = 0.03
    times = np.asarray(schedule)
    accruals = np.diff(np.concatenate(([expiry], times)))
    discounts = np.asarray(curve.discount_factor(times))

    swap_value = float(
        curve.discount_factor(expiry)
        - curve.discount_factor(times[-1])
        - strike * np.dot(accruals, discounts)
    )

    payer = hull_white_swaption_price(
        model,
        expiry,
        schedule,
        strike,
        option_type="payer",
    )
    receiver = hull_white_swaption_price(
        model,
        expiry,
        schedule,
        strike,
        option_type="receiver",
    )

    assert payer == pytest.approx(max(swap_value, 0.0), abs=1e-10)
    assert receiver == pytest.approx(max(-swap_value, 0.0), abs=1e-10)


@pytest.mark.parametrize(
    "option_type",
    ["call", "put", "", "invalid"],
)
def test_invalid_option_type(model, schedule, option_type) -> None:
    with pytest.raises(ValueError):
        hull_white_swaption_price(
            model,
            1.0,
            schedule,
            0.03,
            option_type=option_type,
        )


@pytest.mark.parametrize(
    "expiry",
    [-1.0, float("nan"), float("inf")],
)
def test_invalid_expiry(model, schedule, expiry) -> None:
    with pytest.raises(ValueError):
        hull_white_swaption_price(
            model,
            expiry,
            schedule,
            0.03,
        )


@pytest.mark.parametrize(
    "payment_times",
    [
        (),
        (1.0, 2.0),
        (2.0, 1.5),
        (1.5, 1.5),
        (1.5, float("nan")),
        (1.5, 11.0),
    ],
)
def test_invalid_schedule(model, payment_times) -> None:
    with pytest.raises(ValueError):
        hull_white_swaption_price(
            model,
            1.0,
            payment_times,
            0.03,
        )


@pytest.mark.parametrize(
    "fixed_rate",
    [-0.01, float("nan"), float("inf")],
)
def test_invalid_fixed_rate(model, schedule, fixed_rate) -> None:
    with pytest.raises(ValueError):
        hull_white_swaption_price(
            model,
            1.0,
            schedule,
            fixed_rate,
        )


@pytest.mark.parametrize(
    "notional",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_notional(model, schedule, notional) -> None:
    with pytest.raises(ValueError):
        hull_white_swaption_price(
            model,
            1.0,
            schedule,
            0.03,
            notional=notional,
        )
