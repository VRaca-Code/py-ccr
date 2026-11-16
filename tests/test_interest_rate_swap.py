"""Tests for multi-curve interest-rate swap pricing."""

import numpy as np
import pytest

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.pricing.interest_rate_swap import InterestRateSwap


@pytest.fixture
def curves() -> CurveSet:
    discount = DiscountCurve(
        tenors=(1.0, 5.0, 10.0),
        zero_rates=(0.025, 0.028, 0.030),
    )
    projection = DiscountCurve(
        tenors=(1.0, 5.0, 10.0),
        zero_rates=(0.028, 0.031, 0.033),
    )
    return CurveSet(
        discount_curves={"EUR": discount},
        projection_curves={("EUR", "EURIBOR6M"): projection},
    )


@pytest.fixture
def swap() -> InterestRateSwap:
    return InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=10_000_000,
        fixed_rate=0.03,
        payment_times=tuple(np.arange(0.0, 5.5, 0.5)),
    )


def test_par_swap_has_zero_npv(
    swap: InterestRateSwap,
    curves: CurveSet,
) -> None:
    par_swap = InterestRateSwap(
        currency=swap.currency,
        floating_index=swap.floating_index,
        notional=swap.notional,
        fixed_rate=swap.par_rate(curves),
        payment_times=swap.payment_times,
    )

    assert par_swap.npv(curves) == pytest.approx(0.0, abs=1e-7)


def test_pay_receive_symmetry(
    swap: InterestRateSwap,
    curves: CurveSet,
) -> None:
    receive_fixed = InterestRateSwap(
        currency=swap.currency,
        floating_index=swap.floating_index,
        notional=swap.notional,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
        pay_fixed=False,
    )

    assert swap.npv(curves) == pytest.approx(
        -receive_fixed.npv(curves),
        abs=1e-7,
    )


def test_notional_scaling(
    swap: InterestRateSwap,
    curves: CurveSet,
) -> None:
    doubled = InterestRateSwap(
        currency=swap.currency,
        floating_index=swap.floating_index,
        notional=2 * swap.notional,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
    )

    assert doubled.npv(curves) == pytest.approx(
        2 * swap.npv(curves),
    )


def test_annuity_positive(
    swap: InterestRateSwap,
    curves: CurveSet,
) -> None:
    assert swap.annuity(curves) > 0


def test_par_rate_above_fixed_rate(
    swap: InterestRateSwap,
    curves: CurveSet,
) -> None:
    assert swap.par_rate(curves) > swap.fixed_rate
    assert swap.npv(curves) > 0


def test_single_curve_par_rate_identity() -> None:
    curve = DiscountCurve(
        tenors=(1.0, 5.0),
        zero_rates=(0.03, 0.03),
    )
    market = CurveSet(
        discount_curves={"EUR": curve},
        projection_curves={("EUR", "EURIBOR6M"): curve},
    )
    swap = InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=1_000_000,
        fixed_rate=0.03,
        payment_times=(0.0, 1.0, 2.0, 3.0),
    )

    expected = (1.0 - curve.discount_factor(3.0)) / swap.annuity(market)

    assert swap.par_rate(market) == pytest.approx(expected)


@pytest.mark.parametrize(
    "times",
    [
        (),
        (0.0,),
        (1.0, 2.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0),
        (0.0, float("nan")),
    ],
)
def test_invalid_payment_times(times) -> None:
    with pytest.raises(ValueError):
        InterestRateSwap(
            currency="EUR",
            floating_index="EURIBOR6M",
            notional=1_000_000,
            fixed_rate=0.03,
            payment_times=times,
        )


def test_invalid_notional() -> None:
    with pytest.raises(ValueError):
        InterestRateSwap(
            currency="EUR",
            floating_index="EURIBOR6M",
            notional=0.0,
            fixed_rate=0.03,
            payment_times=(0.0, 1.0),
        )
