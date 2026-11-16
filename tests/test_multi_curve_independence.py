"""Regression tests for independent discount and projection curves."""

import numpy as np
import pytest

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.pricing.interest_rate_swap import InterestRateSwap


@pytest.fixture
def scenarios():
    def curve(rates):
        return DiscountCurve(
            tenors=(1.0, 5.0, 10.0),
            zero_rates=rates,
        )

    ois = curve((0.020, 0.025, 0.028))
    higher_ois = curve((0.030, 0.035, 0.038))
    forward = curve((0.025, 0.030, 0.033))
    higher_forward = curve((0.035, 0.040, 0.043))

    def market(discount, projection):
        return CurveSet(
            discount_curves={"EUR": discount},
            projection_curves={("EUR", "EURIBOR6M"): projection},
        )

    return (
        market(ois, forward),
        market(ois, higher_forward),
        market(higher_ois, forward),
    )


@pytest.fixture
def swap():
    return InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=10_000_000,
        fixed_rate=0.03,
        payment_times=tuple(np.arange(0.0, 5.5, 0.5)),
    )


def test_forward_shift_preserves_discount_factors(swap, scenarios):
    base, higher_forward, _ = scenarios
    np.testing.assert_allclose(
        swap._cashflow_components(base)[1],
        swap._cashflow_components(higher_forward)[1],
    )


def test_forward_shift_changes_projected_rates(swap, scenarios):
    base, higher_forward, _ = scenarios
    assert not np.allclose(
        swap._cashflow_components(base)[2],
        swap._cashflow_components(higher_forward)[2],
    )


def test_discount_shift_preserves_projected_rates(swap, scenarios):
    base, _, higher_ois = scenarios
    np.testing.assert_allclose(
        swap._cashflow_components(base)[2],
        swap._cashflow_components(higher_ois)[2],
    )


def test_discount_shift_changes_discount_factors(swap, scenarios):
    base, _, higher_ois = scenarios
    assert not np.allclose(
        swap._cashflow_components(base)[1],
        swap._cashflow_components(higher_ois)[1],
    )


def test_multi_curve_par_rate(swap, scenarios):
    base, _, _ = scenarios

    times = np.asarray(swap.payment_times)
    accruals = np.diff(times)

    discount = base.discount_curve("EUR")
    projection = base.projection_curve("EUR", "EURIBOR6M")

    dfs = discount.discount_factor(times[1:])
    forwards = (
        projection.discount_factor(times[:-1]) / projection.discount_factor(times[1:])
        - 1.0
    ) / accruals

    expected = np.sum(accruals * dfs * forwards) / np.sum(accruals * dfs)

    assert swap.par_rate(base) == pytest.approx(expected)


def test_multi_curve_npv(swap, scenarios):
    base, _, _ = scenarios

    times = np.asarray(swap.payment_times)
    accruals = np.diff(times)

    discount = base.discount_curve("EUR")
    projection = base.projection_curve("EUR", "EURIBOR6M")

    dfs = discount.discount_factor(times[1:])
    forwards = (
        projection.discount_factor(times[:-1]) / projection.discount_factor(times[1:])
        - 1.0
    ) / accruals

    expected = swap.notional * np.sum(accruals * dfs * (forwards - swap.fixed_rate))

    assert swap.npv(base) == pytest.approx(expected)
