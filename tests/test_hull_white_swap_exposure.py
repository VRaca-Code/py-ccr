"""Regression tests for pathwise Hull-White IRS valuation."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_swap_exposure import (
    value_hull_white_swap_paths,
)
from pyccr.pricing.interest_rate_swap import InterestRateSwap
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.hull_white_paths import (
    simulate_hull_white_short_rates,
)
from pyccr.simulation.time_grid import TimeGrid


@pytest.fixture
def model():
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
        zero_rates=(0.025, 0.026, 0.027, 0.029, 0.030, 0.032),
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=0.01,
    )


@pytest.fixture
def swap():
    return InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=1_000_000.0,
        fixed_rate=0.03,
        payment_times=(0.0, 0.5, 1.0, 1.5, 2.0),
        pay_fixed=True,
    )


@pytest.fixture
def paths(model):
    grid = TimeGrid.uniform(2.0, 8)
    config = SimulationConfig(
        time_grid=grid,
        n_paths=200,
        seed=42,
        antithetic=True,
    )
    rates = simulate_hull_white_short_rates(config, model)
    return np.asarray(grid.times), rates


def test_output_shape_and_finiteness(swap, model, paths):
    times, rates = paths
    values = value_hull_white_swap_paths(swap, model, times, rates)
    assert values.shape == rates.shape
    assert np.all(np.isfinite(values))


def test_initial_value_matches_single_curve_identity(swap, model, paths):
    times, rates = paths
    values = value_hull_white_swap_paths(swap, model, times, rates)

    schedule = np.asarray(swap.payment_times)
    accruals = np.diff(schedule)
    discounts = model.curve.discount_factor(schedule[1:])

    expected = swap.notional * (
        1.0
        - float(model.curve.discount_factor(schedule[-1]))
        - swap.fixed_rate * float(np.dot(accruals, discounts))
    )

    np.testing.assert_allclose(
        values[:, 0],
        expected,
        rtol=1e-11,
        atol=1e-7,
    )


def test_payer_receiver_symmetry(swap, model, paths):
    times, rates = paths

    receiver = InterestRateSwap(
        currency=swap.currency,
        floating_index=swap.floating_index,
        notional=swap.notional,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
        pay_fixed=False,
    )

    payer_values = value_hull_white_swap_paths(swap, model, times, rates)
    receiver_values = value_hull_white_swap_paths(receiver, model, times, rates)

    np.testing.assert_allclose(
        payer_values + receiver_values,
        0.0,
        atol=1e-9,
    )


def test_notional_scaling(swap, model, paths):
    times, rates = paths

    doubled = InterestRateSwap(
        currency=swap.currency,
        floating_index=swap.floating_index,
        notional=2.0 * swap.notional,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
        pay_fixed=swap.pay_fixed,
    )

    base = value_hull_white_swap_paths(swap, model, times, rates)
    scaled = value_hull_white_swap_paths(doubled, model, times, rates)

    np.testing.assert_allclose(
        scaled,
        2.0 * base,
        rtol=1e-12,
        atol=1e-8,
    )


def test_floating_coupon_remains_fixed_after_reset(model):
    swap = InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=1_000_000.0,
        fixed_rate=0.0,
        payment_times=(0.0, 1.0, 2.0),
    )

    times = np.array([0.0, 1.0, 1.5, 2.0])

    initial_rate = float(model.curve.instantaneous_forward(0.0))

    # Both paths have the same reset rate at t=1.
    # They differ at t=1.5, affecting discounting only.
    rates = np.array(
        [
            [initial_rate, 0.03, 0.02, 0.02],
            [initial_rate, 0.03, 0.06, 0.06],
        ]
    )

    values = value_hull_white_swap_paths(swap, model, times, rates)

    reset_bond = float(model.bond_price(1.0, 2.0, 0.03))
    fixed_coupon = 1.0 / reset_bond - 1.0

    for path in range(2):
        discount = float(model.bond_price(1.5, 2.0, rates[path, 2]))
        expected = swap.notional * fixed_coupon * discount
        assert values[path, 2] == pytest.approx(expected, rel=1e-12)


def test_final_settlement_value_is_zero(swap, model, paths):
    times, rates = paths
    values = value_hull_white_swap_paths(swap, model, times, rates)

    np.testing.assert_array_equal(values[:, -1], 0.0)


def test_zero_volatility_deterministic():
    curve = SmoothDiscountCurve(
        tenors=(0.0, 1.0, 2.0, 3.0),
        zero_rates=(0.03, 0.03, 0.03, 0.03),
    )
    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.1,
        volatility=0.0,
    )
    swap = InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=1_000_000.0,
        fixed_rate=np.expm1(0.03),
        payment_times=(0.0, 1.0, 2.0),
    )

    config = SimulationConfig(
        TimeGrid.uniform(2.0, 4),
        n_paths=10,
        seed=7,
    )

    rates = simulate_hull_white_short_rates(config, model)

    values = value_hull_white_swap_paths(
        swap,
        model,
        config.time_grid.times,
        rates,
    )

    np.testing.assert_allclose(
        values,
        0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "bad_times",
    [
        (0.5, 1.0, 2.0),
        (0.0, 1.0, 1.0),
        (0.0, np.nan, 2.0),
        (0.0,),
    ],
)
def test_invalid_time_grid(swap, model, bad_times):
    rates = np.zeros((3, len(bad_times)))

    with pytest.raises(ValueError):
        value_hull_white_swap_paths(swap, model, bad_times, rates)


@pytest.mark.parametrize(
    "bad_rates",
    [
        np.zeros((3, 3)),
        np.zeros((4,)),
        np.zeros((0, 5)),
        np.full((3, 5), np.nan),
    ],
)
def test_invalid_short_rate_paths(swap, model, bad_rates):
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    with pytest.raises(ValueError):
        value_hull_white_swap_paths(swap, model, times, bad_rates)


def test_missing_reset_date_rejected(swap, model):
    times = np.array([0.0, 0.5, 1.0, 2.0])
    rates = np.full((3, len(times)), 0.03)

    with pytest.raises(ValueError, match="schedule"):
        value_hull_white_swap_paths(swap, model, times, rates)


def test_maturity_exceeds_curve_horizon_rejected(model):
    swap = InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR6M",
        notional=1_000_000.0,
        fixed_rate=0.03,
        payment_times=(0.0, 1.0, 6.0),
    )

    times = np.array([0.0, 1.0, 6.0])
    rates = np.full((3, 3), 0.03)

    with pytest.raises(ValueError, match="horizon"):
        value_hull_white_swap_paths(swap, model, times, rates)
