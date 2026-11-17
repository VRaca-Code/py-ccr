"""Regression tests for pathwise Hull-White European swaptions."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_swaption import hull_white_swaption_price
from pyccr.pricing.hull_white_swaption_exposure import (
    HullWhiteSwaptionTrade,
    value_hull_white_swaption_paths,
)


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
def setup(model):
    expiry = 1.0
    payments = (1.5, 2.0, 2.5, 3.0)
    schedule = np.asarray((expiry, *payments))
    accruals = np.diff(schedule)
    discounts = np.asarray(model.curve.discount_factor(payments))

    fixed_rate = (
        float(model.curve.discount_factor(expiry))
        - float(model.curve.discount_factor(payments[-1]))
    ) / float(np.dot(accruals, discounts))

    times = np.arange(0.0, 3.25, 0.25)
    initial_rate = float(model.curve.instantaneous_forward(0.0))

    return {
        "expiry": expiry,
        "payments": payments,
        "fixed_rate": fixed_rate,
        "times": times,
        "initial_rate": initial_rate,
    }


def make_trade(setup, option_type="payer", **overrides):
    params = {
        "expiry": setup["expiry"],
        "payment_times": setup["payments"],
        "fixed_rate": setup["fixed_rate"],
        "notional": 1_000_000.0,
        "option_type": option_type,
    }
    params.update(overrides)
    return HullWhiteSwaptionTrade(**params)


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_time_zero_matches_analytical_price(model, setup, option_type):
    trade = make_trade(setup, option_type)
    rates = np.full(
        (4, setup["times"].size),
        setup["initial_rate"],
    )

    values = value_hull_white_swaption_paths(trade, model, setup["times"], rates)

    analytical = hull_white_swaption_price(
        model=model,
        expiry=trade.expiry,
        payment_times=trade.payment_times,
        fixed_rate=trade.fixed_rate,
        notional=trade.notional,
        option_type=trade.option_type,
    )

    np.testing.assert_allclose(
        values[:, 0],
        analytical,
        rtol=1e-10,
        atol=1e-6,
    )


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_expiry_equals_intrinsic_value(model, setup, option_type):
    trade = make_trade(setup, option_type)

    rates = np.full(
        (3, setup["times"].size),
        setup["initial_rate"],
    )

    expiry_index = np.searchsorted(setup["times"], trade.expiry)

    rates[:, expiry_index] = [0.005, 0.04, 0.09]

    values = value_hull_white_swaption_paths(trade, model, setup["times"], rates)

    accruals = np.diff((trade.expiry, *trade.payment_times))

    bonds = np.column_stack(
        [
            model.bond_price(
                trade.expiry,
                maturity,
                rates[:, expiry_index],
            )
            for maturity in trade.payment_times
        ]
    )

    floating_leg = 1.0 - bonds[:, -1]
    fixed_leg = trade.fixed_rate * (bonds @ accruals)

    payer_mtm = trade.notional * (floating_leg - fixed_leg)

    signed_mtm = payer_mtm if option_type == "payer" else -payer_mtm

    np.testing.assert_allclose(
        values[:, expiry_index],
        np.maximum(signed_mtm, 0.0),
        rtol=1e-10,
        atol=1e-6,
    )


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_unexercised_paths_remain_zero(model, setup, option_type):
    trade = make_trade(setup, option_type)

    rates = np.full(
        (2, setup["times"].size),
        setup["initial_rate"],
    )

    expiry_index = np.searchsorted(setup["times"], trade.expiry)

    # Low rates favor receivers; high rates favor payers.
    rates[0, expiry_index] = -0.02
    rates[1, expiry_index] = 0.12

    values = value_hull_white_swaption_paths(trade, model, setup["times"], rates)

    unexercised_path = 0 if option_type == "payer" else 1

    assert np.all(values[unexercised_path, expiry_index:] == 0.0)


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_exercised_swap_can_have_negative_future_mtm(model, setup, option_type):
    trade = make_trade(setup, option_type)

    rates = np.full(
        (2, setup["times"].size),
        setup["initial_rate"],
    )

    expiry_index = np.searchsorted(setup["times"], trade.expiry)

    # Force exercise at expiry, then reverse the rate move.
    if option_type == "payer":
        rates[0, expiry_index] = 0.12
        rates[0, expiry_index + 1 :] = -0.03
    else:
        rates[0, expiry_index] = -0.03
        rates[0, expiry_index + 1 :] = 0.12

    values = value_hull_white_swaption_paths(trade, model, setup["times"], rates)

    assert values[0, expiry_index] > 0.0
    assert np.any(values[0, expiry_index + 1 : -1] < 0.0)


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_no_look_ahead_before_expiry(model, setup, option_type):
    trade = make_trade(setup, option_type)

    rates_a = np.full(
        (2, setup["times"].size),
        setup["initial_rate"],
    )
    rates_b = rates_a.copy()

    expiry_index = np.searchsorted(setup["times"], trade.expiry)

    # Change only information at and after expiry.
    rates_b[:, expiry_index:] = np.array([[0.12], [-0.03]])

    values_a = value_hull_white_swaption_paths(trade, model, setup["times"], rates_a)
    values_b = value_hull_white_swaption_paths(trade, model, setup["times"], rates_b)

    np.testing.assert_allclose(
        values_a[:, :expiry_index],
        values_b[:, :expiry_index],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_notional_scaling(model, setup, option_type):
    small = make_trade(setup, option_type, notional=1_000_000.0)
    large = make_trade(setup, option_type, notional=5_000_000.0)

    rates = np.full(
        (2, setup["times"].size),
        setup["initial_rate"],
    )

    values_small = value_hull_white_swaption_paths(small, model, setup["times"], rates)
    values_large = value_hull_white_swaption_paths(large, model, setup["times"], rates)

    np.testing.assert_allclose(
        values_large,
        5.0 * values_small,
        rtol=1e-12,
        atol=1e-6,
    )


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_final_settlement_is_zero(model, setup, option_type):
    trade = make_trade(setup, option_type)

    rates = np.full(
        (3, setup["times"].size),
        setup["initial_rate"],
    )

    values = value_hull_white_swaption_paths(trade, model, setup["times"], rates)

    assert np.all(values[:, -1] == 0.0)


@pytest.mark.parametrize(
    "invalid_expiry",
    [-1.0, 0.0, float("nan"), float("inf")],
)
def test_rejects_invalid_expiry(setup, invalid_expiry):
    with pytest.raises(ValueError):
        make_trade(setup, expiry=invalid_expiry)


@pytest.mark.parametrize(
    "invalid_option",
    ["call", "put", "invalid"],
)
def test_rejects_invalid_option_type(setup, invalid_option):
    with pytest.raises(ValueError):
        make_trade(setup, option_type=invalid_option)


def test_rejects_missing_expiry_on_grid(model, setup):
    trade = make_trade(setup)

    times = np.array([0.0, 0.5, 1.5, 2.0, 2.5, 3.0])
    rates = np.full(
        (2, times.size),
        setup["initial_rate"],
    )

    with pytest.raises(ValueError, match="simulation grid"):
        value_hull_white_swaption_paths(trade, model, times, rates)


def test_rejects_invalid_rate_matrix(model, setup):
    trade = make_trade(setup)

    with pytest.raises(ValueError, match="short_rates"):
        value_hull_white_swaption_paths(
            trade,
            model,
            setup["times"],
            np.ones((2, 3)),
        )
