"""Tests for mixed Hull-White IRS and swaption portfolios."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_ir_portfolio import (
    HullWhiteIRPortfolioValues,
    value_hull_white_ir_portfolio,
)
from pyccr.pricing.hull_white_swap_exposure import (
    value_hull_white_swap_paths,
)
from pyccr.pricing.hull_white_swaption_exposure import (
    HullWhiteSwaptionTrade,
    value_hull_white_swaption_paths,
)
from pyccr.pricing.interest_rate_swap import InterestRateSwap


@pytest.fixture
def model() -> HullWhiteCurveModel:
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
        zero_rates=(0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=0.01,
    )


@pytest.fixture
def times() -> np.ndarray:
    return np.linspace(0.0, 2.0, 9)


@pytest.fixture
def short_rates(times: np.ndarray) -> np.ndarray:
    return np.array(
        [
            0.03 + 0.002 * times,
            0.03 - 0.003 * times,
            0.03 + 0.015 * times,
            0.03 - 0.012 * times,
        ],
        dtype=np.float64,
    )


@pytest.fixture
def swap() -> InterestRateSwap:
    return InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR",
        notional=1_000_000.0,
        fixed_rate=0.03,
        payment_times=(0.0, 0.5, 1.0, 1.5, 2.0),
        pay_fixed=True,
    )


@pytest.fixture
def swaption() -> HullWhiteSwaptionTrade:
    return HullWhiteSwaptionTrade(
        expiry=1.0,
        payment_times=(1.5, 2.0),
        fixed_rate=0.03,
        notional=1_000_000.0,
        option_type="payer",
    )


def test_mixed_portfolio_equals_individual_sum(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
    swap: InterestRateSwap,
    swaption: HullWhiteSwaptionTrade,
) -> None:
    result = value_hull_white_ir_portfolio(
        [swap, swaption],
        model,
        times,
        short_rates,
    )

    expected_swap = value_hull_white_swap_paths(swap, model, times, short_rates)
    expected_swaption = value_hull_white_swaption_paths(
        swaption, model, times, short_rates
    )

    assert isinstance(result, HullWhiteIRPortfolioValues)
    assert result.trade_values.shape == (2, 4, 9)
    assert result.portfolio_values.shape == (4, 9)

    np.testing.assert_allclose(
        result.trade_values[0],
        expected_swap,
    )
    np.testing.assert_allclose(
        result.trade_values[1],
        expected_swaption,
    )
    np.testing.assert_allclose(
        result.portfolio_values,
        expected_swap + expected_swaption,
    )


def test_trade_order_preserved(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
    swap: InterestRateSwap,
    swaption: HullWhiteSwaptionTrade,
) -> None:
    first = value_hull_white_ir_portfolio([swap, swaption], model, times, short_rates)
    reversed_result = value_hull_white_ir_portfolio(
        [swaption, swap], model, times, short_rates
    )

    np.testing.assert_allclose(
        first.trade_values[0],
        reversed_result.trade_values[1],
    )
    np.testing.assert_allclose(
        first.trade_values[1],
        reversed_result.trade_values[0],
    )
    np.testing.assert_allclose(
        first.portfolio_values,
        reversed_result.portfolio_values,
    )


def test_opposite_swaps_cancel(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
    swap: InterestRateSwap,
) -> None:
    opposite = InterestRateSwap(
        currency=swap.currency,
        floating_index=swap.floating_index,
        notional=swap.notional,
        fixed_rate=swap.fixed_rate,
        payment_times=swap.payment_times,
        pay_fixed=False,
    )

    result = value_hull_white_ir_portfolio([swap, opposite], model, times, short_rates)

    np.testing.assert_allclose(
        result.portfolio_values,
        0.0,
        atol=1e-8,
    )


def test_empty_portfolio_returns_zero_values(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
) -> None:
    result = value_hull_white_ir_portfolio([], model, times, short_rates)

    assert result.trade_values.shape == (0, 4, 9)
    assert result.portfolio_values.shape == (4, 9)
    np.testing.assert_array_equal(
        result.portfolio_values,
        np.zeros((4, 9)),
    )


def test_single_swap_portfolio(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
    swap: InterestRateSwap,
) -> None:
    result = value_hull_white_ir_portfolio([swap], model, times, short_rates)

    expected = value_hull_white_swap_paths(swap, model, times, short_rates)

    np.testing.assert_allclose(
        result.portfolio_values,
        expected,
    )


def test_single_swaption_portfolio(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
    swaption: HullWhiteSwaptionTrade,
) -> None:
    result = value_hull_white_ir_portfolio([swaption], model, times, short_rates)

    expected = value_hull_white_swaption_paths(swaption, model, times, short_rates)

    np.testing.assert_allclose(
        result.portfolio_values,
        expected,
    )


def test_portfolio_at_final_settlement_is_zero(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
    swap: InterestRateSwap,
    swaption: HullWhiteSwaptionTrade,
) -> None:
    result = value_hull_white_ir_portfolio([swap, swaption], model, times, short_rates)

    np.testing.assert_allclose(
        result.portfolio_values[:, -1],
        0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "invalid_trades",
    [
        None,
        3,
        "not a portfolio",
        {"trade": "invalid"},
    ],
)
def test_invalid_trade_container(
    invalid_trades: object,
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
) -> None:
    with pytest.raises(TypeError):
        value_hull_white_ir_portfolio(
            invalid_trades,
            model,
            times,
            short_rates,
        )


def test_invalid_trade_type(
    model: HullWhiteCurveModel,
    times: np.ndarray,
    short_rates: np.ndarray,
) -> None:
    with pytest.raises(TypeError):
        value_hull_white_ir_portfolio(
            [object()],
            model,
            times,
            short_rates,
        )


def test_invalid_model(
    times: np.ndarray,
    short_rates: np.ndarray,
    swap: InterestRateSwap,
) -> None:
    with pytest.raises(TypeError):
        value_hull_white_ir_portfolio(
            [swap],
            None,
            times,
            short_rates,
        )


@pytest.mark.parametrize(
    "invalid_times",
    [
        [0.25, 0.5, 1.0],
        [0.0, 0.5, 0.5],
        [0.0, np.nan, 1.0],
        [[0.0, 0.5, 1.0]],
    ],
)
def test_invalid_time_grid(
    invalid_times: object,
    model: HullWhiteCurveModel,
) -> None:
    rates = np.full((4, 3), 0.03)

    with pytest.raises(ValueError):
        value_hull_white_ir_portfolio([], model, invalid_times, rates)


@pytest.mark.parametrize(
    "invalid_rates",
    [
        np.ones(9),
        np.ones((4, 8)),
        np.ones((0, 9)),
        np.full((4, 9), np.nan),
    ],
)
def test_invalid_short_rate_paths(
    invalid_rates: np.ndarray,
    model: HullWhiteCurveModel,
    times: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        value_hull_white_ir_portfolio([], model, times, invalid_rates)
