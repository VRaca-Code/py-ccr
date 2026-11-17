"""Regression tests for FX-forward portfolio valuation."""

import numpy as np
import pytest

from pyccr.pricing.fx_forward_exposure import (
    FXForwardExposureTrade,
    value_fx_forward_paths,
)
from pyccr.pricing.fx_forward_portfolio import (
    FXForwardPortfolioValues,
    value_fx_forward_portfolio,
)


def test_single_trade_matches_standalone_valuation() -> None:
    trade = FXForwardExposureTrade(1_000_000.0, 1.12, 2.0)
    times = np.array([0.0, 0.5, 1.0, 2.0])
    spots = np.array([[1.10, 1.15, 1.20, 1.25]])

    result = value_fx_forward_portfolio(
        [trade],
        times,
        spots,
        domestic_rate=0.04,
        foreign_rate=0.02,
    )

    expected = value_fx_forward_paths(
        trade,
        times,
        spots,
        domestic_rate=0.04,
        foreign_rate=0.02,
    )

    assert isinstance(result, FXForwardPortfolioValues)
    assert result.trade_values.shape == (1, 1, 4)
    assert result.portfolio_values.shape == (1, 4)
    np.testing.assert_array_equal(result.trade_values[0], expected)
    np.testing.assert_array_equal(result.portfolio_values, expected)


def test_multiple_trades_aggregate_signed_values() -> None:
    trades = [
        FXForwardExposureTrade(100.0, 1.10, 2.0),
        FXForwardExposureTrade(250.0, 1.15, 1.5),
        FXForwardExposureTrade(-75.0, 1.20, 2.0),
    ]
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    spots = np.array(
        [
            [1.10, 1.12, 1.15, 1.18, 1.20],
            [1.10, 1.08, 1.05, 1.03, 1.00],
        ]
    )

    result = value_fx_forward_portfolio(
        trades,
        times,
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    expected_trades = np.stack(
        [
            value_fx_forward_paths(
                trade,
                times,
                spots,
                domestic_rate=0.03,
                foreign_rate=0.01,
            )
            for trade in trades
        ]
    )

    assert result.trade_values.shape == (3, 2, 5)
    assert result.portfolio_values.shape == (2, 5)
    np.testing.assert_allclose(result.trade_values, expected_trades)
    np.testing.assert_allclose(result.portfolio_values, expected_trades.sum(axis=0))


def test_exactly_offsetting_positions_cancel() -> None:
    trades = [
        FXForwardExposureTrade(1000.0, 1.12, 2.0),
        FXForwardExposureTrade(-1000.0, 1.12, 2.0),
    ]
    spots = np.array(
        [
            [1.10, 1.15, 1.20],
            [1.10, 1.05, 0.95],
        ]
    )

    result = value_fx_forward_portfolio(
        trades,
        [0.0, 0.5, 1.0],
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    np.testing.assert_allclose(
        result.portfolio_values,
        np.zeros((2, 3)),
        atol=1e-12,
    )


def test_different_maturities_settle_independently() -> None:
    trades = [
        FXForwardExposureTrade(100.0, 1.10, 0.5),
        FXForwardExposureTrade(200.0, 1.15, 1.0),
    ]

    result = value_fx_forward_portfolio(
        trades,
        [0.0, 0.5, 0.75, 1.0, 1.5],
        [[1.10, 1.20, 1.25, 1.30, 1.35]],
        domestic_rate=0.0,
        foreign_rate=0.0,
    )

    np.testing.assert_array_equal(
        result.trade_values[0, 0, 1:],
        np.zeros(4),
    )
    np.testing.assert_array_equal(
        result.trade_values[1, 0, 3:],
        np.zeros(2),
    )
    assert result.portfolio_values[0, 2] == pytest.approx(20.0)
    assert result.portfolio_values[0, 4] == 0.0


def test_empty_portfolio_returns_zero_values() -> None:
    spots = np.array(
        [
            [1.10, 1.20, 1.30],
            [1.10, 1.05, 0.95],
        ]
    )

    result = value_fx_forward_portfolio(
        [],
        [0.0, 0.5, 1.0],
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    assert result.trade_values.shape == (0, 2, 3)
    assert result.portfolio_values.shape == (2, 3)
    np.testing.assert_array_equal(result.portfolio_values, np.zeros((2, 3)))


def test_tuple_of_trades_is_accepted() -> None:
    trades = (
        FXForwardExposureTrade(100.0, 1.10, 1.0),
        FXForwardExposureTrade(-50.0, 1.15, 1.0),
    )

    result = value_fx_forward_portfolio(
        trades,
        [0.0, 0.5],
        [[1.10, 1.20]],
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    assert result.trade_values.shape == (2, 1, 2)


def test_portfolio_values_are_not_gross_positive_exposures() -> None:
    trades = [
        FXForwardExposureTrade(100.0, 1.10, 2.0),
        FXForwardExposureTrade(-100.0, 1.00, 2.0),
    ]

    result = value_fx_forward_portfolio(
        trades,
        [0.0, 0.5],
        [[1.10, 1.15]],
        domestic_rate=0.0,
        foreign_rate=0.0,
    )

    expected = result.trade_values.sum(axis=0)

    np.testing.assert_allclose(result.portfolio_values, expected)

    # Signed portfolio value must not be replaced by gross exposure.
    gross_positive = np.maximum(result.trade_values, 0.0).sum(axis=0)
    assert np.any(np.abs(result.portfolio_values - gross_positive) > 1e-12)


@pytest.mark.parametrize(
    "trades",
    [
        None,
        "not a portfolio",
        FXForwardExposureTrade(100.0, 1.10, 1.0),
        {1: FXForwardExposureTrade(100.0, 1.10, 1.0)},
        [object()],
        [FXForwardExposureTrade(100.0, 1.10, 1.0), None],
    ],
)
def test_invalid_trade_collections(trades: object) -> None:
    with pytest.raises(TypeError):
        value_fx_forward_portfolio(
            trades,
            [0.0, 0.5],
            [[1.10, 1.20]],
            domestic_rate=0.03,
            foreign_rate=0.01,
        )


@pytest.mark.parametrize(
    ("times", "spots"),
    [
        ([0.5, 1.0], [[1.10, 1.20]]),
        ([0.0, 0.0], [[1.10, 1.20]]),
        ([0.0, 0.5], [[1.10]]),
        ([0.0, 0.5], [[0.0, 1.20]]),
        ([0.0, 0.5], [[float("nan"), 1.20]]),
        ([0.0, 0.5], []),
    ],
)
def test_invalid_scenarios(
    times: list,
    spots: list,
) -> None:
    with pytest.raises(ValueError):
        value_fx_forward_portfolio(
            [],
            times,
            spots,
            domestic_rate=0.03,
            foreign_rate=0.01,
        )


@pytest.mark.parametrize(
    ("rd", "rf"),
    [
        (float("nan"), 0.01),
        (float("inf"), 0.01),
        (0.03, float("nan")),
        (0.03, float("-inf")),
    ],
)
def test_invalid_rates_for_empty_portfolio(
    rd: float,
    rf: float,
) -> None:
    with pytest.raises(ValueError):
        value_fx_forward_portfolio(
            [],
            [0.0, 0.5],
            [[1.10, 1.20]],
            domestic_rate=rd,
            foreign_rate=rf,
        )


def test_input_spots_are_not_mutated() -> None:
    spots = np.array([[1.10, 1.15, 1.20]])
    original = spots.copy()

    value_fx_forward_portfolio(
        [FXForwardExposureTrade(100.0, 1.12, 2.0)],
        [0.0, 0.5, 1.0],
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    np.testing.assert_array_equal(spots, original)


def test_empty_portfolio_does_not_mutate_spots() -> None:
    spots = np.array([[1.10, 1.15]])
    original = spots.copy()

    value_fx_forward_portfolio(
        [],
        [0.0, 0.5],
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    np.testing.assert_array_equal(spots, original)
