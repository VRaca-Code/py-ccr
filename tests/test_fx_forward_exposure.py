"""Regression tests for pathwise FX-forward exposure valuation."""

import numpy as np
import pytest

from pyccr.pricing.fx_forward_exposure import (
    FXForwardExposureTrade,
    value_fx_forward_paths,
)


def test_trade_fields() -> None:
    trade = FXForwardExposureTrade(
        foreign_notional=1_000_000.0,
        strike=1.12,
        maturity=2.0,
    )

    assert trade.foreign_notional == 1_000_000.0
    assert trade.strike == 1.12
    assert trade.maturity == 2.0


@pytest.mark.parametrize(
    ("notional", "strike", "maturity"),
    [
        (0.0, 1.1, 1.0),
        (float("nan"), 1.1, 1.0),
        (float("inf"), 1.1, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
        (1.0, 1.1, -1.0),
        (1.0, 1.1, float("nan")),
        (1.0, 1.1, float("inf")),
    ],
)
def test_invalid_trade_parameters(
    notional: float,
    strike: float,
    maturity: float,
) -> None:
    with pytest.raises(ValueError):
        FXForwardExposureTrade(notional, strike, maturity)


def test_analytical_value() -> None:
    trade = FXForwardExposureTrade(2_000_000.0, 1.15, 2.0)
    times = np.array([0.0, 0.5, 1.0])
    spots = np.array([[1.10, 1.20, 1.25]])

    actual = value_fx_forward_paths(
        trade,
        times,
        spots,
        domestic_rate=0.04,
        foreign_rate=0.02,
    )

    remaining = trade.maturity - times
    expected = trade.foreign_notional * (
        spots * np.exp(-0.02 * remaining)[None, :]
        - trade.strike * np.exp(-0.04 * remaining)[None, :]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_opposite_trade_directions() -> None:
    times = [0.0, 0.5]
    spots = [[1.1, 1.2]]

    positive = value_fx_forward_paths(
        FXForwardExposureTrade(100.0, 1.15, 1.0),
        times,
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    negative = value_fx_forward_paths(
        FXForwardExposureTrade(-100.0, 1.15, 1.0),
        times,
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    np.testing.assert_allclose(negative, -positive)


def test_par_forward_has_zero_initial_value() -> None:
    spot = 1.10
    rd = 0.04
    rf = 0.02
    maturity = 2.0

    strike = spot * np.exp((rd - rf) * maturity)

    values = value_fx_forward_paths(
        FXForwardExposureTrade(1_000_000.0, strike, maturity),
        [0.0, 1.0],
        [[spot, 1.15]],
        domestic_rate=rd,
        foreign_rate=rf,
    )

    assert abs(values[0, 0]) < 1e-8


def test_maturity_and_post_maturity_values_zero() -> None:
    trade = FXForwardExposureTrade(100.0, 1.1, 1.0)

    values = value_fx_forward_paths(
        trade,
        [0.0, 0.5, 1.0, 1.5],
        [[1.1, 1.2, 1.3, 1.4]],
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    assert values.shape == (1, 4)
    assert values[0, 1] != 0.0
    np.testing.assert_array_equal(values[0, 2:], [0.0, 0.0])


def test_multiple_paths() -> None:
    trade = FXForwardExposureTrade(100.0, 1.1, 1.0)

    spots = np.array(
        [
            [1.1, 1.2, 1.3],
            [1.1, 1.0, 0.9],
            [1.1, 1.1, 1.1],
        ]
    )

    values = value_fx_forward_paths(
        trade,
        [0.0, 0.5, 1.0],
        spots,
        domestic_rate=0.0,
        foreign_rate=0.0,
    )

    assert values.shape == spots.shape
    np.testing.assert_allclose(
        values[:, 1],
        [10.0, -10.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_array_equal(values[:, 2], np.zeros(3))


def test_zero_maturity() -> None:
    values = value_fx_forward_paths(
        FXForwardExposureTrade(100.0, 1.1, 0.0),
        [0.0, 0.5],
        [[1.2, 1.3]],
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    np.testing.assert_array_equal(values, np.zeros((1, 2)))


@pytest.mark.parametrize(
    "times",
    [
        [],
        [0.1, 0.5],
        [0.0, 0.0],
        [0.0, -0.1],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [[0.0, 0.5]],
    ],
)
def test_invalid_time_grids(times: list) -> None:
    with pytest.raises(ValueError):
        value_fx_forward_paths(
            FXForwardExposureTrade(100.0, 1.1, 1.0),
            times,
            [[1.1, 1.2]],
            domestic_rate=0.03,
            foreign_rate=0.01,
        )


@pytest.mark.parametrize(
    "spots",
    [
        [1.1, 1.2],
        [[1.1]],
        [],
        [[0.0, 1.2]],
        [[-1.0, 1.2]],
        [[float("nan"), 1.2]],
        [[float("inf"), 1.2]],
    ],
)
def test_invalid_spot_paths(spots: list) -> None:
    with pytest.raises(ValueError):
        value_fx_forward_paths(
            FXForwardExposureTrade(100.0, 1.1, 1.0),
            [0.0, 0.5],
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
def test_invalid_rates(rd: float, rf: float) -> None:
    with pytest.raises(ValueError):
        value_fx_forward_paths(
            FXForwardExposureTrade(100.0, 1.1, 1.0),
            [0.0, 0.5],
            [[1.1, 1.2]],
            domestic_rate=rd,
            foreign_rate=rf,
        )


def test_wrong_trade_type() -> None:
    with pytest.raises(TypeError):
        value_fx_forward_paths(
            object(),
            [0.0, 0.5],
            [[1.1, 1.2]],
            domestic_rate=0.03,
            foreign_rate=0.01,
        )


def test_input_spots_not_mutated() -> None:
    trade = FXForwardExposureTrade(100.0, 1.1, 1.0)
    spots = np.array([[1.1, 1.2, 1.3]])
    original = spots.copy()

    value_fx_forward_paths(
        trade,
        [0.0, 0.5, 1.0],
        spots,
        domestic_rate=0.03,
        foreign_rate=0.01,
    )

    np.testing.assert_array_equal(spots, original)
