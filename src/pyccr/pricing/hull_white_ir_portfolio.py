"""Signed pathwise aggregation of Hull-White interest-rate trades."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_swap_exposure import (
    value_hull_white_swap_paths,
)
from pyccr.pricing.hull_white_swaption_exposure import (
    HullWhiteSwaptionTrade,
    value_hull_white_swaption_paths,
)
from pyccr.pricing.interest_rate_swap import InterestRateSwap

IRTrade = InterestRateSwap | HullWhiteSwaptionTrade


@dataclass(frozen=True)
class HullWhiteIRPortfolioValues:
    """Individual and aggregated signed interest-rate trade MTMs.

    trade_values has shape (n_trades, n_paths, n_times).
    portfolio_values has shape (n_paths, n_times).
    """

    trade_values: NDArray[np.float64]
    portfolio_values: NDArray[np.float64]


def value_hull_white_ir_portfolio(
    trades: Sequence[IRTrade],
    model: HullWhiteCurveModel,
    times: ArrayLike,
    short_rates: ArrayLike,
) -> HullWhiteIRPortfolioValues:
    """Value IRS and physically settled European swaptions jointly.

    Every trade uses the same Hull-White model and the same
    simulated short-rate paths. Values are summed with their
    original signs before exposure is calculated.

    This is single-curve portfolio aggregation only. The caller
    must ensure that all trades share the same currency and
    compatible floating-rate conventions. In particular, the
    swaption trade type does not carry a currency identifier.

    This function does not establish legal netting eligibility,
    apply collateral, or calculate XVA.
    """
    if not isinstance(trades, (list, tuple)):
        raise TypeError("trades must be a list or tuple")

    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")

    for trade in trades:
        if not isinstance(
            trade,
            (InterestRateSwap, HullWhiteSwaptionTrade),
        ):
            raise TypeError(
                "all trades must be InterestRateSwap or "
                "HullWhiteSwaptionTrade instances"
            )

    grid = np.asarray(times, dtype=np.float64)
    rates = np.asarray(short_rates, dtype=np.float64)

    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or grid[0] != 0.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("times must start at zero and increase strictly")

    if (
        rates.ndim != 2
        or rates.shape[0] < 1
        or rates.shape[1] != grid.size
        or not np.all(np.isfinite(rates))
    ):
        raise ValueError("short_rates must be a finite (n_paths, n_times) matrix")

    if grid[-1] > model.curve.tenors[-1]:
        raise ValueError("simulation horizon exceeds model curve horizon")

    n_paths, n_times = rates.shape

    if not trades:
        return HullWhiteIRPortfolioValues(
            trade_values=np.empty(
                (0, n_paths, n_times),
                dtype=np.float64,
            ),
            portfolio_values=np.zeros(
                (n_paths, n_times),
                dtype=np.float64,
            ),
        )

    individual_values = []

    for trade in trades:
        if isinstance(trade, InterestRateSwap):
            values = value_hull_white_swap_paths(
                trade,
                model,
                grid,
                rates,
            )
        else:
            values = value_hull_white_swaption_paths(
                trade,
                model,
                grid,
                rates,
            )

        individual_values.append(values)

    trade_values = np.stack(individual_values, axis=0)

    portfolio_values = np.sum(
        trade_values,
        axis=0,
    )

    return HullWhiteIRPortfolioValues(
        trade_values=np.asarray(
            trade_values,
            dtype=np.float64,
        ),
        portfolio_values=np.asarray(
            portfolio_values,
            dtype=np.float64,
        ),
    )
