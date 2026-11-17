"""Portfolio aggregation of pathwise FX-forward valuations."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.pricing.fx_forward_exposure import (
    FXForwardExposureTrade,
    value_fx_forward_paths,
)


@dataclass(frozen=True)
class FXForwardPortfolioValues:
    """Individual and aggregated domestic-currency trade values.

    trade_values has shape (n_trades, n_paths, n_times).
    portfolio_values has shape (n_paths, n_times).
    """

    trade_values: NDArray[np.float64]
    portfolio_values: NDArray[np.float64]


def value_fx_forward_portfolio(
    trades: Sequence[FXForwardExposureTrade],
    times: ArrayLike,
    fx_spots: ArrayLike,
    *,
    domestic_rate: float,
    foreign_rate: float,
) -> FXForwardPortfolioValues:
    """Value FX forwards and sum signed pathwise values.

    All trades must reference the same domestic/foreign FX pair.
    Currency identifiers are not represented in the trade type,
    so callers must enforce that convention.

    This function performs signed portfolio aggregation only.
    It does not implement legal netting sets or collateral.
    """
    if not isinstance(trades, (list, tuple)):
        raise TypeError("trades must be a list or tuple")

    for trade in trades:
        if not isinstance(trade, FXForwardExposureTrade):
            raise TypeError("all trades must be FXForwardExposureTrade instances")

    # Validate scenario inputs and rate assumptions using the
    # existing valuation implementation, even for empty portfolios.
    reference = FXForwardExposureTrade(
        foreign_notional=1.0,
        strike=1.0,
        maturity=0.0,
    )

    reference_values = value_fx_forward_paths(
        reference,
        times,
        fx_spots,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
    )

    n_paths, n_times = reference_values.shape

    if not trades:
        return FXForwardPortfolioValues(
            trade_values=np.empty(
                (0, n_paths, n_times),
                dtype=np.float64,
            ),
            portfolio_values=np.zeros(
                (n_paths, n_times),
                dtype=np.float64,
            ),
        )

    trade_values = np.stack(
        [
            value_fx_forward_paths(
                trade,
                times,
                fx_spots,
                domestic_rate=domestic_rate,
                foreign_rate=foreign_rate,
            )
            for trade in trades
        ],
        axis=0,
    )

    portfolio_values = np.sum(trade_values, axis=0)

    return FXForwardPortfolioValues(
        trade_values=np.asarray(
            trade_values,
            dtype=np.float64,
        ),
        portfolio_values=np.asarray(
            portfolio_values,
            dtype=np.float64,
        ),
    )
