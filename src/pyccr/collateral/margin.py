"""Pathwise bilateral variation-margin simulation.

Portfolio values have shape (n_paths, n_times).

Positive VM means collateral held by the reporting institution.
Negative VM means collateral posted by the reporting institution.

Margin calls are based on current portfolio values and outstanding
scheduled transfers. Settlement occurs on the simulation grid.

This module does not model default, MPOR, initial margin, collateral
interest, collateral haircuts, or legal closeout enforceability.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.collateral.csa import CSAConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MarginResult:
    """Pathwise VM balances and residual exposure."""

    vm_balance: FloatArray
    margin_calls: FloatArray
    settled_transfers: FloatArray
    pending_transfers: FloatArray
    positive_exposure: FloatArray
    negative_exposure: FloatArray


def _validate_values(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError("portfolio_values must have shape (n_paths, n_times)")

    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("portfolio_values must have nonempty path and time axes")

    if not np.all(np.isfinite(array)):
        raise ValueError("portfolio_values must contain finite values")

    return array


def variation_margin_target(
    portfolio_values: FloatArray,
    csa: CSAConfig,
) -> FloatArray:
    """Compute signed VM target from bilateral CSA thresholds."""

    values = np.asarray(portfolio_values, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio_values must be finite")

    return np.maximum(values - csa.counterparty_threshold, 0.0) - np.maximum(
        -values - csa.own_threshold, 0.0
    )


def simulate_variation_margin(
    portfolio_values: FloatArray,
    csa: CSAConfig,
) -> MarginResult:
    """Simulate bilateral VM calls and settlements path by path.

    Conventions
    -----------
    - Grid index zero is a scheduled call date.
    - Due settlements are applied before new calls at a grid point.
    - Calls account for unsettled transfers already scheduled.
    - Calls below MTA are not scheduled.
    - With zero settlement lag, a new call settles immediately.
    - Pending transfers beyond the final grid point remain pending
      and are not counted as effective collateral.
    """

    values = _validate_values(portfolio_values)

    n_paths, n_times = values.shape

    balance = np.empty_like(values)
    calls = np.zeros_like(values)
    settled = np.zeros_like(values)
    pending = np.zeros_like(values)

    # scheduled[:, t] contains transfers settling at grid index t.
    scheduled = np.zeros_like(values)

    current = np.full(
        n_paths,
        float(csa.initial_vm_balance),
        dtype=np.float64,
    )

    # Outstanding transfers settling after the current grid index.
    outstanding = np.zeros(n_paths, dtype=np.float64)

    for t in range(n_times):
        due = scheduled[:, t].copy()

        current += due
        outstanding -= due
        settled[:, t] += due

        if t % csa.call_frequency_steps == 0:
            target = variation_margin_target(values[:, t], csa)

            required = target - (current + outstanding)

            should_call = (np.abs(required) >= csa.minimum_transfer_amount) & (
                required != 0.0
            )

            transfer = np.where(should_call, required, 0.0)

            calls[:, t] = transfer

            settlement_time = t + csa.settlement_lag_steps

            if csa.settlement_lag_steps == 0:
                current += transfer
                settled[:, t] += transfer

            else:
                outstanding += transfer

                if settlement_time < n_times:
                    scheduled[:, settlement_time] += transfer

        balance[:, t] = current
        pending[:, t] = outstanding

    residual = values - balance

    return MarginResult(
        vm_balance=balance,
        margin_calls=calls,
        settled_transfers=settled,
        pending_transfers=pending,
        positive_exposure=np.maximum(residual, 0.0),
        negative_exposure=np.maximum(-residual, 0.0),
    )
