"""Deterministic-spread FVA using contemporaneous settled variation margin.

This is a VM-only funding approximation, not a jointly coupled XVA model.

Funding requirements use portfolio value minus effective settled VM on
the full valuation grid. Pending transfers, MPOR closeout exposures,
segregated initial margin, collateral remuneration, stochastic funding
spreads, and default-survival weighting are outside this adapter.

All monetary amounts must use consistent currency units. The caller
must ensure that portfolio paths and margin paths have identical
ordering and correspond to the same valuation-time grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyccr.collateral.margin import MarginResult
from pyccr.xva.fva import FVAResult, calculate_fva


def calculate_collateral_fva(
    times: ArrayLike,
    portfolio_values: ArrayLike,
    margin_result: MarginResult,
    discount_factors: ArrayLike,
    *,
    borrowing_spread: ArrayLike,
    lending_spread: ArrayLike = 0.0,
) -> FVAResult:
    """Calculate VM-adjusted funding cost and benefit.

    The pathwise funding requirement is

        R[p, i] = V[p, i] - VM[p, i],

    where VM is effective settled variation margin, positive when
    collateral is held and negative when collateral is posted.

    Funding EE is mean(max(R, 0)) and funding ENE is
    mean(max(-R, 0)), averaged across simulation paths.

    The existing calculate_fva engine performs endpoint discounting,
    spread validation, interval integration, and sign accounting.

    A MarginResult does not carry its source time grid or path IDs.
    Shape equality cannot establish economic alignment; callers must
    supply consistently ordered paths and matching valuation dates.
    """
    if not isinstance(margin_result, MarginResult):
        raise TypeError("margin_result must be a MarginResult")

    try:
        values = np.asarray(portfolio_values, dtype=np.float64)
        vm = np.asarray(margin_result.vm_balance, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "portfolio_values, vm_balance and times must be numeric"
        ) from exc

    if values.ndim != 2 or min(values.shape, default=0) < 1:
        raise ValueError(
            "portfolio_values must have shape (n_paths, n_times) with nonempty axes"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio_values must contain finite values")

    if vm.shape != values.shape:
        raise ValueError("margin_result.vm_balance must match portfolio_values shape")

    if not np.all(np.isfinite(vm)):
        raise ValueError("margin_result.vm_balance must be finite")

    if t.ndim != 1 or t.shape != (values.shape[1],):
        raise ValueError("times must match the portfolio valuation-time axis")

    residual = values - vm

    if not np.all(np.isfinite(residual)):
        raise ValueError("portfolio values minus VM must be finite")

    funding_ee = np.mean(np.maximum(residual, 0.0), axis=0)
    funding_ene = np.mean(np.maximum(-residual, 0.0), axis=0)

    if not np.all(np.isfinite(funding_ee)) or not np.all(np.isfinite(funding_ene)):
        raise ValueError("aggregated funding exposures must be finite")

    return calculate_fva(
        t,
        funding_ee,
        funding_ene,
        discount_factors,
        borrowing_spread=borrowing_spread,
        lending_spread=lending_spread,
    )
