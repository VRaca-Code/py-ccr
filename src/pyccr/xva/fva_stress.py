"""Deterministic funding-spread stress comparisons.

The same EE, ENE, discount factors and time grid are used in both
scenarios. Only funding spreads change. This does not model
stochastic funding, collateral, default survival, or joint XVA.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.xva.fva import FVAResult, calculate_fva


@dataclass(frozen=True)
class FVAStressComparison:
    """Base/stressed FVA results and stressed-minus-base changes."""

    base: FVAResult
    stressed: FVAResult
    borrowing_shock_bps: NDArray[np.float64]
    lending_shock_bps: NDArray[np.float64]
    delta_fca: float
    delta_fba: float
    delta_fva: float
    delta_contributions: NDArray[np.float64]


def _shock_vector(
    shock: ArrayLike,
    name: str,
    n_intervals: int,
) -> NDArray[np.float64]:
    """Validate scalar or interval-specific basis-point shocks."""
    try:
        values = np.asarray(shock, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc

    if values.ndim == 0:
        values = np.full(n_intervals, float(values), dtype=np.float64)
    elif values.ndim != 1 or values.shape != (n_intervals,):
        raise ValueError(f"{name} must be scalar or have length len(times) - 1")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")

    return values.copy()


def compare_fva_spread_stress(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    *,
    borrowing_spread: ArrayLike,
    lending_spread: ArrayLike = 0.0,
    borrowing_shock_bps: ArrayLike = 0.0,
    lending_shock_bps: ArrayLike = 0.0,
) -> FVAStressComparison:
    """Compare FVA after absolute funding-spread shocks in basis points.

    A positive borrowing shock increases funding cost (FCA);
    a positive lending shock increases funding benefit (FBA).

    Negative shocks are permitted only when the resulting spread
    remains nonnegative. One basis point equals 0.0001.

    Both scenarios use identical EE, ENE, time grids and discount
    factors. Changes are therefore spread-only sensitivities,
    not joint market, exposure, credit or funding revaluations.
    """
    base = calculate_fva(
        times,
        expected_exposure,
        expected_negative_exposure,
        discount_factors,
        borrowing_spread,
        lending_spread,
    )

    n_intervals = base.times.size - 1
    borrow_shock = _shock_vector(
        borrowing_shock_bps,
        "borrowing_shock_bps",
        n_intervals,
    )
    lend_shock = _shock_vector(
        lending_shock_bps,
        "lending_shock_bps",
        n_intervals,
    )

    stressed_borrow = base.borrowing_spreads + borrow_shock / 10_000.0
    stressed_lend = base.lending_spreads + lend_shock / 10_000.0

    if not np.all(np.isfinite(stressed_borrow)) or np.any(stressed_borrow < 0.0):
        raise ValueError("stressed borrowing spread must be non-negative")

    if not np.all(np.isfinite(stressed_lend)) or np.any(stressed_lend < 0.0):
        raise ValueError("stressed lending spread must be non-negative")

    stressed = calculate_fva(
        base.times,
        expected_exposure,
        expected_negative_exposure,
        discount_factors,
        stressed_borrow,
        stressed_lend,
    )

    return FVAStressComparison(
        base=base,
        stressed=stressed,
        borrowing_shock_bps=borrow_shock,
        lending_shock_bps=lend_shock,
        delta_fca=float(stressed.fca - base.fca),
        delta_fba=float(stressed.fba - base.fba),
        delta_fva=float(stressed.fva - base.fva),
        delta_contributions=(stressed.net_contributions - base.net_contributions),
    )
