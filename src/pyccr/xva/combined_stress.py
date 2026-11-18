"""Deterministic credit-hazard and funding-spread stress reporting.

This module reuses standalone CVA/DVA/FVA calculations. Exposures,
discount factors, and recovery rates are held fixed across scenarios.
No first-to-default coupling, stochastic funding, collateral,
wrong-way risk, or joint exposure/default revaluation is included.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.combined import CombinedXVAResult, calculate_combined_xva


@dataclass(frozen=True)
class CombinedXVAStressResult:
    """Base/stressed adjustments and stressed-minus-base differences."""

    base: CombinedXVAResult
    stressed: CombinedXVAResult
    delta_cva: float
    delta_dva: float
    delta_fca: float
    delta_fba: float
    delta_fva: float
    delta_net_xva: float
    delta_net_contributions: NDArray[np.float64]


def _finite_scalar(value: float, name: str) -> float:
    """Require a finite real scalar, excluding booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if arr.ndim != 0 or not np.isfinite(arr):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(arr)


def _shock_hazard(
    curve: ConstantHazardCurve,
    shock_bps: float,
    name: str,
) -> ConstantHazardCurve:
    if not isinstance(curve, ConstantHazardCurve):
        raise TypeError(f"{name} curve must be a ConstantHazardCurve")
    shock = _finite_scalar(shock_bps, name)
    stressed_rate = curve.hazard_rate + shock / 10_000.0
    if not np.isfinite(stressed_rate) or stressed_rate < 0.0:
        raise ValueError(f"{name} produces an invalid stressed hazard rate")
    return ConstantHazardCurve(stressed_rate)


def _shock_spread(
    spread: ArrayLike,
    shock_bps: ArrayLike,
    name: str,
    n_intervals: int,
) -> NDArray[np.float64]:
    """Apply scalar or per-interval funding shocks."""
    try:
        base = np.asarray(spread, dtype=np.float64)
        shock = np.asarray(shock_bps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc

    if base.ndim == 0:
        base = np.full(n_intervals, float(base))
    elif base.shape != (n_intervals,):
        raise ValueError(f"{name} base spread has invalid shape")

    if shock.ndim == 0:
        shock = np.full(n_intervals, float(shock))
    elif shock.shape != (n_intervals,):
        raise ValueError(f"{name} shock must have length len(times) - 1")

    if not np.all(np.isfinite(base)) or np.any(base < 0.0):
        raise ValueError(f"{name} base spread must be non-negative and finite")
    if not np.all(np.isfinite(shock)):
        raise ValueError(f"{name} shock must be finite")

    stressed = base + shock / 10_000.0
    if not np.all(np.isfinite(stressed)) or np.any(stressed < 0.0):
        raise ValueError(f"{name} stressed spread must be non-negative")
    return stressed


def compare_combined_xva_stress(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    counterparty_credit_curve: ConstantHazardCurve,
    own_credit_curve: ConstantHazardCurve,
    *,
    borrowing_spread: ArrayLike,
    lending_spread: ArrayLike = 0.0,
    counterparty_hazard_shock_bps: float = 0.0,
    own_hazard_shock_bps: float = 0.0,
    borrowing_shock_bps: ArrayLike = 0.0,
    lending_shock_bps: ArrayLike = 0.0,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
) -> CombinedXVAStressResult:
    """Compare additive XVA under deterministic credit/funding shocks.

    Positive counterparty hazard shocks typically increase CVA;
    positive own hazard shocks typically increase DVA.
    Positive borrowing spread shocks increase FCA; positive
    lending spread shocks increase FBA.

    Negative shocks are allowed only when resulting hazards and
    funding spreads remain nonnegative. All shocks are absolute
    changes in basis points, not relative percentage changes.
    """
    base = calculate_combined_xva(
        times,
        expected_exposure,
        expected_negative_exposure,
        discount_factors,
        counterparty_credit_curve,
        own_credit_curve,
        borrowing_spread=borrowing_spread,
        lending_spread=lending_spread,
        counterparty_recovery_rate=counterparty_recovery_rate,
        own_recovery_rate=own_recovery_rate,
    )

    stressed_cp = _shock_hazard(
        counterparty_credit_curve,
        counterparty_hazard_shock_bps,
        "counterparty_hazard_shock_bps",
    )
    stressed_own = _shock_hazard(
        own_credit_curve,
        own_hazard_shock_bps,
        "own_hazard_shock_bps",
    )

    n_intervals = base.times.size - 1
    stressed_borrow = _shock_spread(
        borrowing_spread,
        borrowing_shock_bps,
        "borrowing",
        n_intervals,
    )
    stressed_lend = _shock_spread(
        lending_spread,
        lending_shock_bps,
        "lending",
        n_intervals,
    )

    stressed = calculate_combined_xva(
        base.times,
        expected_exposure,
        expected_negative_exposure,
        discount_factors,
        stressed_cp,
        stressed_own,
        borrowing_spread=stressed_borrow,
        lending_spread=stressed_lend,
        counterparty_recovery_rate=counterparty_recovery_rate,
        own_recovery_rate=own_recovery_rate,
    )

    return CombinedXVAStressResult(
        base=base,
        stressed=stressed,
        delta_cva=float(stressed.cva - base.cva),
        delta_dva=float(stressed.dva - base.dva),
        delta_fca=float(stressed.fca - base.fca),
        delta_fba=float(stressed.fba - base.fba),
        delta_fva=float(stressed.fva - base.fva),
        delta_net_xva=float(stressed.net_xva - base.net_xva),
        delta_net_contributions=(stressed.net_contributions - base.net_contributions),
    )
