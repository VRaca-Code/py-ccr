"""Collateral-aware CVA, DVA and bilateral XVA composition.

The supplied collateral exposures are indexed by hypothetical default
time, even when portfolio closeout values occur later due to MPOR.

Discount factors must be supplied on the same default-time grid.
No interpolation, implicit truncation, or maturity extrapolation is
performed.

This adapter reuses existing independent-default unilateral and
bilateral calculators, plus the independent first-to-default model.
It does not add funding costs, wrong-way risk, or joint default
dependence.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.collateral_profiles import CollateralExposureResult
from pyccr.xva.bilateral import BilateralXVAResult, calculate_bilateral_xva
from pyccr.xva.cva import CVAResult, calculate_unilateral_cva
from pyccr.xva.dva import DVAResult, calculate_unilateral_dva
from pyccr.xva.first_to_default import (
    FirstToDefaultXVAResult,
    calculate_first_to_default_xva,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CollateralXVAResult:
    """Existing XVA results evaluated on the collateral default-time grid."""

    times: FloatArray
    discount_factors: FloatArray
    cva_result: CVAResult
    dva_result: DVAResult
    bilateral_result: BilateralXVAResult
    first_to_default_result: FirstToDefaultXVAResult


def calculate_collateral_xva(
    collateral_exposure: CollateralExposureResult,
    discount_factors: ArrayLike,
    counterparty_credit_curve: ConstantHazardCurve,
    own_credit_curve: ConstantHazardCurve,
    *,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
) -> CollateralXVAResult:
    """Calculate collateral-aware CVA, DVA and bilateral XVA.

    ``discount_factors`` must correspond exactly to
    ``collateral_exposure.profiles.times``, which contains hypothetical
    default dates rather than MPOR closeout dates.

    For positive MPOR, this grid may end before the original
    simulation horizon. The caller must explicitly accept that
    shorter XVA horizon or extend the original simulation horizon.

    At least two default-time observations are required because the
    underlying XVA calculators integrate over time intervals.
    """
    if not isinstance(collateral_exposure, CollateralExposureResult):
        raise TypeError("collateral_exposure must be a CollateralExposureResult")

    profiles = collateral_exposure.profiles
    times = np.asarray(profiles.times, dtype=np.float64)

    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            "collateral XVA requires at least two default-time observations"
        )

    if not np.all(np.isfinite(times)):
        raise ValueError("default times must be finite")

    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("default times must start at zero and strictly increase")

    closeout_times = np.asarray(
        collateral_exposure.closeout_times,
        dtype=np.float64,
    )

    if closeout_times.shape != times.shape:
        raise ValueError("closeout times must match default-time grid")

    if not np.all(np.isfinite(closeout_times)):
        raise ValueError("closeout times must be finite")

    if np.any(closeout_times < times):
        raise ValueError("closeout times cannot precede default times")

    df = np.asarray(discount_factors, dtype=np.float64)

    if df.shape != times.shape:
        raise ValueError("discount_factors must match the collateral default-time grid")

    if not np.all(np.isfinite(df)) or np.any(df <= 0.0):
        raise ValueError("discount_factors must be finite and strictly positive")

    ee = np.asarray(profiles.ee, dtype=np.float64)
    ene = np.asarray(profiles.ene, dtype=np.float64)

    if ee.shape != times.shape or ene.shape != times.shape:
        raise ValueError("EE and ENE must match the default-time grid")

    if (
        not np.all(np.isfinite(ee))
        or not np.all(np.isfinite(ene))
        or np.any(ee < 0.0)
        or np.any(ene < 0.0)
    ):
        raise ValueError("EE and ENE must be finite and nonnegative")

    cva_result = calculate_unilateral_cva(
        times,
        ee,
        df,
        counterparty_credit_curve,
        recovery_rate=counterparty_recovery_rate,
    )

    dva_result = calculate_unilateral_dva(
        times,
        ene,
        df,
        own_credit_curve,
        recovery_rate=own_recovery_rate,
    )

    bilateral_result = calculate_bilateral_xva(
        times,
        ee,
        ene,
        df,
        counterparty_credit_curve,
        own_credit_curve,
        counterparty_recovery_rate=counterparty_recovery_rate,
        own_recovery_rate=own_recovery_rate,
    )

    first_to_default_result = calculate_first_to_default_xva(
        times,
        ee,
        ene,
        df,
        counterparty_credit_curve,
        own_credit_curve,
        counterparty_recovery_rate=counterparty_recovery_rate,
        own_recovery_rate=own_recovery_rate,
    )

    return CollateralXVAResult(
        times=times.copy(),
        discount_factors=df.copy(),
        cva_result=cva_result,
        dva_result=dva_result,
        bilateral_result=bilateral_result,
        first_to_default_result=first_to_default_result,
    )
