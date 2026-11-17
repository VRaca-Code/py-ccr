"""Independent unilateral CVA/DVA aggregation without first-to-default."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.cva import CVAResult, calculate_unilateral_cva
from pyccr.xva.dva import DVAResult, calculate_unilateral_dva


@dataclass(frozen=True)
class BilateralXVAResult:
    """Standalone CVA/DVA components and signed net valuation adjustment.

    A negative net_xva reduces the risk-free portfolio value.
    This is not a bilateral first-to-default valuation.
    """

    cva: float
    dva: float
    net_xva: float
    times: NDArray[np.float64]
    cva_contributions: NDArray[np.float64]
    dva_contributions: NDArray[np.float64]
    net_contributions: NDArray[np.float64]
    cva_result: CVAResult
    dva_result: DVAResult


def calculate_bilateral_xva(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    counterparty_credit_curve: ConstantHazardCurve,
    own_credit_curve: ConstantHazardCurve,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
) -> BilateralXVAResult:
    """Aggregate standalone unilateral CVA and DVA.

    EE and ENE must be nonnegative magnitudes, such as the ``ee``
    and ``ene`` arrays from ``calculate_exposure_profiles``.

    Counterparty and own default probabilities are each unconditional.
    Neither adjustment is conditioned on survival of the other party.

    Consequently this function does not implement first-to-default,
    joint default dependence, collateral, or wrong-way risk.
    """
    cva_result = calculate_unilateral_cva(
        times=times,
        expected_exposure=expected_exposure,
        discount_factors=discount_factors,
        credit_curve=counterparty_credit_curve,
        recovery_rate=counterparty_recovery_rate,
    )

    dva_result = calculate_unilateral_dva(
        times=times,
        expected_negative_exposure=expected_negative_exposure,
        discount_factors=discount_factors,
        own_credit_curve=own_credit_curve,
        recovery_rate=own_recovery_rate,
    )

    cva_contributions = cva_result.contributions.copy()
    dva_contributions = dva_result.contributions.copy()
    net_contributions = dva_contributions - cva_contributions

    return BilateralXVAResult(
        cva=cva_result.cva,
        dva=dva_result.dva,
        net_xva=float(dva_result.dva - cva_result.cva),
        times=cva_result.times.copy(),
        cva_contributions=cva_contributions,
        dva_contributions=dva_contributions,
        net_contributions=net_contributions,
        cva_result=cva_result,
        dva_result=dva_result,
    )
