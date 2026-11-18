"""Additive reporting of standalone CVA, DVA, and deterministic FVA.

This reporting layer is not a jointly calibrated or coupled XVA
valuation. CVA/DVA are standalone unilateral adjustments; FVA
has no default-survival weighting. Double-counting and funding
policy considerations are outside this simplified approximation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.bilateral import BilateralXVAResult, calculate_bilateral_xva
from pyccr.xva.fva import FVAResult, calculate_fva


@dataclass(frozen=True)
class CombinedXVAResult:
    """Component results and additive signed valuation adjustment.

    Positive DVA/FBA are benefits; positive CVA/FCA are costs.
    Signed net_xva equals DVA - CVA + FBA - FCA.
    """

    cva: float
    dva: float
    fca: float
    fba: float
    fva: float
    net_xva: float
    times: NDArray[np.float64]
    cva_contributions: NDArray[np.float64]
    dva_contributions: NDArray[np.float64]
    fca_contributions: NDArray[np.float64]
    fba_contributions: NDArray[np.float64]
    net_contributions: NDArray[np.float64]
    bilateral_result: BilateralXVAResult
    fva_result: FVAResult


def calculate_combined_xva(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    counterparty_credit_curve: ConstantHazardCurve,
    own_credit_curve: ConstantHazardCurve,
    *,
    borrowing_spread: ArrayLike,
    lending_spread: ArrayLike = 0.0,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
) -> CombinedXVAResult:
    """Aggregate standalone CVA/DVA and deterministic FVA.

    EE and ENE must be nonnegative exposure magnitudes.
    All components share the same grid and deterministic discount
    factors. Each interval uses its endpoint exposure and discount
    factor, consistent with the existing CVA/DVA and FVA modules.

    This does not incorporate first-to-default, joint funding/default
    dependence, collateral, wrong-way risk, or closeout conventions.
    """
    bilateral = calculate_bilateral_xva(
        times,
        expected_exposure,
        expected_negative_exposure,
        discount_factors,
        counterparty_credit_curve,
        own_credit_curve,
        counterparty_recovery_rate,
        own_recovery_rate,
    )

    funding = calculate_fva(
        times,
        expected_exposure,
        expected_negative_exposure,
        discount_factors,
        borrowing_spread,
        lending_spread,
    )

    net_contributions = bilateral.net_contributions + funding.net_contributions

    return CombinedXVAResult(
        cva=bilateral.cva,
        dva=bilateral.dva,
        fca=funding.fca,
        fba=funding.fba,
        fva=funding.fva,
        net_xva=float(bilateral.net_xva + funding.fva),
        times=bilateral.times.copy(),
        cva_contributions=bilateral.cva_contributions.copy(),
        dva_contributions=bilateral.dva_contributions.copy(),
        fca_contributions=funding.fca_contributions.copy(),
        fba_contributions=funding.fba_contributions.copy(),
        net_contributions=np.asarray(net_contributions, dtype=np.float64),
        bilateral_result=bilateral,
        fva_result=funding,
    )
