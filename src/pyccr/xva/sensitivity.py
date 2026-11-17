"""Bilateral XVA sensitivity to counterparty and own CDS spreads."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.cds_calibration import calibrate_flat_hazard_from_cds
from pyccr.xva.bilateral import calculate_bilateral_xva
from pyccr.xva.first_to_default import calculate_first_to_default_xva


@dataclass(frozen=True)
class BilateralXVASensitivityResult:
    """Credit-spread grid of standalone and first-to-default XVA.

    All valuation matrices have shape
    ``(len(own_spreads_bps), len(counterparty_spreads_bps))``.

    Net XVA is DVA minus CVA. Corrections are first-to-default
    values minus standalone bilateral values.
    """

    counterparty_spreads_bps: NDArray[np.float64]
    own_spreads_bps: NDArray[np.float64]
    naive_cva: NDArray[np.float64]
    naive_dva: NDArray[np.float64]
    naive_net_xva: NDArray[np.float64]
    first_to_default_cva: NDArray[np.float64]
    first_to_default_dva: NDArray[np.float64]
    first_to_default_net_xva: NDArray[np.float64]
    cva_correction: NDArray[np.float64]
    dva_correction: NDArray[np.float64]
    net_xva_correction: NDArray[np.float64]


def _validate_spreads(
    spreads: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    """Validate a nonempty one-dimensional CDS spread grid."""
    values = np.asarray(spreads, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")

    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative spreads")

    return values.copy()


def calculate_bilateral_xva_sensitivity(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    counterparty_spreads_bps: ArrayLike,
    own_spreads_bps: ArrayLike,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
) -> BilateralXVASensitivityResult:
    """Evaluate bilateral XVA across counterparty and own CDS spreads.

    CDS spreads are converted to independent constant hazard rates
    using the credit-triangle approximation.

    Rows correspond to own-credit spreads; columns correspond to
    counterparty spreads. Exposure and discount inputs are held
    fixed across scenarios.

    EE and ENE must be nonnegative magnitudes. This model does not
    include correlated defaults, wrong-way risk, collateral, or
    legally enforceable netting.
    """
    cp_spreads = _validate_spreads(
        counterparty_spreads_bps,
        "counterparty_spreads_bps",
    )
    own_spreads = _validate_spreads(
        own_spreads_bps,
        "own_spreads_bps",
    )

    cp_curves = [
        calibrate_flat_hazard_from_cds(
            float(spread),
            recovery_rate=counterparty_recovery_rate,
        )
        for spread in cp_spreads
    ]
    own_curves = [
        calibrate_flat_hazard_from_cds(
            float(spread),
            recovery_rate=own_recovery_rate,
        )
        for spread in own_spreads
    ]

    # Copy the shared inputs once. Existing XVA calculators perform
    # their own shape, finiteness, and exposure validation.
    t = np.asarray(times, dtype=np.float64).copy()
    ee = np.asarray(expected_exposure, dtype=np.float64).copy()
    ene = np.asarray(expected_negative_exposure, dtype=np.float64).copy()
    df = np.asarray(discount_factors, dtype=np.float64).copy()

    shape = (own_spreads.size, cp_spreads.size)

    naive_cva = np.empty(shape, dtype=np.float64)
    naive_dva = np.empty(shape, dtype=np.float64)
    naive_net = np.empty(shape, dtype=np.float64)

    ftd_cva = np.empty(shape, dtype=np.float64)
    ftd_dva = np.empty(shape, dtype=np.float64)
    ftd_net = np.empty(shape, dtype=np.float64)

    for row, own_curve in enumerate(own_curves):
        for col, cp_curve in enumerate(cp_curves):
            naive = calculate_bilateral_xva(
                times=t,
                expected_exposure=ee,
                expected_negative_exposure=ene,
                discount_factors=df,
                counterparty_credit_curve=cp_curve,
                own_credit_curve=own_curve,
                counterparty_recovery_rate=counterparty_recovery_rate,
                own_recovery_rate=own_recovery_rate,
            )

            ftd = calculate_first_to_default_xva(
                times=t,
                expected_exposure=ee,
                expected_negative_exposure=ene,
                discount_factors=df,
                counterparty_credit_curve=cp_curve,
                own_credit_curve=own_curve,
                counterparty_recovery_rate=counterparty_recovery_rate,
                own_recovery_rate=own_recovery_rate,
            )

            naive_cva[row, col] = naive.cva
            naive_dva[row, col] = naive.dva
            naive_net[row, col] = naive.net_xva

            ftd_cva[row, col] = ftd.cva
            ftd_dva[row, col] = ftd.dva
            ftd_net[row, col] = ftd.net_xva

    return BilateralXVASensitivityResult(
        counterparty_spreads_bps=cp_spreads,
        own_spreads_bps=own_spreads,
        naive_cva=naive_cva,
        naive_dva=naive_dva,
        naive_net_xva=naive_net,
        first_to_default_cva=ftd_cva,
        first_to_default_dva=ftd_dva,
        first_to_default_net_xva=ftd_net,
        cva_correction=ftd_cva - naive_cva,
        dva_correction=ftd_dva - naive_dva,
        net_xva_correction=ftd_net - naive_net,
    )
