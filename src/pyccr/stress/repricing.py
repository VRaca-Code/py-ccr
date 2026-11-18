"""Compare FX-forward exposure and standalone bilateral XVA under stress.

Base and stressed FX paths are supplied by the caller. This module does
not generate scenarios, enforce common random numbers, or model legal
netting, collateral, wrong-way risk, or first-to-default.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.profiles import ExposureProfiles, calculate_exposure_profiles
from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
from pyccr.pricing.fx_forward_portfolio import (
    FXForwardPortfolioValues,
    value_fx_forward_portfolio,
)
from pyccr.xva.bilateral import BilateralXVAResult, calculate_bilateral_xva


@dataclass(frozen=True)
class FXStressValuation:
    """One scenario's signed valuations, exposure profile and XVA."""

    portfolio: FXForwardPortfolioValues
    exposures: ExposureProfiles
    xva: BilateralXVAResult
    discount_factors: NDArray[np.float64]


@dataclass(frozen=True)
class FXStressComparison:
    """Stressed-minus-base exposure and XVA changes."""

    times: NDArray[np.float64]
    base: FXStressValuation
    stressed: FXStressValuation
    delta_ee: NDArray[np.float64]
    delta_ene: NDArray[np.float64]
    delta_pfe: NDArray[np.float64]
    delta_epe: float
    delta_cva: float
    delta_dva: float
    delta_net_xva: float


def compare_fx_forward_stress(
    trades: Sequence[FXForwardExposureTrade],
    times: ArrayLike,
    base_fx_spots: ArrayLike,
    stressed_fx_spots: ArrayLike,
    *,
    base_domestic_rate: float,
    base_foreign_rate: float,
    stressed_domestic_rate: float,
    stressed_foreign_rate: float,
    base_counterparty_curve: ConstantHazardCurve,
    base_own_curve: ConstantHazardCurve,
    stressed_counterparty_curve: ConstantHazardCurve,
    stressed_own_curve: ConstantHazardCurve,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
    pfe_confidence_level: float = 0.95,
) -> FXStressComparison:
    """Reprice the same FX-forward portfolio in two supplied scenarios.

    Both scenarios must have the same time grid and path count.
    Stressed paths are not assumed paired with base paths; callers
    must establish common random numbers if paired comparisons
    are required.

    Each scenario uses its own deterministic domestic discount
    factors and independent standalone CVA/DVA credit curves.
    The valuation convention excludes settled cashflows.
    """
    t = np.asarray(times, dtype=np.float64)

    base_spots = np.asarray(base_fx_spots, dtype=np.float64)
    stressed_spots = np.asarray(stressed_fx_spots, dtype=np.float64)

    if base_spots.shape != stressed_spots.shape:
        raise ValueError("base and stressed FX paths must have identical shapes")

    def value_scenario(
        spots: NDArray[np.float64],
        domestic_rate: float,
        foreign_rate: float,
        counterparty_curve: ConstantHazardCurve,
        own_curve: ConstantHazardCurve,
    ) -> FXStressValuation:
        portfolio = value_fx_forward_portfolio(
            trades,
            t,
            spots,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
        )

        exposures = calculate_exposure_profiles(
            t,
            portfolio.portfolio_values,
            confidence_level=pfe_confidence_level,
        )

        discounts = np.exp(-float(domestic_rate) * t)

        xva = calculate_bilateral_xva(
            times=t,
            expected_exposure=exposures.ee,
            expected_negative_exposure=exposures.ene,
            discount_factors=discounts,
            counterparty_credit_curve=counterparty_curve,
            own_credit_curve=own_curve,
            counterparty_recovery_rate=counterparty_recovery_rate,
            own_recovery_rate=own_recovery_rate,
        )

        return FXStressValuation(
            portfolio=portfolio,
            exposures=exposures,
            xva=xva,
            discount_factors=np.asarray(discounts, dtype=np.float64),
        )

    base = value_scenario(
        base_spots,
        base_domestic_rate,
        base_foreign_rate,
        base_counterparty_curve,
        base_own_curve,
    )

    stressed = value_scenario(
        stressed_spots,
        stressed_domestic_rate,
        stressed_foreign_rate,
        stressed_counterparty_curve,
        stressed_own_curve,
    )

    return FXStressComparison(
        times=t.copy(),
        base=base,
        stressed=stressed,
        delta_ee=stressed.exposures.ee - base.exposures.ee,
        delta_ene=stressed.exposures.ene - base.exposures.ene,
        delta_pfe=stressed.exposures.pfe - base.exposures.pfe,
        delta_epe=float(stressed.exposures.epe - base.exposures.epe),
        delta_cva=float(stressed.xva.cva - base.xva.cva),
        delta_dva=float(stressed.xva.dva - base.xva.dva),
        delta_net_xva=float(stressed.xva.net_xva - base.xva.net_xva),
    )
