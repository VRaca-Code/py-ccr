"""Bilateral first-to-default XVA with independent flat hazard rates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve


@dataclass(frozen=True)
class FirstToDefaultXVAResult:
    """First-to-default CVA, DVA, and signed net valuation adjustment.

    CVA and DVA are nonnegative amounts. Net XVA is DVA minus CVA.
    All interval arrays have length ``len(times) - 1``.
    """

    cva: float
    dva: float
    net_xva: float
    times: NDArray[np.float64]
    counterparty_first_default_probabilities: NDArray[np.float64]
    own_first_default_probabilities: NDArray[np.float64]
    discounted_expected_exposures: NDArray[np.float64]
    discounted_expected_negative_exposures: NDArray[np.float64]
    cva_contributions: NDArray[np.float64]
    dva_contributions: NDArray[np.float64]
    net_contributions: NDArray[np.float64]


def calculate_first_to_default_xva(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    counterparty_credit_curve: ConstantHazardCurve,
    own_credit_curve: ConstantHazardCurve,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
) -> FirstToDefaultXVAResult:
    """Calculate bilateral XVA under independent competing default times.

    Both default times are exponentially distributed with constant
    hazard rates. The probability of each party defaulting first
    within an interval is integrated exactly.

    EE, ENE and discount factors are evaluated at interval endpoints.
    ENE must be the nonnegative magnitude of negative portfolio MTM.

    This calculation assumes exposure/default independence and
    deterministic discount factors. It does not model correlated
    defaults, wrong-way risk, collateral, or closeout conventions
    beyond the stated loss-given-default approximation.
    """
    t = np.asarray(times, dtype=np.float64)
    ee = np.asarray(expected_exposure, dtype=np.float64)
    ene = np.asarray(expected_negative_exposure, dtype=np.float64)
    df = np.asarray(discount_factors, dtype=np.float64)

    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a one-dimensional grid")

    if not np.all(np.isfinite(t)) or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and increase strictly")

    if ee.shape != t.shape or not np.all(np.isfinite(ee)):
        raise ValueError("expected_exposure must be finite and match times")

    if np.any(ee < 0.0):
        raise ValueError("expected_exposure must be non-negative")

    if ene.shape != t.shape or not np.all(np.isfinite(ene)):
        raise ValueError("expected_negative_exposure must be finite and match times")

    if np.any(ene < 0.0):
        raise ValueError("expected_negative_exposure must be non-negative")

    if df.shape != t.shape or not np.all(np.isfinite(df)):
        raise ValueError("discount_factors must be finite and match times")

    if np.any(df <= 0.0):
        raise ValueError("discount_factors must be strictly positive")

    if not isinstance(counterparty_credit_curve, ConstantHazardCurve):
        raise TypeError("counterparty_credit_curve must be a ConstantHazardCurve")

    if not isinstance(own_credit_curve, ConstantHazardCurve):
        raise TypeError("own_credit_curve must be a ConstantHazardCurve")

    cp_recovery = float(counterparty_recovery_rate)
    own_recovery = float(own_recovery_rate)

    if not np.isfinite(cp_recovery) or not 0.0 <= cp_recovery <= 1.0:
        raise ValueError("counterparty_recovery_rate must lie between zero and one")

    if not np.isfinite(own_recovery) or not 0.0 <= own_recovery <= 1.0:
        raise ValueError("own_recovery_rate must lie between zero and one")

    cp_hazard = counterparty_credit_curve.hazard_rate
    own_hazard = own_credit_curve.hazard_rate
    total_hazard = cp_hazard + own_hazard

    if total_hazard == 0.0:
        cp_first = np.zeros(t.size - 1, dtype=np.float64)
        own_first = np.zeros(t.size - 1, dtype=np.float64)
    else:
        dt = np.diff(t)

        interval_first_default = np.exp(-total_hazard * t[:-1]) * -np.expm1(
            -total_hazard * dt
        )

        cp_first = (cp_hazard / total_hazard) * interval_first_default

        own_first = (own_hazard / total_hazard) * interval_first_default

    discounted_ee = df[1:] * ee[1:]
    discounted_ene = df[1:] * ene[1:]

    cva_contributions = (1.0 - cp_recovery) * discounted_ee * cp_first
    dva_contributions = (1.0 - own_recovery) * discounted_ene * own_first

    net_contributions = dva_contributions - cva_contributions

    cva = float(np.sum(cva_contributions))
    dva = float(np.sum(dva_contributions))

    return FirstToDefaultXVAResult(
        cva=cva,
        dva=dva,
        net_xva=float(dva - cva),
        times=t.copy(),
        counterparty_first_default_probabilities=cp_first,
        own_first_default_probabilities=own_first,
        discounted_expected_exposures=discounted_ee,
        discounted_expected_negative_exposures=discounted_ene,
        cva_contributions=cva_contributions,
        dva_contributions=dva_contributions,
        net_contributions=net_contributions,
    )
