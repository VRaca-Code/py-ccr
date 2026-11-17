"""Discrete unilateral debit valuation adjustment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve


@dataclass(frozen=True)
class DVAResult:
    """Positive DVA benefit and contributions over successive intervals."""

    dva: float
    times: NDArray[np.float64]
    marginal_default_probabilities: NDArray[np.float64]
    discounted_expected_negative_exposures: NDArray[np.float64]
    contributions: NDArray[np.float64]


def calculate_unilateral_dva(
    times: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    own_credit_curve: ConstantHazardCurve,
    recovery_rate: float = 0.4,
) -> DVAResult:
    """Calculate endpoint-discretized, independent-default unilateral DVA.

    Expected negative exposure is the non-negative magnitude of negative
    portfolio MTM, as provided by ExposureProfiles.ene.

    Each interval uses the exposure and discount factor at its endpoint.
    DVA is reported as a positive own-default valuation benefit.

    This is not a bilateral first-to-default model.
    """
    t = np.asarray(times, dtype=np.float64)
    ene = np.asarray(expected_negative_exposure, dtype=np.float64)
    df = np.asarray(discount_factors, dtype=np.float64)
    recovery = float(recovery_rate)

    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a one-dimensional grid")

    if not np.all(np.isfinite(t)) or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and increase strictly")

    if ene.shape != t.shape or not np.all(np.isfinite(ene)):
        raise ValueError("expected_negative_exposure must be finite and match times")

    if np.any(ene < 0.0):
        raise ValueError("expected_negative_exposure must be non-negative")

    if df.shape != t.shape or not np.all(np.isfinite(df)):
        raise ValueError("discount_factors must be finite and match times")

    if np.any(df <= 0.0):
        raise ValueError("discount_factors must be strictly positive")

    if not np.isfinite(recovery) or not 0.0 <= recovery <= 1.0:
        raise ValueError("recovery_rate must lie between zero and one")

    if not isinstance(own_credit_curve, ConstantHazardCurve):
        raise TypeError("own_credit_curve must be a ConstantHazardCurve")

    marginal_pd = own_credit_curve.marginal_default_probabilities(t)
    discounted_ene = df[1:] * ene[1:]
    contributions = (1.0 - recovery) * discounted_ene * marginal_pd

    return DVAResult(
        dva=float(np.sum(contributions)),
        times=t.copy(),
        marginal_default_probabilities=marginal_pd,
        discounted_expected_negative_exposures=discounted_ene,
        contributions=contributions,
    )
