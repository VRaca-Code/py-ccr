"""Discrete unilateral credit valuation adjustment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve


@dataclass(frozen=True)
class CVAResult:
    """CVA amount and contributions over successive time intervals."""

    cva: float
    times: NDArray[np.float64]
    marginal_default_probabilities: NDArray[np.float64]
    discounted_expected_exposures: NDArray[np.float64]
    contributions: NDArray[np.float64]


def calculate_unilateral_cva(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    discount_factors: ArrayLike,
    credit_curve: ConstantHazardCurve,
    recovery_rate: float = 0.4,
) -> CVAResult:
    """Calculate endpoint-discretized, independent-default unilateral CVA.

    Expected exposure and discount factors are sampled at every grid
    point, including time zero. The first grid point must be zero.
    Each interval uses the exposure and discount factor at its end.
    """
    t = np.asarray(times, dtype=np.float64)
    ee = np.asarray(expected_exposure, dtype=np.float64)
    df = np.asarray(discount_factors, dtype=np.float64)
    recovery = float(recovery_rate)

    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a one-dimensional grid")

    if not np.all(np.isfinite(t)) or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and increase strictly")

    if ee.shape != t.shape or not np.all(np.isfinite(ee)):
        raise ValueError("expected_exposure must be finite and match times")

    if np.any(ee < 0.0):
        raise ValueError("expected_exposure must be non-negative")

    if df.shape != t.shape or not np.all(np.isfinite(df)):
        raise ValueError("discount_factors must be finite and match times")

    if np.any(df <= 0.0):
        raise ValueError("discount_factors must be strictly positive")

    if not np.isfinite(recovery) or not 0.0 <= recovery <= 1.0:
        raise ValueError("recovery_rate must lie between zero and one")

    if not isinstance(credit_curve, ConstantHazardCurve):
        raise TypeError("credit_curve must be a ConstantHazardCurve")

    marginal_pd = credit_curve.marginal_default_probabilities(t)
    discounted_ee = df[1:] * ee[1:]
    contributions = (1.0 - recovery) * discounted_ee * marginal_pd

    return CVAResult(
        cva=float(np.sum(contributions)),
        times=t.copy(),
        marginal_default_probabilities=marginal_pd,
        discounted_expected_exposures=discounted_ee,
        contributions=contributions,
    )
