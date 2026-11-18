"""Numerical diagnostics for the initial curve and Hull-White drift."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel


@dataclass(frozen=True)
class TermStructureDiagnostics:
    """Curve quantities and decomposed Hull-White deterministic drift."""

    times: NDArray[np.float64]
    discount_factors: NDArray[np.float64]
    zero_rates: NDArray[np.float64]
    instantaneous_forwards: NDArray[np.float64]
    forward_slopes: NDArray[np.float64]
    mean_reversion_contribution: NDArray[np.float64]
    volatility_contribution: NDArray[np.float64]
    theta: NDArray[np.float64]


def calculate_term_structure_diagnostics(
    curve: SmoothDiscountCurve,
    *,
    mean_reversion: float,
    volatility: float,
    n_points: int = 121,
) -> TermStructureDiagnostics:
    """Evaluate initial curve and Hull-White drift on a dense maturity grid.

    The initial market curve is independent of Hull-White parameters.
    The drift decomposition is model-dependent.
    """
    if not isinstance(curve, SmoothDiscountCurve):
        raise TypeError("curve must be a SmoothDiscountCurve")

    if isinstance(n_points, bool) or not isinstance(n_points, int):
        raise TypeError("n_points must be an integer")

    if n_points < 2:
        raise ValueError("n_points must be at least two")

    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=mean_reversion,
        volatility=volatility,
    )

    times = np.linspace(
        0.0,
        curve.tenors[-1],
        n_points,
        dtype=np.float64,
    )

    discount_factors = np.asarray(
        curve.discount_factor(times),
        dtype=np.float64,
    )

    zero_rates = np.asarray(
        curve.zero_rate(times),
        dtype=np.float64,
    )

    forwards = np.asarray(
        curve.instantaneous_forward(times),
        dtype=np.float64,
    )

    slopes = np.asarray(
        curve.forward_derivative(times),
        dtype=np.float64,
    )

    mean_reversion_term = mean_reversion * forwards

    volatility_term = (
        volatility**2
        * (-np.expm1(-2.0 * mean_reversion * times))
        / (2.0 * mean_reversion)
    )

    theta = slopes + mean_reversion_term + volatility_term

    np.testing.assert_allclose(
        theta,
        model.theta(times),
        rtol=1e-12,
        atol=1e-12,
    )

    return TermStructureDiagnostics(
        times=times,
        discount_factors=discount_factors,
        zero_rates=zero_rates,
        instantaneous_forwards=forwards,
        forward_slopes=slopes,
        mean_reversion_contribution=mean_reversion_term,
        volatility_contribution=volatility_term,
        theta=theta,
    )
