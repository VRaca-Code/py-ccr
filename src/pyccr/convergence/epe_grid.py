"""Deterministic time-grid convergence of FX-linked expected exposure.

This module isolates trapezoidal time-integration error from Monte
Carlo sampling error using analytical lognormal expected exposure.

The illustrative signed value is scale * (spot - strike).
No discounting, collateral, netting or XVA is applied.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import ndtr

from pyccr.simulation.time_grid import TimeGrid


@dataclass(frozen=True)
class EPEGridConvergenceResult:
    """Trapezoidal EPE estimates compared with adaptive quadrature."""

    step_counts: NDArray[np.int64]
    time_steps: NDArray[np.float64]
    epe_estimates: NDArray[np.float64]
    reference_epe: float
    absolute_errors: NDArray[np.float64]
    signed_errors: NDArray[np.float64]
    reference_error_estimate: float
    horizon: float


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def fx_expected_positive_exposure(
    times: ArrayLike,
    *,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    strike: float,
    exposure_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Analytical E[max(scale * (S_t - K), 0)] for lognormal FX.

    Returns an array with the same shape as the input time array.
    This is an undiscounted expectation, not an option price.
    """
    s0 = _finite_scalar(initial_spot, "initial_spot")
    rd = _finite_scalar(domestic_rate, "domestic_rate")
    rf = _finite_scalar(foreign_rate, "foreign_rate")
    sigma = _finite_scalar(volatility, "volatility")
    k = _finite_scalar(strike, "strike")
    scale = _finite_scalar(exposure_scale, "exposure_scale")

    if s0 <= 0:
        raise ValueError("initial_spot must be positive")
    if sigma < 0:
        raise ValueError("volatility must be non-negative")
    if k <= 0:
        raise ValueError("strike must be positive")
    if scale <= 0:
        raise ValueError("exposure_scale must be positive")

    try:
        t = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("times must be numeric") from exc

    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("times must be finite and non-negative")

    mean_spot = s0 * np.exp((rd - rf) * t)
    if not np.all(np.isfinite(mean_spot)):
        raise ValueError("FX expected spot is not finite")

    result = np.maximum(mean_spot - k, 0.0)

    positive_time = t > 0
    if sigma > 0 and np.any(positive_time):
        tp = t[positive_time]
        standard_deviation = sigma * np.sqrt(tp)
        d1 = (np.log(s0 / k) + (rd - rf + 0.5 * sigma**2) * tp) / standard_deviation
        d2 = d1 - standard_deviation

        result = np.asarray(result, dtype=np.float64)
        result[positive_time] = mean_spot[positive_time] * ndtr(d1) - k * ndtr(d2)

    return np.asarray(scale * np.maximum(result, 0.0))


def fx_epe_grid_convergence(
    *,
    horizon: float,
    step_counts: tuple[int, ...],
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    strike: float,
    exposure_scale: float = 1.0,
) -> EPEGridConvergenceResult:
    """Measure trapezoidal EPE integration error across uniform grids.

    The reference uses adaptive quadrature with tight tolerances.
    A quadrature error estimate is reported, not a rigorous bound.
    Errors need not decrease monotonically for arbitrary inputs.
    """
    if not isinstance(step_counts, tuple) or not step_counts:
        raise ValueError("step_counts must be a nonempty tuple")

    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in step_counts
    ):
        raise ValueError("step_counts must contain positive integers")

    if any(later <= earlier for earlier, later in pairwise(step_counts)):
        raise ValueError("step_counts must be strictly increasing")

    # Validate the horizon through the existing simulation grid.
    base_grid = TimeGrid.uniform(horizon, 1)
    t_end = base_grid.horizon

    def expected_exposure(times: ArrayLike) -> NDArray[np.float64]:
        return fx_expected_positive_exposure(
            times,
            initial_spot=initial_spot,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            volatility=volatility,
            strike=strike,
            exposure_scale=exposure_scale,
        )

    # Validate all model parameters even when the time horizon
    # is very small.
    expected_exposure(np.asarray([0.0, t_end]))

    reference_integral, quadrature_error = quad(
        lambda time: float(expected_exposure(np.asarray([time]))[0]),
        0.0,
        t_end,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=300,
    )

    reference_epe = float(reference_integral / t_end)

    estimates = []
    for steps in step_counts:
        grid = TimeGrid.uniform(t_end, steps)
        times = np.asarray(grid.times, dtype=np.float64)
        ee = expected_exposure(times)
        estimates.append(float(np.trapezoid(ee, times) / t_end))

    values = np.asarray(estimates, dtype=np.float64)
    signed = values - reference_epe

    return EPEGridConvergenceResult(
        step_counts=np.asarray(step_counts, dtype=np.int64),
        time_steps=t_end / np.asarray(step_counts, dtype=np.float64),
        epe_estimates=values,
        reference_epe=reference_epe,
        absolute_errors=np.abs(signed),
        signed_errors=signed,
        reference_error_estimate=float(quadrature_error / t_end),
        horizon=t_end,
    )
