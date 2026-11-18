"""Monte Carlo uncertainty for expected exposure and EPE.

The estimator reuses PyCCR's exposure-profile definitions.
EE uncertainty is calculated across paths at each time.
EPE uncertainty is calculated from each path's time-averaged
positive exposure, retaining within-path time dependence.

These intervals describe Monte Carlo sampling uncertainty,
not market-model uncertainty or PFE quantile uncertainty.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.convergence.mean import (
    MonteCarloMeanResult,
    monte_carlo_mean_interval,
)
from pyccr.exposure.profiles import calculate_exposure_profiles


@dataclass(frozen=True)
class ExposureConvergenceResult:
    """EE time-series intervals and a pathwise EPE interval."""

    times: NDArray[np.float64]
    ee: NDArray[np.float64]
    ee_standard_errors: NDArray[np.float64]
    ee_lower_bounds: NDArray[np.float64]
    ee_upper_bounds: NDArray[np.float64]
    epe: MonteCarloMeanResult
    confidence_level: float
    n_paths: int
    n_independent_units: int
    antithetic: bool


def exposure_mean_intervals(
    times: ArrayLike,
    portfolio_values: ArrayLike,
    *,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> ExposureConvergenceResult:
    """Estimate Monte Carlo EE and EPE uncertainty.

    The input must contain signed portfolio values of shape
    (n_paths, n_times). Positive exposure is max(value, 0).

    Antithetic paths must follow the first-half/second-half
    pairing convention used by PyCCR's simulation engines.
    Pairing cannot be verified from portfolio values alone.

    For a single observation time, EPE equals EE at that time.
    Otherwise EPE uses the same trapezoidal time weights as
    calculate_exposure_profiles.
    """
    profiles = calculate_exposure_profiles(
        times,
        portfolio_values,
        confidence_level=confidence_level,
    )

    positive = profiles.positive_exposures
    t = profiles.times

    # Validate confidence level and antithetic configuration
    # through the common estimator, including n_paths >= 2.
    first = monte_carlo_mean_interval(
        positive[:, 0],
        confidence_level=confidence_level,
        antithetic=antithetic,
    )

    estimates = [first]

    for index in range(1, t.size):
        estimates.append(
            monte_carlo_mean_interval(
                positive[:, index],
                confidence_level=confidence_level,
                antithetic=antithetic,
            )
        )

    if t.size == 1:
        pathwise_epe = positive[:, 0]
    else:
        dt = np.diff(t)
        weights = np.empty(t.size, dtype=np.float64)
        weights[0] = 0.5 * dt[0]
        weights[-1] = 0.5 * dt[-1]

        if t.size > 2:
            weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])

        weights /= t[-1] - t[0]
        pathwise_epe = positive @ weights

    epe_result = monte_carlo_mean_interval(
        pathwise_epe,
        confidence_level=confidence_level,
        antithetic=antithetic,
    )

    ee = np.array(
        [item.estimate for item in estimates],
        dtype=np.float64,
    )
    standard_errors = np.array(
        [item.standard_error for item in estimates],
        dtype=np.float64,
    )
    lower = np.array(
        [item.lower_bound for item in estimates],
        dtype=np.float64,
    )
    upper = np.array(
        [item.upper_bound for item in estimates],
        dtype=np.float64,
    )

    return ExposureConvergenceResult(
        times=t.copy(),
        ee=ee,
        ee_standard_errors=standard_errors,
        ee_lower_bounds=lower,
        ee_upper_bounds=upper,
        epe=epe_result,
        confidence_level=first.confidence_level,
        n_paths=first.n_paths,
        n_independent_units=first.n_independent_units,
        antithetic=first.antithetic,
    )
