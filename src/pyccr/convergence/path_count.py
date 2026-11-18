"""Path-count convergence diagnostics for exposure simulations.

Results from nested prefixes of one simulation are correlated.
They are descriptive convergence diagnostics, not independent
replications or proof of monotonic error reduction.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.convergence.exposure import exposure_mean_intervals
from pyccr.exposure.profiles import calculate_exposure_profiles


@dataclass(frozen=True)
class PathCountConvergenceResult:
    """Exposure statistics evaluated at requested path counts."""

    path_counts: NDArray[np.int64]
    independent_unit_counts: NDArray[np.int64]
    ee: NDArray[np.float64]
    ee_standard_errors: NDArray[np.float64]
    epe: NDArray[np.float64]
    epe_standard_errors: NDArray[np.float64]
    pfe: NDArray[np.float64]
    times: NDArray[np.float64]
    confidence_level: float
    antithetic: bool


def path_count_convergence(
    times: ArrayLike,
    portfolio_values: ArrayLike,
    path_counts: ArrayLike,
    *,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> PathCountConvergenceResult:
    """Evaluate nested Monte Carlo samples.

    For ordinary simulations, use the first N paths.

    For antithetic simulations, the full input must be ordered
    [all first-half paths, all matched second-half paths].
    A count of N selects the first N/2 pairs, taking their
    partners from the second half of the FULL input.

    The function does not generate new random paths.
    """
    if not isinstance(antithetic, (bool, np.bool_)):
        raise TypeError("antithetic must be boolean")

    values = np.asarray(portfolio_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("portfolio_values must be a 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio_values must be finite")

    n_total = values.shape[0]
    counts = np.asarray(path_counts)

    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("path_counts must be a nonempty 1D array")
    if counts.dtype.kind not in "iu":
        raise TypeError("path_counts must contain integers")
    if np.any(counts < 2) or np.any(counts > n_total):
        raise ValueError("path_counts must be between 2 and n_paths")
    if np.any(np.diff(counts) <= 0):
        raise ValueError("path_counts must be strictly increasing")

    if antithetic:
        if n_total < 4 or n_total % 2:
            raise ValueError("antithetic requires an even total path count >= 4")
        if np.any(counts < 4) or np.any(counts % 2):
            raise ValueError("antithetic path_counts must be even and >= 4")

    # Validate the complete input using the existing exposure
    # definition, including time grid and confidence level.
    full = calculate_exposure_profiles(times, values, confidence_level=confidence_level)
    t = full.times
    n_times = t.size

    ee = np.empty((counts.size, n_times), dtype=np.float64)
    ee_se = np.empty_like(ee)
    pfe = np.empty_like(ee)
    epe = np.empty(counts.size, dtype=np.float64)
    epe_se = np.empty_like(epe)

    for index, count in enumerate(counts):
        count = int(count)

        if antithetic:
            half_total = n_total // 2
            half_sample = count // 2
            sample = np.concatenate(
                (
                    values[:half_sample],
                    values[half_total : half_total + half_sample],
                ),
                axis=0,
            )
        else:
            sample = values[:count]

        means = exposure_mean_intervals(
            t,
            sample,
            confidence_level=confidence_level,
            antithetic=antithetic,
        )
        profile = calculate_exposure_profiles(
            t, sample, confidence_level=confidence_level
        )

        ee[index] = means.ee
        ee_se[index] = means.ee_standard_errors
        epe[index] = means.epe.estimate
        epe_se[index] = means.epe.standard_error
        pfe[index] = profile.pfe

    return PathCountConvergenceResult(
        path_counts=counts.astype(np.int64, copy=True),
        independent_unit_counts=(
            counts.astype(np.int64, copy=True) // 2
            if antithetic
            else counts.astype(np.int64, copy=True)
        ),
        ee=ee,
        ee_standard_errors=ee_se,
        epe=epe,
        epe_standard_errors=epe_se,
        pfe=pfe,
        times=t.copy(),
        confidence_level=float(confidence_level),
        antithetic=bool(antithetic),
    )
