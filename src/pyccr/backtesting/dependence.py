"""Descriptive serial-dependence diagnostics for breach indicators.

These diagnostics do not supply dependence-adjusted hypothesis
tests or p-values. Rare breaches can make sample autocorrelations
noisy, and overlapping forecast horizons require special care.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BreachDependenceResult:
    """Descriptive diagnostics for a binary breach sequence."""

    n_observations: int
    n_breaches: int
    breach_frequency: float
    max_lag: int
    lags: NDArray[np.int64]
    autocorrelations: NDArray[np.float64]
    transition_counts: NDArray[np.int64]
    consecutive_breach_pairs: int
    variance_inflation_estimate: float | None
    effective_sample_size_estimate: float | None


def breach_dependence_diagnostics(
    breaches: ArrayLike,
    *,
    max_lag: int = 10,
) -> BreachDependenceResult:
    """Compute descriptive autocorrelations and transition counts.

    Autocorrelations use the full-sample mean and variance, with
    the lagged cross-products normalized by (n - lag). The
    variance-inflation estimate uses the finite-sample weighted
    sum of these estimated correlations. It is not guaranteed to
    be positive for noisy finite samples; in that case no
    effective-sample-size estimate is reported.

    Constant breach sequences have undefined autocorrelations.
    Their autocorrelations and variance-based estimates are NaN
    or None, rather than being represented as zero dependence.
    """
    indicators = np.asarray(breaches)

    if indicators.ndim != 1 or indicators.size < 2 or indicators.dtype.kind != "b":
        raise ValueError("breaches must be a 1D boolean array with at least two values")

    n = int(indicators.size)

    if (
        isinstance(max_lag, (bool, np.bool_))
        or not isinstance(max_lag, (int, np.integer))
        or not 1 <= max_lag < n
    ):
        raise ValueError("max_lag must be an integer between 1 and n_observations - 1")

    values = indicators.astype(np.float64)
    frequency = float(np.mean(values))
    centered = values - frequency
    variance = float(np.mean(centered**2))

    lags = np.arange(1, int(max_lag) + 1, dtype=np.int64)
    autocorrelations = np.full(int(max_lag), np.nan, dtype=np.float64)

    if variance > 0:
        for index, lag in enumerate(lags):
            cross_product = np.mean(centered[:-lag] * centered[lag:])
            autocorrelations[index] = cross_product / variance

        weights = 1.0 - lags / n
        inflation = float(1.0 + 2.0 * np.sum(weights * autocorrelations))

        effective_n = (
            float(n / inflation) if inflation > 0 and np.isfinite(inflation) else None
        )
    else:
        inflation = None
        effective_n = None

    transitions = np.zeros((2, 2), dtype=np.int64)
    for previous, current in pairwise(indicators):
        transitions[int(previous), int(current)] += 1

    return BreachDependenceResult(
        n_observations=n,
        n_breaches=int(np.count_nonzero(indicators)),
        breach_frequency=frequency,
        max_lag=int(max_lag),
        lags=lags,
        autocorrelations=autocorrelations,
        transition_counts=transitions,
        consecutive_breach_pairs=int(transitions[1, 1]),
        variance_inflation_estimate=inflation,
        effective_sample_size_estimate=effective_n,
    )
