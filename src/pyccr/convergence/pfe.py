"""Pointwise percentile-bootstrap uncertainty for PFE quantiles.

Resampling is across independent Monte Carlo sampling units:
individual paths for IID sampling, or complete matched pairs
for antithetic sampling.

The resulting intervals are pointwise, not simultaneous bands.
Bootstrap percentile intervals may perform poorly for rare
extreme quantiles, small samples, or degenerate distributions.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.exposure.profiles import calculate_exposure_profiles


@dataclass(frozen=True)
class PFEBootstrapResult:
    """PFE estimates and pointwise bootstrap percentile intervals."""

    times: NDArray[np.float64]
    pfe: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    bootstrap_quantiles: NDArray[np.float64]
    pfe_confidence_level: float
    interval_confidence_level: float
    n_resamples: int
    n_paths: int
    n_independent_units: int
    antithetic: bool


def pfe_bootstrap_intervals(
    times: ArrayLike,
    portfolio_values: ArrayLike,
    *,
    pfe_confidence_level: float = 0.95,
    interval_confidence_level: float = 0.95,
    n_resamples: int = 500,
    seed: int | None = 42,
    antithetic: bool = False,
) -> PFEBootstrapResult:
    """Estimate pointwise PFE percentile-bootstrap intervals.

    Signed portfolio values have shape (n_paths, n_times).
    PFE uses NumPy's linear quantile interpolation.

    In antithetic mode, path i is paired with path i+n_paths/2.
    The function cannot verify that these are genuine matched
    antithetic paths.

    Bootstrap replicates use complete paths or complete pairs,
    retaining dependence across dates within each sampling unit.
    """
    if isinstance(pfe_confidence_level, (bool, np.bool_)):
        raise TypeError("pfe_confidence_level must be numeric")
    if isinstance(interval_confidence_level, (bool, np.bool_)):
        raise TypeError("interval_confidence_level must be numeric")

    for name, value in (
        ("pfe_confidence_level", pfe_confidence_level),
        ("interval_confidence_level", interval_confidence_level),
    ):
        if not np.isscalar(value):
            raise TypeError(f"{name} must be scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
        if not np.isfinite(numeric) or not 0.0 < numeric < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")

    if (
        isinstance(n_resamples, (bool, np.bool_))
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 100
    ):
        raise ValueError("n_resamples must be an integer >= 100")

    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")

    if not isinstance(antithetic, (bool, np.bool_)):
        raise TypeError("antithetic must be boolean")

    pfe_level = float(pfe_confidence_level)
    interval_level = float(interval_confidence_level)

    profiles = calculate_exposure_profiles(
        times,
        portfolio_values,
        confidence_level=pfe_level,
    )
    positive = profiles.positive_exposures
    n_paths, n_times = positive.shape

    if antithetic:
        if n_paths < 4 or n_paths % 2:
            raise ValueError(
                "antithetic requires at least four paths and an even path count"
            )
        n_units = n_paths // 2
    else:
        if n_paths < 2:
            raise ValueError("PFE bootstrap requires at least two paths")
        n_units = n_paths

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(
        (n_resamples, n_times),
        dtype=np.float64,
    )

    # Process one replicate at a time to avoid allocating
    # an n_resamples x n_paths x n_times data cube.
    for index in range(n_resamples):
        sampled_units = rng.integers(0, n_units, size=n_units)
        if antithetic:
            sampled_paths = np.concatenate((sampled_units, sampled_units + n_units))
        else:
            sampled_paths = sampled_units

        bootstrap[index] = np.quantile(
            positive[sampled_paths],
            pfe_level,
            axis=0,
        )

    alpha = 1.0 - interval_level
    lower = np.quantile(bootstrap, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap, 1.0 - alpha / 2.0, axis=0)

    return PFEBootstrapResult(
        times=profiles.times.copy(),
        pfe=profiles.pfe.copy(),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        bootstrap_quantiles=bootstrap,
        pfe_confidence_level=pfe_level,
        interval_confidence_level=interval_level,
        n_resamples=int(n_resamples),
        n_paths=n_paths,
        n_independent_units=n_units,
        antithetic=bool(antithetic),
    )
