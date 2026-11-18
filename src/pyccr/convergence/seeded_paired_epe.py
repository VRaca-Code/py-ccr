"""Independent-seed validation of paired FX EPE grid differences.

Within-run uncertainty uses paired paths or antithetic pairs.
Across-run uncertainty treats independently seeded runs as units.

Across-run Student-t intervals describe the expected paired-grid
difference. They are not intervals for an individual run, and
they do not establish a numerical convergence order.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.convergence.paired_epe_grid import (
    paired_fx_epe_grid_differences,
)


@dataclass(frozen=True)
class SeededPairedEPEResult:
    """Paired-grid estimates across independent simulation seeds."""

    seeds: NDArray[np.int64]
    coarse_step_counts: NDArray[np.int64]
    fine_step_counts: NDArray[np.int64]
    run_differences: NDArray[np.float64]
    run_standard_errors: NDArray[np.float64]
    run_lower_bounds: NDArray[np.float64]
    run_upper_bounds: NDArray[np.float64]
    across_run_means: NDArray[np.float64]
    across_run_standard_errors: NDArray[np.float64]
    across_run_lower_bounds: NDArray[np.float64]
    across_run_upper_bounds: NDArray[np.float64]
    deterministic_differences: NDArray[np.float64]
    across_run_sampling_errors: NDArray[np.float64]
    n_runs: int
    n_paths_per_run: int
    n_independent_units_per_run: int
    antithetic: bool
    confidence_level: float
    horizon: float


def independent_seed_paired_fx_epe_study(
    *,
    horizon: float,
    step_counts: tuple[int, ...],
    seeds: tuple[int, ...],
    n_paths: int,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    strike: float,
    exposure_scale: float = 1.0,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> SeededPairedEPEResult:
    """Estimate paired EPE grid differences over independent seeds.

    Each run uses common random numbers across its nested grids.
    Different seeds generate separate simulation runs.

    The across-run Student-t interval uses the number of runs,
    not the total number of paths, as its sample size.

    Within-run standard errors are reported separately and are
    not added to across-run standard errors.
    """
    if not isinstance(seeds, tuple) or len(seeds) < 3:
        raise ValueError("seeds must be a tuple of at least three integers")

    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must contain non-negative integers")

    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")

    # int64 is the result representation. Reject values that
    # cannot be represented rather than silently overflowing.
    if any(seed > np.iinfo(np.int64).max for seed in seeds):
        raise ValueError("seeds must fit in signed int64")

    runs = []

    for seed in seeds:
        run = paired_fx_epe_grid_differences(
            horizon=horizon,
            step_counts=step_counts,
            n_paths=n_paths,
            seed=seed,
            initial_spot=initial_spot,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            volatility=volatility,
            strike=strike,
            exposure_scale=exposure_scale,
            confidence_level=confidence_level,
            antithetic=antithetic,
        )
        runs.append(run)

    differences = np.stack(
        [run.paired_differences for run in runs],
        axis=0,
    )
    within_se = np.stack(
        [run.paired_standard_errors for run in runs],
        axis=0,
    )
    within_lower = np.stack(
        [run.paired_lower_bounds for run in runs],
        axis=0,
    )
    within_upper = np.stack(
        [run.paired_upper_bounds for run in runs],
        axis=0,
    )

    # Each column contains one grid refinement evaluated
    # independently across all simulation seeds.
    intervals = [
        monte_carlo_mean_interval(
            differences[:, column],
            confidence_level=confidence_level,
        )
        for column in range(differences.shape[1])
    ]

    across_means = np.asarray(
        [item.estimate for item in intervals],
        dtype=np.float64,
    )
    across_se = np.asarray(
        [item.standard_error for item in intervals],
        dtype=np.float64,
    )
    across_lower = np.asarray(
        [item.lower_bound for item in intervals],
        dtype=np.float64,
    )
    across_upper = np.asarray(
        [item.upper_bound for item in intervals],
        dtype=np.float64,
    )

    deterministic = runs[0].deterministic_differences.copy()

    for run in runs[1:]:
        np.testing.assert_allclose(
            run.deterministic_differences,
            deterministic,
            rtol=0.0,
            atol=0.0,
        )

    return SeededPairedEPEResult(
        seeds=np.asarray(seeds, dtype=np.int64),
        coarse_step_counts=runs[0].coarse_step_counts.copy(),
        fine_step_counts=runs[0].fine_step_counts.copy(),
        run_differences=differences,
        run_standard_errors=within_se,
        run_lower_bounds=within_lower,
        run_upper_bounds=within_upper,
        across_run_means=across_means,
        across_run_standard_errors=across_se,
        across_run_lower_bounds=across_lower,
        across_run_upper_bounds=across_upper,
        deterministic_differences=deterministic,
        across_run_sampling_errors=across_means - deterministic,
        n_runs=len(seeds),
        n_paths_per_run=runs[0].n_paths,
        n_independent_units_per_run=(runs[0].n_independent_units),
        antithetic=runs[0].antithetic,
        confidence_level=runs[0].confidence_level,
        horizon=runs[0].horizon,
    )
