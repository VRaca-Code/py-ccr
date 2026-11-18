"""Paired Monte Carlo intervals for nested FX EPE grid differences.

All grids use aggregated increments from one finest Brownian grid.
Pairwise differences preserve common-random-number dependence.

Confidence intervals quantify Monte Carlo sampling uncertainty
in the difference between two grid-specific EPE expectations.
They do not establish a convergence rate or bound model error.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.convergence.simulated_epe_grid import (
    simulated_fx_epe_grid_convergence,
)
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


@dataclass(frozen=True)
class PairedEPEGridResult:
    """Paired estimates for each adjacent nested-grid refinement."""

    coarse_step_counts: NDArray[np.int64]
    fine_step_counts: NDArray[np.int64]
    paired_differences: NDArray[np.float64]
    paired_standard_errors: NDArray[np.float64]
    paired_lower_bounds: NDArray[np.float64]
    paired_upper_bounds: NDArray[np.float64]
    deterministic_differences: NDArray[np.float64]
    monte_carlo_difference_errors: NDArray[np.float64]
    naive_independence_standard_errors: NDArray[np.float64]
    n_paths: int
    n_independent_units: int
    antithetic: bool
    confidence_level: float
    seed: int
    horizon: float


def paired_fx_epe_grid_differences(
    *,
    horizon: float,
    step_counts: tuple[int, ...],
    n_paths: int,
    seed: int,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    strike: float,
    exposure_scale: float = 1.0,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> PairedEPEGridResult:
    """Estimate successive fine-minus-coarse EPE differences.

    Requires at least two nested grids. The finest-grid random
    numbers are shared across all grids. For antithetic sampling,
    each first-half path is paired with its second-half negative
    shock counterpart before computing Student-t uncertainty.

    The naive independence standard error is supplied solely as
    a diagnostic comparison and must not be used for inference
    about the paired difference.
    """
    if not isinstance(step_counts, tuple) or len(step_counts) < 2:
        raise ValueError("step_counts must contain at least two grids")

    # Reuse the existing validation and deterministic benchmark.
    # Its results also provide an independent consistency check
    # on the paired pathwise estimates.
    baseline = simulated_fx_epe_grid_convergence(
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

    finest_steps = step_counts[-1]
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        first = rng.standard_normal((half, finest_steps))
        finest_normals = np.concatenate((first, -first), axis=0)
    else:
        finest_normals = rng.standard_normal((n_paths, finest_steps))

    pathwise_epe = []

    for count in step_counts:
        aggregation = finest_steps // count

        if aggregation == 1:
            normals = finest_normals
        else:
            normals = finest_normals.reshape(n_paths, count, aggregation).sum(
                axis=2
            ) / np.sqrt(aggregation)

        grid = TimeGrid.uniform(baseline.horizon, count)
        config = SimulationConfig(
            time_grid=grid,
            n_paths=n_paths,
            seed=seed,
            antithetic=antithetic,
        )

        spots = simulate_fx_spot_paths(
            config,
            initial_spot=initial_spot,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            volatility=volatility,
            normal_draws=normals,
        )

        positive = np.maximum(
            float(exposure_scale) * (spots - float(strike)),
            0.0,
        )

        per_path = (
            np.trapezoid(
                positive,
                np.asarray(grid.times),
                axis=1,
            )
            / baseline.horizon
        )

        pathwise_epe.append(per_path)

    # The paired construction must reproduce the existing
    # simulated grid study, up to floating-point rounding.
    reconstructed = np.asarray([np.mean(values) for values in pathwise_epe])
    np.testing.assert_allclose(
        reconstructed,
        baseline.simulated_epe,
        rtol=1e-12,
        atol=1e-12,
    )

    paired_estimates = []
    paired_se = []
    paired_lower = []
    paired_upper = []
    naive_se = []

    for coarse, fine in pairwise(pathwise_epe):
        differences = fine - coarse

        interval = monte_carlo_mean_interval(
            differences,
            confidence_level=confidence_level,
            antithetic=antithetic,
        )

        paired_estimates.append(interval.estimate)
        paired_se.append(interval.standard_error)
        paired_lower.append(interval.lower_bound)
        paired_upper.append(interval.upper_bound)

        if antithetic:
            half = n_paths // 2
            coarse_units = 0.5 * (coarse[:half] + coarse[half:])
            fine_units = 0.5 * (fine[:half] + fine[half:])
        else:
            coarse_units = coarse
            fine_units = fine

        n_units = len(coarse_units)
        naive_se.append(
            float(
                np.sqrt(np.var(coarse_units, ddof=1) + np.var(fine_units, ddof=1))
                / np.sqrt(n_units)
            )
        )

    estimates = np.asarray(paired_estimates, dtype=np.float64)
    deterministic = np.diff(baseline.deterministic_grid_epe)

    np.testing.assert_allclose(
        estimates,
        np.diff(baseline.simulated_epe),
        rtol=1e-11,
        atol=1e-12,
    )

    return PairedEPEGridResult(
        coarse_step_counts=np.asarray(step_counts[:-1], dtype=np.int64),
        fine_step_counts=np.asarray(step_counts[1:], dtype=np.int64),
        paired_differences=estimates,
        paired_standard_errors=np.asarray(paired_se, dtype=np.float64),
        paired_lower_bounds=np.asarray(paired_lower, dtype=np.float64),
        paired_upper_bounds=np.asarray(paired_upper, dtype=np.float64),
        deterministic_differences=deterministic,
        monte_carlo_difference_errors=(estimates - deterministic),
        naive_independence_standard_errors=np.asarray(naive_se, dtype=np.float64),
        n_paths=baseline.n_paths,
        n_independent_units=baseline.n_independent_units,
        antithetic=baseline.antithetic,
        confidence_level=baseline.confidence_level,
        seed=baseline.seed,
        horizon=baseline.horizon,
    )
