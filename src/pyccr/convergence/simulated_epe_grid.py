"""FX simulated EPE grid study with coupled Brownian increments.

Nested grids reuse one finest-grid Brownian realization.
Monte Carlo error and deterministic trapezoidal integration
error are reported separately.

The illustrative exposure is max(scale * (S_t - strike), 0).
There is no discounting, collateral, netting, or XVA.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.convergence.epe_grid import fx_epe_grid_convergence
from pyccr.convergence.exposure import exposure_mean_intervals
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


@dataclass(frozen=True)
class SimulatedEPEGridResult:
    """Grid EPE estimates and distinct uncertainty/error components."""

    step_counts: NDArray[np.int64]
    time_steps: NDArray[np.float64]
    simulated_epe: NDArray[np.float64]
    monte_carlo_standard_errors: NDArray[np.float64]
    monte_carlo_lower_bounds: NDArray[np.float64]
    monte_carlo_upper_bounds: NDArray[np.float64]
    deterministic_grid_epe: NDArray[np.float64]
    continuous_reference_epe: float
    deterministic_grid_errors: NDArray[np.float64]
    monte_carlo_deviations: NDArray[np.float64]
    total_reference_deviations: NDArray[np.float64]
    reference_error_estimate: float
    n_paths: int
    n_independent_units: int
    antithetic: bool
    confidence_level: float
    seed: int
    horizon: float


def simulated_fx_epe_grid_convergence(
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
) -> SimulatedEPEGridResult:
    """Compare simulated EPE on nested grids with analytical benchmarks.

    Every coarser grid must divide the finest step count. Standard
    normals on the finest grid are aggregated into Brownian
    increments on coarser grids. This provides common random
    numbers, not independent grid estimates.

    Student-t intervals estimate Monte Carlo uncertainty for each
    grid-specific EPE expectation. They do not describe the
    deterministic quadrature bias or model uncertainty.

    Antithetic paths are paired by the simulation engine's
    first-half/second-half convention.
    """
    if not isinstance(antithetic, bool):
        raise TypeError("antithetic must be boolean")

    if (
        isinstance(n_paths, bool)
        or not isinstance(n_paths, int)
        or n_paths < (4 if antithetic else 2)
    ):
        raise ValueError("n_paths is too small")

    if antithetic and n_paths % 2:
        raise ValueError("antithetic n_paths must be even")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    # This also validates the horizon, step counts, model inputs,
    # and deterministic continuous-time EPE reference.
    benchmark = fx_epe_grid_convergence(
        horizon=horizon,
        step_counts=step_counts,
        initial_spot=initial_spot,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        volatility=volatility,
        strike=strike,
        exposure_scale=exposure_scale,
    )

    finest_steps = step_counts[-1]
    if any(finest_steps % count for count in step_counts):
        raise ValueError("each step count must divide the finest step count")

    # The finest-grid normals are the sole random-number source.
    # For antithetic sampling, the second half negates the first.
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        first_half = rng.standard_normal((half, finest_steps))
        finest_normals = np.concatenate((first_half, -first_half), axis=0)
    else:
        finest_normals = rng.standard_normal((n_paths, finest_steps))

    simulated = []
    standard_errors = []
    lower_bounds = []
    upper_bounds = []
    independent_units = None

    for count in step_counts:
        aggregation = finest_steps // count
        if aggregation == 1:
            normals = finest_normals
        else:
            normals = finest_normals.reshape(n_paths, count, aggregation).sum(
                axis=2
            ) / np.sqrt(aggregation)

        grid = TimeGrid.uniform(benchmark.horizon, count)
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

        signed = float(exposure_scale) * (spots - float(strike))
        result = exposure_mean_intervals(
            grid.times,
            signed,
            confidence_level=confidence_level,
            antithetic=antithetic,
        )

        simulated.append(result.epe.estimate)
        standard_errors.append(result.epe.standard_error)
        lower_bounds.append(result.epe.lower_bound)
        upper_bounds.append(result.epe.upper_bound)
        independent_units = result.n_independent_units

    estimates = np.asarray(simulated, dtype=np.float64)
    grid_expected = benchmark.epe_estimates.copy()

    return SimulatedEPEGridResult(
        step_counts=benchmark.step_counts.copy(),
        time_steps=benchmark.time_steps.copy(),
        simulated_epe=estimates,
        monte_carlo_standard_errors=np.asarray(standard_errors, dtype=np.float64),
        monte_carlo_lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        monte_carlo_upper_bounds=np.asarray(upper_bounds, dtype=np.float64),
        deterministic_grid_epe=grid_expected,
        continuous_reference_epe=benchmark.reference_epe,
        deterministic_grid_errors=benchmark.signed_errors.copy(),
        monte_carlo_deviations=estimates - grid_expected,
        total_reference_deviations=(estimates - benchmark.reference_epe),
        reference_error_estimate=(benchmark.reference_error_estimate),
        n_paths=n_paths,
        n_independent_units=int(independent_units),
        antithetic=antithetic,
        confidence_level=float(confidence_level),
        seed=seed,
        horizon=benchmark.horizon,
    )
