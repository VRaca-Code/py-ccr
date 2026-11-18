"""Independent-seed convergence study for risk-neutral FX paths.

Different seeds produce independent simulation replications under
the model's pseudorandom-number assumptions. Between-run
variability is distinct from within-run Monte Carlo uncertainty.

These diagnostics do not measure model calibration uncertainty,
historical forecast accuracy, or time-discretization bias.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.convergence.exposure import exposure_mean_intervals
from pyccr.convergence.mean import (
    MonteCarloMeanResult,
    monte_carlo_mean_interval,
)
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


@dataclass(frozen=True)
class IndependentSeedFXResult:
    """Independent FX simulation replication statistics."""

    seeds: NDArray[np.int64]
    n_paths_per_run: int
    n_independent_units_per_run: int
    antithetic: bool
    confidence_level: float
    terminal_spot_means: NDArray[np.float64]
    terminal_spot_analytic_mean: float
    terminal_spot_across_runs: MonteCarloMeanResult
    terminal_ee: NDArray[np.float64]
    terminal_ee_within_standard_errors: NDArray[np.float64]
    terminal_ee_across_runs: MonteCarloMeanResult
    epe: NDArray[np.float64]
    epe_within_standard_errors: NDArray[np.float64]
    epe_across_runs: MonteCarloMeanResult
    terminal_pfe: NDArray[np.float64]
    terminal_pfe_across_runs: MonteCarloMeanResult


def independent_seed_fx_study(
    *,
    time_grid: TimeGrid,
    seeds: tuple[int, ...],
    n_paths: int,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    exposure_scale: float = 1.0,
    exposure_strike: float | None = None,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> IndependentSeedFXResult:
    """Compare FX simulations with distinct explicitly supplied seeds.

    Signed illustrative exposure is exposure_scale * (S_t - K).
    This is not a full discounted or collateralized derivative
    valuation. PFE is computed using confidence_level.

    Across-run Student-t intervals describe uncertainty in the
    mean of the corresponding run-level statistic, not in a
    population PFE quantile or in the underlying risk model.
    """
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid")
    if not isinstance(seeds, tuple) or len(seeds) < 3:
        raise ValueError("seeds must be a tuple of >= 3 integers")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    if not isinstance(antithetic, bool):
        raise TypeError("antithetic must be boolean")
    if (
        isinstance(n_paths, bool)
        or not isinstance(n_paths, int)
        or n_paths < (4 if antithetic else 2)
    ):
        raise ValueError("n_paths is too small for uncertainty")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic n_paths must be even")

    for name, value in (
        ("exposure_scale", exposure_scale),
        ("exposure_strike", exposure_strike),
    ):
        if value is None and name == "exposure_strike":
            continue
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be finite")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")

    if float(exposure_scale) <= 0:
        raise ValueError("exposure_scale must be positive")

    strike = float(initial_spot) if exposure_strike is None else float(exposure_strike)

    spot_means = []
    terminal_ee = []
    terminal_ee_se = []
    epe = []
    epe_se = []
    terminal_pfe = []

    for seed in seeds:
        config = SimulationConfig(
            time_grid=time_grid,
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
        )
        spot_estimate = monte_carlo_mean_interval(
            spots[:, -1],
            confidence_level=confidence_level,
            antithetic=antithetic,
        )

        signed = float(exposure_scale) * (spots - strike)
        means = exposure_mean_intervals(
            time_grid.times,
            signed,
            confidence_level=confidence_level,
            antithetic=antithetic,
        )
        profile = calculate_exposure_profiles(
            time_grid.times,
            signed,
            confidence_level=confidence_level,
        )

        spot_means.append(spot_estimate.estimate)
        terminal_ee.append(float(means.ee[-1]))
        terminal_ee_se.append(float(means.ee_standard_errors[-1]))
        epe.append(means.epe.estimate)
        epe_se.append(means.epe.standard_error)
        terminal_pfe.append(float(profile.pfe[-1]))

    def values(items: list[float]) -> NDArray[np.float64]:
        return np.asarray(items, dtype=np.float64)

    def across(items: list[float]) -> MonteCarloMeanResult:
        return monte_carlo_mean_interval(items, confidence_level=confidence_level)

    return IndependentSeedFXResult(
        seeds=np.asarray(seeds, dtype=np.int64),
        n_paths_per_run=n_paths,
        n_independent_units_per_run=(n_paths // 2 if antithetic else n_paths),
        antithetic=antithetic,
        confidence_level=float(confidence_level),
        terminal_spot_means=values(spot_means),
        terminal_spot_analytic_mean=float(
            initial_spot * np.exp((domestic_rate - foreign_rate) * time_grid.horizon)
        ),
        terminal_spot_across_runs=across(spot_means),
        terminal_ee=values(terminal_ee),
        terminal_ee_within_standard_errors=values(terminal_ee_se),
        terminal_ee_across_runs=across(terminal_ee),
        epe=values(epe),
        epe_within_standard_errors=values(epe_se),
        epe_across_runs=across(epe),
        terminal_pfe=values(terminal_pfe),
        terminal_pfe_across_runs=across(terminal_pfe),
    )
