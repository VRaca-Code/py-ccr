"""Terminal-moment controls across exact-transition simulation grids.

The FX lognormal and Hull-White OU state transitions are exact
over each time interval. Their terminal marginal distributions
should therefore be invariant to the number of intervals.

Separate seeds are used for separate grids. Differences between
grid estimates include Monte Carlo noise and are not estimates
of strong convergence error or discretization order.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.hull_white_paths import (
    simulate_hull_white_short_rates,
)
from pyccr.simulation.time_grid import TimeGrid


@dataclass(frozen=True)
class TerminalGridConvergenceResult:
    """Terminal means and sampling uncertainty for each grid."""

    model_name: str
    step_counts: NDArray[np.int64]
    seeds: NDArray[np.int64]
    horizon: float
    n_paths: int
    n_independent_units: int
    antithetic: bool
    analytic_terminal_mean: float
    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    errors: NDArray[np.float64]
    confidence_level: float


def _validate_study(
    step_counts: tuple[int, ...],
    seeds: tuple[int, ...],
    n_paths: int,
    antithetic: bool,
) -> None:
    if not isinstance(step_counts, tuple) or not step_counts:
        raise ValueError("step_counts must be a nonempty tuple")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in step_counts
    ):
        raise ValueError("step_counts must contain positive integers")
    if any(later <= earlier for earlier, later in itertools.pairwise(step_counts)):
        raise ValueError("step_counts must be strictly increasing")
    if not isinstance(seeds, tuple) or len(seeds) != len(step_counts):
        raise ValueError("seeds must match step_counts")
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
        or (antithetic and n_paths % 2 != 0)
    ):
        raise ValueError("n_paths is invalid")


def _study(
    *,
    model_name: str,
    horizon: float,
    step_counts: tuple[int, ...],
    seeds: tuple[int, ...],
    n_paths: int,
    antithetic: bool,
    confidence_level: float,
    analytic_mean: float,
    simulator,
) -> TerminalGridConvergenceResult:
    _validate_study(step_counts, seeds, n_paths, antithetic)

    estimates = []
    errors = []
    lower = []
    upper = []

    for steps, seed in zip(step_counts, seeds, strict=True):
        config = SimulationConfig(
            time_grid=TimeGrid.uniform(horizon, steps),
            n_paths=n_paths,
            seed=seed,
            antithetic=antithetic,
        )
        terminal = simulator(config)
        result = monte_carlo_mean_interval(
            terminal,
            confidence_level=confidence_level,
            antithetic=antithetic,
        )
        estimates.append(result.estimate)
        errors.append(result.standard_error)
        lower.append(result.lower_bound)
        upper.append(result.upper_bound)

    values = np.asarray(estimates, dtype=np.float64)

    return TerminalGridConvergenceResult(
        model_name=model_name,
        step_counts=np.asarray(step_counts, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        horizon=float(horizon),
        n_paths=n_paths,
        n_independent_units=(n_paths // 2 if antithetic else n_paths),
        antithetic=antithetic,
        analytic_terminal_mean=float(analytic_mean),
        estimates=values,
        standard_errors=np.asarray(errors, dtype=np.float64),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        errors=values - analytic_mean,
        confidence_level=float(confidence_level),
    )


def fx_terminal_grid_convergence(
    *,
    horizon: float,
    step_counts: tuple[int, ...],
    seeds: tuple[int, ...],
    n_paths: int,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> TerminalGridConvergenceResult:
    """Compare exact-transition FX terminal means with E[S(T)]."""
    grid = TimeGrid.uniform(horizon, 1)
    analytic = float(
        initial_spot * np.exp((domestic_rate - foreign_rate) * grid.horizon)
    )

    def terminal(config: SimulationConfig) -> NDArray[np.float64]:
        paths = simulate_fx_spot_paths(
            config,
            initial_spot=initial_spot,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            volatility=volatility,
        )
        return paths[:, -1]

    return _study(
        model_name="FX exact lognormal",
        horizon=grid.horizon,
        step_counts=step_counts,
        seeds=seeds,
        n_paths=n_paths,
        antithetic=antithetic,
        confidence_level=confidence_level,
        analytic_mean=analytic,
        simulator=terminal,
    )


def hull_white_terminal_grid_convergence(
    *,
    model: HullWhiteCurveModel,
    horizon: float,
    step_counts: tuple[int, ...],
    seeds: tuple[int, ...],
    n_paths: int,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> TerminalGridConvergenceResult:
    """Compare exact-transition HW terminal means with its shift."""
    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")

    grid = TimeGrid.uniform(horizon, 1)
    if grid.horizon > model.curve.tenors[-1]:
        raise ValueError("horizon exceeds curve horizon")

    a = model.mean_reversion
    sigma = model.volatility
    t = grid.horizon

    analytic = float(
        model.curve.instantaneous_forward(t)
        + 0.5 * (sigma / a) ** 2 * (-np.expm1(-a * t)) ** 2
    )

    def terminal(config: SimulationConfig) -> NDArray[np.float64]:
        paths = simulate_hull_white_short_rates(config, model)
        return paths[:, -1]

    return _study(
        model_name="Hull-White exact OU",
        horizon=t,
        step_counts=step_counts,
        seeds=seeds,
        n_paths=n_paths,
        antithetic=antithetic,
        confidence_level=confidence_level,
        analytic_mean=analytic,
        simulator=terminal,
    )
