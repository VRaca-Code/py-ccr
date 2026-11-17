"""Correlated FX spot and Hull-White short-rate scenario simulation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.brownian import brownian_increments
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.hull_white_paths import (
    simulate_hull_white_short_rates,
)


@dataclass(frozen=True)
class JointFXHullWhitePaths:
    """Joint FX and domestic short-rate Monte Carlo scenarios."""

    times: NDArray[np.float64]
    short_rates: NDArray[np.float64]
    fx_spots: NDArray[np.float64]


def simulate_joint_fx_hull_white_paths(
    config: SimulationConfig,
    model: HullWhiteCurveModel,
    *,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    fx_volatility: float,
    correlation: float,
) -> JointFXHullWhitePaths:
    """Simulate correlated Hull-White and FX spot paths.

    The first Brownian factor drives the Hull-White domestic
    short rate. The second drives FX spot.

    The FX drift uses constant domestic and foreign rates;
    it is not yet coupled to the simulated short-rate paths.

    Returns paths with shape (n_paths, n_steps + 1).
    """
    rho = float(correlation)

    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("correlation must be finite and strictly between -1 and 1")

    correlation_matrix = np.array(
        [[1.0, rho], [rho, 1.0]],
        dtype=np.float64,
    )

    increments = brownian_increments(
        config,
        correlation_matrix,
    )

    dt = np.asarray(config.time_grid.dt, dtype=np.float64)

    normal_draws = increments / np.sqrt(dt)[None, :, None]

    short_rates = simulate_hull_white_short_rates(
        config,
        model,
        normal_draws=normal_draws[:, :, 0],
    )

    fx_spots = simulate_fx_spot_paths(
        config,
        initial_spot=initial_spot,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        volatility=fx_volatility,
        normal_draws=normal_draws[:, :, 1],
    )

    return JointFXHullWhitePaths(
        times=np.asarray(
            config.time_grid.times,
            dtype=np.float64,
        ).copy(),
        short_rates=short_rates,
        fx_spots=fx_spots,
    )
