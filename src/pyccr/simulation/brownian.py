"""Correlated Brownian motion for multi-factor Monte Carlo simulations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.correlation import cholesky_factor


def brownian_increments(
    config: SimulationConfig,
    correlation: ArrayLike,
) -> NDArray[np.float64]:
    """Generate correlated Brownian increments.

    Returns an array with shape (n_paths, n_steps, n_factors).
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance")

    factor = cholesky_factor(correlation)
    n_factors = factor.shape[0]
    n_steps = config.time_grid.n_steps

    rng = np.random.default_rng(config.seed)

    if config.antithetic:
        half = config.n_paths // 2
        independent = rng.standard_normal((half, n_steps, n_factors))
        independent = np.concatenate(
            (independent, -independent),
            axis=0,
        )
    else:
        independent = rng.standard_normal((config.n_paths, n_steps, n_factors))

    correlated = independent @ factor.T

    dt = np.asarray(config.time_grid.dt, dtype=np.float64)
    return correlated * np.sqrt(dt)[None, :, None]


def brownian_paths(
    increments: ArrayLike,
) -> NDArray[np.float64]:
    """Integrate Brownian increments, including the initial zero."""
    values = np.asarray(increments, dtype=np.float64)

    if values.ndim != 3:
        raise ValueError("increments must have three dimensions")

    if not np.all(np.isfinite(values)):
        raise ValueError("increments must contain finite values")

    initial = np.zeros(
        (values.shape[0], 1, values.shape[2]),
        dtype=np.float64,
    )

    return np.concatenate(
        (initial, np.cumsum(values, axis=1)),
        axis=1,
    )
