"""Exact simulation of the Ornstein-Uhlenbeck process."""

import numpy as np
from numpy.typing import NDArray

from pyccr.simulation.config import SimulationConfig


def simulate_ou(
    config: SimulationConfig,
    mean_reversion: float,
    volatility: float,
    initial_value: float = 0.0,
) -> NDArray[np.float64]:
    """Simulate exact OU transitions with shape (n_paths, n_steps + 1)."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance")

    for name, value in (
        ("mean_reversion", mean_reversion),
        ("volatility", volatility),
        ("initial_value", initial_value),
    ):
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar")

    if mean_reversion < 0:
        raise ValueError("mean_reversion must be non-negative")

    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    dt = np.asarray(config.time_grid.dt, dtype=np.float64)
    n_steps = config.time_grid.n_steps

    rng = np.random.default_rng(config.seed)

    if config.antithetic:
        half = config.n_paths // 2
        shocks = rng.standard_normal((half, n_steps))
        shocks = np.concatenate((shocks, -shocks), axis=0)
    else:
        shocks = rng.standard_normal((config.n_paths, n_steps))

    paths = np.empty((config.n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = initial_value

    for step, delta_t in enumerate(dt):
        decay = np.exp(-mean_reversion * delta_t)

        if mean_reversion == 0:
            variance = volatility**2 * delta_t
        else:
            variance = (
                volatility**2
                * -np.expm1(-2 * mean_reversion * delta_t)
                / (2 * mean_reversion)
            )

        paths[:, step + 1] = (
            paths[:, step] * decay + np.sqrt(variance) * shocks[:, step]
        )

    return paths
