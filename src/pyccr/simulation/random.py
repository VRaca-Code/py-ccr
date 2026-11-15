"""Random-number generation for Monte Carlo simulations."""

import numpy as np
from numpy.typing import NDArray

from pyccr.simulation.config import SimulationConfig


def standard_normal_draws(
    config: SimulationConfig,
) -> NDArray[np.float64]:
    """Generate standard normal draws, optionally in antithetic pairs."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance")

    rng = np.random.default_rng(config.seed)
    n_steps = config.time_grid.n_steps

    if config.antithetic:
        half = config.n_paths // 2
        draws = rng.standard_normal((half, n_steps))
        return np.concatenate((draws, -draws), axis=0)

    return rng.standard_normal((config.n_paths, n_steps))
