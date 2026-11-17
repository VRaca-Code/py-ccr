"""Risk-neutral geometric Brownian motion FX spot simulation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.simulation.config import SimulationConfig


def simulate_fx_spot_paths(
    config: SimulationConfig,
    *,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    normal_draws: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Simulate FX spot quoted as domestic currency per foreign unit.

    Uses exact lognormal transitions with constant domestic and foreign
    continuously compounded interest rates.

    Returns shape (n_paths, n_steps + 1).

    External standard-normal draws permit correlated multi-asset shocks.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance")

    for name, value in (
        ("initial_spot", initial_spot),
        ("domestic_rate", domestic_rate),
        ("foreign_rate", foreign_rate),
        ("volatility", volatility),
    ):
        if not np.isscalar(value):
            raise ValueError(f"{name} must be a finite scalar")

        try:
            finite = bool(np.isfinite(value))
        except TypeError as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc

        if not finite:
            raise ValueError(f"{name} must be finite")

    if initial_spot <= 0:
        raise ValueError("initial_spot must be positive")

    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    shape = (config.n_paths, config.time_grid.n_steps)

    if normal_draws is None:
        rng = np.random.default_rng(config.seed)

        if config.antithetic:
            half = config.n_paths // 2
            half_draws = rng.standard_normal((half, config.time_grid.n_steps))
            draws = np.concatenate(
                (half_draws, -half_draws),
                axis=0,
            )
        else:
            draws = rng.standard_normal(shape)
    else:
        draws = np.asarray(normal_draws, dtype=np.float64)

        if draws.shape != shape:
            raise ValueError(f"normal_draws must have shape {shape}")

        if not np.all(np.isfinite(draws)):
            raise ValueError("normal_draws must contain finite values")

    dt = np.asarray(config.time_grid.dt, dtype=np.float64)

    drift = (domestic_rate - foreign_rate - 0.5 * volatility**2) * dt

    diffusion = volatility * np.sqrt(dt)[None, :] * draws

    log_increments = drift[None, :] + diffusion

    log_spot = np.concatenate(
        (
            np.zeros((config.n_paths, 1), dtype=np.float64),
            np.cumsum(log_increments, axis=1),
        ),
        axis=1,
    )

    return initial_spot * np.exp(log_spot)
