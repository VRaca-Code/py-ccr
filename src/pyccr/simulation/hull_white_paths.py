"""Curve-consistent Hull-White one-factor short-rate simulation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.config import SimulationConfig


def simulate_hull_white_short_rates(
    config: SimulationConfig,
    model: HullWhiteCurveModel,
    *,
    normal_draws: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Simulate Hull-White short rates using exact Gaussian transitions.

    Returns shape (n_paths, n_steps + 1).

    The deterministic shift is constructed from the initial instantaneous
    forward curve, preserving the initial term structure.

    External normal draws support shared shocks across risk factors.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance")

    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel instance")

    times = np.asarray(config.time_grid.times, dtype=np.float64)
    dt = np.asarray(config.time_grid.dt, dtype=np.float64)

    if times[-1] > model.curve.tenors[-1]:
        raise ValueError("simulation horizon exceeds curve horizon")

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

    a = model.mean_reversion
    sigma = model.volatility

    # Hull-White decomposition:
    # r(t) = x(t) + phi(t), where x(0) = 0 and
    # dx(t) = -a*x(t)*dt + sigma*dW(t).
    #
    # phi(t) = f(0,t) + sigma^2/(2*a^2)*(1-exp(-a*t))^2.
    forward = np.asarray(
        model.curve.instantaneous_forward(times),
        dtype=np.float64,
    )

    shift = forward + (0.5 * (sigma / a) ** 2 * (-np.expm1(-a * times)) ** 2)

    paths = np.empty(
        (config.n_paths, config.time_grid.n_steps + 1),
        dtype=np.float64,
    )

    paths[:, 0] = shift[0]
    state = np.zeros(config.n_paths, dtype=np.float64)

    for step, delta_t in enumerate(dt):
        decay = np.exp(-a * delta_t)

        variance = sigma**2 * -np.expm1(-2.0 * a * delta_t) / (2.0 * a)

        state = decay * state + np.sqrt(variance) * draws[:, step]

        paths[:, step + 1] = state + shift[step + 1]

    return paths
