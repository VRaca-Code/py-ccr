"""Simulation time-grid utilities."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """Immutable simulation time grid measured in year fractions."""

    times: tuple[float, ...]
    dt: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(float(t) for t in self.times)

        if len(values) < 2:
            raise ValueError("TimeGrid requires at least two time points.")

        if not np.all(np.isfinite(values)):
            raise ValueError("All time points must be finite.")

        if values[0] != 0.0:
            raise ValueError("TimeGrid must start at zero.")

        increments = np.diff(values)

        if np.any(increments <= 0.0):
            raise ValueError("Time points must be strictly increasing.")

        object.__setattr__(self, "times", values)
        object.__setattr__(
            self,
            "dt",
            tuple(float(x) for x in increments),
        )

    @property
    def n_steps(self) -> int:
        """Number of simulation intervals."""
        return len(self.dt)

    @property
    def horizon(self) -> float:
        """Final simulation time in years."""
        return self.times[-1]

    @classmethod
    def uniform(
        cls,
        horizon: float,
        n_steps: int,
    ) -> "TimeGrid":
        """Construct a uniform simulation time grid."""
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Horizon must be positive and finite.")

        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")

        times = np.linspace(0.0, horizon, n_steps + 1)
        return cls(tuple(float(t) for t in times))
