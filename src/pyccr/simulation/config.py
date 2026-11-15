"""Validated configuration for Monte Carlo simulations."""

from dataclasses import dataclass

from pyccr.simulation.time_grid import TimeGrid


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable Monte Carlo simulation settings."""

    time_grid: TimeGrid
    n_paths: int
    seed: int | None = None
    antithetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid instance")

        if (
            isinstance(self.n_paths, bool)
            or not isinstance(self.n_paths, int)
            or self.n_paths < 1
        ):
            raise ValueError("n_paths must be a positive integer")

        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")

        if not isinstance(self.antithetic, bool):
            raise TypeError("antithetic must be a boolean")

        if self.antithetic and self.n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic sampling is enabled")
