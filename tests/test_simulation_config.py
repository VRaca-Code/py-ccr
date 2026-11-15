"""Tests for Monte Carlo simulation configuration."""

from dataclasses import FrozenInstanceError

import pytest

from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.time_grid import TimeGrid


@pytest.fixture
def grid() -> TimeGrid:
    return TimeGrid.uniform(horizon=1.0, n_steps=12)


def test_valid_configuration(grid: TimeGrid) -> None:
    config = SimulationConfig(time_grid=grid, n_paths=10_000, seed=42)

    assert config.time_grid == grid
    assert config.n_paths == 10_000
    assert config.seed == 42
    assert config.antithetic is False


def test_valid_antithetic_configuration(grid: TimeGrid) -> None:
    config = SimulationConfig(
        time_grid=grid,
        n_paths=2_000,
        seed=0,
        antithetic=True,
    )

    assert config.antithetic is True


@pytest.mark.parametrize(
    "n_paths",
    [0, -1, 1.5, True, False, "1000", None],
)
def test_invalid_path_counts(grid: TimeGrid, n_paths: object) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(time_grid=grid, n_paths=n_paths)


@pytest.mark.parametrize(
    "seed",
    [-1, 2.5, True, False, "42"],
)
def test_invalid_seeds(grid: TimeGrid, seed: object) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(time_grid=grid, n_paths=100, seed=seed)


def test_invalid_time_grid() -> None:
    with pytest.raises(TypeError):
        SimulationConfig(time_grid=None, n_paths=100)


def test_invalid_antithetic_type(grid: TimeGrid) -> None:
    with pytest.raises(TypeError):
        SimulationConfig(time_grid=grid, n_paths=100, antithetic=1)


def test_antithetic_requires_even_paths(grid: TimeGrid) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(time_grid=grid, n_paths=101, antithetic=True)


def test_configuration_is_immutable(grid: TimeGrid) -> None:
    config = SimulationConfig(time_grid=grid, n_paths=100)

    with pytest.raises(FrozenInstanceError):
        config.n_paths = 200
