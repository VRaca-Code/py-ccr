"""Tests for Monte Carlo random-number generation."""

import numpy as np
import pytest

from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.random import standard_normal_draws
from pyccr.simulation.time_grid import TimeGrid


@pytest.fixture
def grid() -> TimeGrid:
    return TimeGrid.uniform(horizon=1.0, n_steps=12)


def test_draw_shape(grid: TimeGrid) -> None:
    config = SimulationConfig(time_grid=grid, n_paths=100, seed=42)
    assert standard_normal_draws(config).shape == (100, 12)


def test_same_seed_is_reproducible(grid: TimeGrid) -> None:
    config = SimulationConfig(time_grid=grid, n_paths=100, seed=42)
    np.testing.assert_array_equal(
        standard_normal_draws(config),
        standard_normal_draws(config),
    )


def test_different_seeds_produce_different_draws(grid: TimeGrid) -> None:
    first = SimulationConfig(time_grid=grid, n_paths=100, seed=1)
    second = SimulationConfig(time_grid=grid, n_paths=100, seed=2)

    assert not np.array_equal(
        standard_normal_draws(first),
        standard_normal_draws(second),
    )


def test_antithetic_pairs(grid: TimeGrid) -> None:
    config = SimulationConfig(
        time_grid=grid,
        n_paths=100,
        seed=42,
        antithetic=True,
    )

    draws = standard_normal_draws(config)
    np.testing.assert_array_equal(draws[50:], -draws[:50])


def test_antithetic_shape(grid: TimeGrid) -> None:
    config = SimulationConfig(
        time_grid=grid,
        n_paths=100,
        seed=42,
        antithetic=True,
    )
    assert standard_normal_draws(config).shape == (100, 12)


def test_standard_normal_statistics(grid: TimeGrid) -> None:
    config = SimulationConfig(
        time_grid=grid,
        n_paths=100_000,
        seed=123,
    )
    draws = standard_normal_draws(config)

    assert abs(float(draws.mean())) < 0.01
    assert abs(float(draws.var()) - 1.0) < 0.02


def test_invalid_config() -> None:
    with pytest.raises(TypeError):
        standard_normal_draws(None)


def test_minimum_path_count(grid: TimeGrid) -> None:
    config = SimulationConfig(time_grid=grid, n_paths=1, seed=0)
    assert standard_normal_draws(config).shape == (1, 12)
