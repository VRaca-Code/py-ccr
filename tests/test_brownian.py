"""Tests for correlated Brownian motion."""

import numpy as np
import pytest

from pyccr.simulation.brownian import (
    brownian_increments,
    brownian_paths,
)
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.time_grid import TimeGrid


def test_output_shape() -> None:
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 12),
        n_paths=100,
        seed=42,
    )

    increments = brownian_increments(config, np.eye(3))

    assert increments.shape == (100, 12, 3)


def test_reproducibility() -> None:
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 10),
        n_paths=50,
        seed=123,
    )

    first = brownian_increments(config, np.eye(2))
    second = brownian_increments(config, np.eye(2))

    np.testing.assert_array_equal(first, second)


def test_antithetic_pairs() -> None:
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=100,
        seed=7,
        antithetic=True,
    )

    increments = brownian_increments(config, np.eye(3))

    np.testing.assert_array_equal(
        increments[:50],
        -increments[50:],
    )


def test_nonuniform_time_scaling() -> None:
    grid = TimeGrid((0.0, 0.1, 0.4, 1.0))
    config = SimulationConfig(
        time_grid=grid,
        n_paths=100_000,
        seed=42,
    )

    increments = brownian_increments(config, np.eye(1))

    observed = np.var(increments[:, :, 0], axis=0)
    expected = np.asarray(grid.dt)

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=0.03,
    )


def test_cross_factor_covariance() -> None:
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 1),
        n_paths=100_000,
        seed=42,
    )

    correlation = np.array(
        [
            [1.0, 0.35, -0.25],
            [0.35, 1.0, 0.20],
            [-0.25, 0.20, 1.0],
        ]
    )

    increments = brownian_increments(config, correlation)
    empirical = np.cov(increments[:, 0, :], rowvar=False)

    np.testing.assert_allclose(
        empirical,
        correlation,
        atol=0.02,
    )


def test_paths_start_at_zero() -> None:
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 5),
        n_paths=20,
        seed=42,
    )

    increments = brownian_increments(config, np.eye(2))
    paths = brownian_paths(increments)

    assert paths.shape == (20, 6, 2)
    np.testing.assert_array_equal(paths[:, 0, :], 0.0)
    np.testing.assert_allclose(
        np.diff(paths, axis=1),
        increments,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "correlation",
    [
        [[1.0, 1.5], [1.5, 1.0]],
        [[1.0, 0.2], [0.3, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ],
)
def test_invalid_correlation(correlation: list[list[float]]) -> None:
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 2),
        n_paths=10,
    )

    with pytest.raises(ValueError):
        brownian_increments(config, correlation)


def test_invalid_config() -> None:
    with pytest.raises(TypeError):
        brownian_increments(None, np.eye(2))


def test_invalid_path_dimensions() -> None:
    with pytest.raises(ValueError):
        brownian_paths(np.ones((10, 5)))


def test_nonfinite_path_increments() -> None:
    with pytest.raises(ValueError):
        brownian_paths(np.array([[[np.nan]]]))
