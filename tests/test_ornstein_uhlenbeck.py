"""Tests for exact Ornstein-Uhlenbeck simulation."""

import numpy as np
import pytest

from pyccr.models.ornstein_uhlenbeck import simulate_ou
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.time_grid import TimeGrid


def make_config(
    n_paths: int = 100,
    n_steps: int = 12,
    seed: int = 42,
    antithetic: bool = False,
) -> SimulationConfig:
    return SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, n_steps),
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )


def test_output_shape() -> None:
    paths = simulate_ou(make_config(), 0.3, 0.01)
    assert paths.shape == (100, 13)


def test_initial_value() -> None:
    paths = simulate_ou(make_config(), 0.3, 0.01, 0.025)
    np.testing.assert_array_equal(paths[:, 0], 0.025)


def test_reproducibility() -> None:
    config = make_config()
    first = simulate_ou(config, 0.3, 0.01)
    second = simulate_ou(config, 0.3, 0.01)
    np.testing.assert_array_equal(first, second)


def test_zero_volatility() -> None:
    config = make_config(n_steps=20)
    a = 0.4
    x0 = 0.03

    paths = simulate_ou(config, a, 0.0, x0)

    expected = x0 * np.exp(-a * np.asarray(config.time_grid.times))

    np.testing.assert_allclose(
        paths,
        np.broadcast_to(expected, paths.shape),
        rtol=1e-13,
        atol=1e-15,
    )


def test_zero_mean_reversion() -> None:
    config = make_config(n_paths=100_000, n_steps=1)
    sigma = 0.02
    x0 = 0.01

    paths = simulate_ou(config, 0.0, sigma, x0)

    np.testing.assert_allclose(
        paths[:, -1].mean(),
        x0,
        atol=0.0003,
    )

    np.testing.assert_allclose(
        paths[:, -1].var(),
        sigma**2,
        rtol=0.03,
    )


def test_exact_transition_moments() -> None:
    config = make_config(n_paths=100_000, n_steps=1)
    a = 0.35
    sigma = 0.015
    x0 = 0.02
    dt = 1.0

    paths = simulate_ou(config, a, sigma, x0)
    final = paths[:, -1]

    expected_mean = x0 * np.exp(-a * dt)
    expected_variance = sigma**2 * -np.expm1(-2 * a * dt) / (2 * a)

    np.testing.assert_allclose(
        final.mean(),
        expected_mean,
        atol=0.0003,
    )

    np.testing.assert_allclose(
        final.var(),
        expected_variance,
        rtol=0.03,
    )


def test_antithetic_symmetry() -> None:
    config = make_config(
        n_paths=100,
        n_steps=12,
        antithetic=True,
    )

    x0 = 0.02
    a = 0.3

    paths = simulate_ou(config, a, 0.015, x0)

    deterministic = x0 * np.exp(-a * np.asarray(config.time_grid.times))

    paired_sum = paths[:50] + paths[50:]

    expected = np.broadcast_to(
        2 * deterministic,
        paired_sum.shape,
    )

    np.testing.assert_allclose(
        paired_sum,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_nonuniform_time_grid() -> None:
    grid = TimeGrid((0.0, 0.1, 0.4, 1.0))
    config = SimulationConfig(
        time_grid=grid,
        n_paths=20,
        seed=42,
    )

    paths = simulate_ou(config, 0.3, 0.01)

    assert paths.shape == (20, 4)
    assert np.all(np.isfinite(paths))


@pytest.mark.parametrize(
    "mean_reversion, volatility",
    [
        (-0.1, 0.01),
        (0.3, -0.01),
        (float("nan"), 0.01),
        (0.3, float("inf")),
    ],
)
def test_invalid_parameters(
    mean_reversion: float,
    volatility: float,
) -> None:
    with pytest.raises(ValueError):
        simulate_ou(
            make_config(),
            mean_reversion,
            volatility,
        )


def test_invalid_config() -> None:
    with pytest.raises(TypeError):
        simulate_ou(None, 0.3, 0.01)


def test_invalid_initial_value() -> None:
    with pytest.raises(ValueError):
        simulate_ou(
            make_config(),
            0.3,
            0.01,
            float("nan"),
        )
