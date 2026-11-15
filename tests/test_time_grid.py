"""Unit tests for simulation time grids."""

from dataclasses import FrozenInstanceError

import pytest

from pyccr.simulation.time_grid import TimeGrid


def test_uniform_grid() -> None:
    grid = TimeGrid.uniform(horizon=1.0, n_steps=4)

    assert grid.times == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert grid.dt == (0.25, 0.25, 0.25, 0.25)
    assert grid.n_steps == 4
    assert grid.horizon == 1.0


def test_nonuniform_grid() -> None:
    grid = TimeGrid((0.0, 0.1, 0.4, 1.0))

    assert grid.n_steps == 3
    assert grid.dt == pytest.approx((0.1, 0.3, 0.6))


@pytest.mark.parametrize(
    "times",
    [
        (0.0,),
        (0.0, 0.0),
        (0.0, 1.0, 0.5),
        (0.1, 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
    ],
)
def test_invalid_times(times: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        TimeGrid(times)


@pytest.mark.parametrize(
    "horizon,n_steps",
    [
        (0.0, 4),
        (-1.0, 4),
        (float("inf"), 4),
        (1.0, 0),
        (1.0, -2),
        (1.0, 2.5),
        (1.0, True),
    ],
)
def test_invalid_uniform_parameters(
    horizon: float,
    n_steps: int,
) -> None:
    with pytest.raises(ValueError):
        TimeGrid.uniform(horizon, n_steps)


def test_immutable_grid() -> None:
    grid = TimeGrid.uniform(1.0, 4)

    with pytest.raises(FrozenInstanceError):
        grid.times = (0.0, 1.0)


def test_time_increments_sum_to_horizon() -> None:
    grid = TimeGrid.uniform(2.0, 100)

    assert sum(grid.dt) == pytest.approx(grid.horizon)


def test_reproducible_grid() -> None:
    first = TimeGrid.uniform(1.0, 12)
    second = TimeGrid.uniform(1.0, 12)

    assert first == second
