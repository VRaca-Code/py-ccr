"""Tests for correlation visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pyccr.visualization.correlation import plot_correlation_diagnostics


@pytest.fixture
def matrices() -> tuple[np.ndarray, np.ndarray]:
    target = np.array(
        [
            [1.0, 0.35, -0.25],
            [0.35, 1.0, 0.20],
            [-0.25, 0.20, 1.0],
        ]
    )
    empirical = target.copy()
    empirical[0, 1] = empirical[1, 0] = 0.351
    return target, empirical


def test_returns_figure(
    matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    target, empirical = matrices
    fig = plot_correlation_diagnostics(
        target, empirical, ["Domestic IR", "Foreign IR", "FX"]
    )

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 3
    finally:
        plt.close(fig)


def test_two_factor_matrices() -> None:
    target = np.array([[1.0, 0.3], [0.3, 1.0]])
    fig = plot_correlation_diagnostics(target, target, ["Factor A", "Factor B"])

    try:
        assert isinstance(fig, Figure)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "target, empirical, labels",
    [
        (np.ones((2, 3)), np.ones((2, 3)), ["A", "B"]),
        (np.eye(2), np.eye(3), ["A", "B"]),
        (np.eye(2), np.eye(2), ["A"]),
        (np.array([[np.nan]]), np.array([[1.0]]), ["A"]),
        (np.array([[1.0]]), np.array([[np.inf]]), ["A"]),
        (np.empty((0, 0)), np.empty((0, 0)), []),
    ],
)
def test_invalid_inputs(
    target: np.ndarray,
    empirical: np.ndarray,
    labels: list[str],
) -> None:
    with pytest.raises(ValueError):
        plot_correlation_diagnostics(target, empirical, labels)
