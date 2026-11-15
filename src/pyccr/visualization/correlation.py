"""Correlation diagnostic plots using Seaborn."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_correlation_diagnostics(
    target: ArrayLike,
    empirical: ArrayLike,
    labels: list[str],
) -> Figure:
    """Plot target, empirical, and correlation-error heatmaps."""
    target_matrix = np.asarray(target, dtype=np.float64)
    empirical_matrix = np.asarray(empirical, dtype=np.float64)

    if (
        target_matrix.ndim != 2
        or target_matrix.shape[0] == 0
        or target_matrix.shape[0] != target_matrix.shape[1]
    ):
        raise ValueError("target must be a nonempty square matrix")

    if empirical_matrix.shape != target_matrix.shape:
        raise ValueError("matrix shapes must match")

    if len(labels) != target_matrix.shape[0]:
        raise ValueError("label count must match matrix dimension")

    if not np.all(np.isfinite(target_matrix)):
        raise ValueError("target must contain finite values")

    if not np.all(np.isfinite(empirical_matrix)):
        raise ValueError("empirical must contain finite values")

    error = empirical_matrix - target_matrix

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    matrices = [target_matrix, empirical_matrix, error]
    titles = [
        "Target correlation",
        "Simulated correlation",
        "Simulation error",
    ]

    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        is_error = title == "Simulation error"

        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            center=0,
            vmin=-0.02 if is_error else -1.0,
            vmax=0.02 if is_error else 1.0,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
        )

        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig
