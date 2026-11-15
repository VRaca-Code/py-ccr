"""Correlation matrix validation and Gaussian shock transformation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cholesky_factor(
    correlation: ArrayLike,
) -> NDArray[np.float64]:
    """Validate a correlation matrix and return its Cholesky factor."""
    matrix = np.asarray(correlation, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be a square matrix")

    if matrix.shape[0] == 0:
        raise ValueError("correlation must not be empty")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must contain finite values")

    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("correlation must be symmetric")

    if not np.allclose(
        np.diag(matrix),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("correlation must have a unit diagonal")

    if np.any(np.abs(matrix) > 1.0 + 1e-12):
        raise ValueError("correlations must be between -1 and 1")

    # Avoid silently modifying an asymmetric input.
    symmetric_matrix = (matrix + matrix.T) / 2.0

    try:
        return np.linalg.cholesky(symmetric_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation must be positive definite") from exc


def correlate_normals(
    independent_draws: ArrayLike,
    correlation: ArrayLike,
) -> NDArray[np.float64]:
    """Apply correlation to Gaussian draws along their last axis.

    Expected shape: (..., n_factors).
    """
    draws = np.asarray(independent_draws, dtype=np.float64)
    factor = cholesky_factor(correlation)

    if draws.ndim < 1:
        raise ValueError("independent_draws must have at least one dimension")

    if draws.shape[-1] != factor.shape[0]:
        raise ValueError("last draw dimension must match the number of factors")

    if not np.all(np.isfinite(draws)):
        raise ValueError("independent_draws must contain finite values")

    return draws @ factor.T
