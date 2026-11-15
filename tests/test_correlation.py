"""Tests for correlation and Cholesky decomposition."""

import numpy as np
import pytest

from pyccr.simulation.correlation import (
    cholesky_factor,
    correlate_normals,
)


@pytest.fixture
def correlation() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.35, -0.25],
            [0.35, 1.0, 0.20],
            [-0.25, 0.20, 1.0],
        ]
    )


def test_cholesky_reconstructs_correlation(
    correlation: np.ndarray,
) -> None:
    factor = cholesky_factor(correlation)

    np.testing.assert_allclose(
        factor @ factor.T,
        correlation,
        rtol=0.0,
        atol=1e-12,
    )


def test_factor_is_lower_triangular(
    correlation: np.ndarray,
) -> None:
    factor = cholesky_factor(correlation)

    np.testing.assert_allclose(factor, np.tril(factor))


def test_correlated_draw_shape(
    correlation: np.ndarray,
) -> None:
    draws = np.random.default_rng(42).standard_normal((100, 12, 3))

    result = correlate_normals(draws, correlation)

    assert result.shape == (100, 12, 3)


def test_empirical_correlation(
    correlation: np.ndarray,
) -> None:
    draws = np.random.default_rng(123).standard_normal((100_000, 3))

    correlated = correlate_normals(draws, correlation)
    empirical = np.corrcoef(correlated, rowvar=False)

    np.testing.assert_allclose(
        empirical,
        correlation,
        rtol=0.0,
        atol=0.015,
    )


def test_identity_correlation_preserves_draws() -> None:
    draws = np.random.default_rng(7).standard_normal((100, 3))

    result = correlate_normals(draws, np.eye(3))

    np.testing.assert_array_equal(result, draws)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.2]],
        [],
        [[1.0, 0.5], [0.3, 1.0]],
        [[1.0, 1.2], [1.2, 1.0]],
        [[1.0, np.nan], [np.nan, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 0.9]],
        [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]],
    ],
)
def test_invalid_correlation_matrix(matrix: object) -> None:
    with pytest.raises(ValueError):
        cholesky_factor(matrix)


def test_wrong_factor_dimension(
    correlation: np.ndarray,
) -> None:
    draws = np.ones((100, 2))

    with pytest.raises(ValueError):
        correlate_normals(draws, correlation)


def test_nonfinite_draws(
    correlation: np.ndarray,
) -> None:
    draws = np.array([[0.1, np.nan, 0.2]])

    with pytest.raises(ValueError):
        correlate_normals(draws, correlation)


def test_reproducible_transformation(
    correlation: np.ndarray,
) -> None:
    draws = np.random.default_rng(42).standard_normal((100, 3))

    first = correlate_normals(draws, correlation)
    second = correlate_normals(draws, correlation)

    np.testing.assert_array_equal(first, second)
