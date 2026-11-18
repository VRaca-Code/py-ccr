"""Tests for numerical term-structure diagnostics."""

import numpy as np
import pytest

from pyccr.dashboard.term_structure import calculate_term_structure_diagnostics
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel


@pytest.fixture
def curve():
    return SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
        zero_rates=(0.02, 0.022, 0.024, 0.026, 0.028, 0.03),
    )


@pytest.fixture
def diagnostics(curve):
    return calculate_term_structure_diagnostics(
        curve,
        mean_reversion=0.08,
        volatility=0.01,
    )


def test_default_grid_size(diagnostics):
    assert diagnostics.times.shape == (121,)


def test_grid_endpoints(diagnostics):
    assert diagnostics.times[0] == 0.0
    assert diagnostics.times[-1] == 3.0


def test_grid_is_increasing(diagnostics):
    assert np.all(np.diff(diagnostics.times) > 0.0)


@pytest.mark.parametrize(
    "field",
    [
        "discount_factors",
        "zero_rates",
        "instantaneous_forwards",
        "forward_slopes",
        "mean_reversion_contribution",
        "volatility_contribution",
        "theta",
    ],
)
def test_all_series_match_grid(diagnostics, field):
    assert getattr(diagnostics, field).shape == diagnostics.times.shape


def test_discount_factors_match_curve(curve, diagnostics):
    np.testing.assert_allclose(
        diagnostics.discount_factors,
        curve.discount_factor(diagnostics.times),
    )


def test_zero_rates_match_curve(curve, diagnostics):
    np.testing.assert_allclose(
        diagnostics.zero_rates,
        curve.zero_rate(diagnostics.times),
    )


def test_forwards_match_curve(curve, diagnostics):
    np.testing.assert_allclose(
        diagnostics.instantaneous_forwards,
        curve.instantaneous_forward(diagnostics.times),
    )


def test_forward_slopes_match_curve(curve, diagnostics):
    np.testing.assert_allclose(
        diagnostics.forward_slopes,
        curve.forward_derivative(diagnostics.times),
    )


def test_drift_decomposition(diagnostics):
    np.testing.assert_allclose(
        diagnostics.theta,
        diagnostics.forward_slopes
        + diagnostics.mean_reversion_contribution
        + diagnostics.volatility_contribution,
    )


def test_theta_matches_model(curve, diagnostics):
    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=0.01,
    )
    np.testing.assert_allclose(
        diagnostics.theta,
        model.theta(diagnostics.times),
    )


def test_zero_volatility_contribution(curve):
    result = calculate_term_structure_diagnostics(
        curve,
        mean_reversion=0.08,
        volatility=0.0,
    )
    np.testing.assert_array_equal(
        result.volatility_contribution,
        np.zeros_like(result.times),
    )


def test_curve_independent_of_model_parameters(curve):
    first = calculate_term_structure_diagnostics(
        curve, mean_reversion=0.05, volatility=0.01
    )
    second = calculate_term_structure_diagnostics(
        curve, mean_reversion=0.20, volatility=0.025
    )
    for name in (
        "discount_factors",
        "zero_rates",
        "instantaneous_forwards",
        "forward_slopes",
    ):
        np.testing.assert_array_equal(
            getattr(first, name),
            getattr(second, name),
        )


def test_custom_grid_size(curve):
    result = calculate_term_structure_diagnostics(
        curve,
        mean_reversion=0.08,
        volatility=0.01,
        n_points=25,
    )
    assert len(result.times) == 25


@pytest.mark.parametrize("invalid", [0, 1, -5, True, 2.5, "25"])
def test_invalid_grid_size(curve, invalid):
    with pytest.raises((TypeError, ValueError)):
        calculate_term_structure_diagnostics(
            curve,
            mean_reversion=0.08,
            volatility=0.01,
            n_points=invalid,
        )


def test_invalid_curve_type():
    with pytest.raises(TypeError):
        calculate_term_structure_diagnostics(
            object(),
            mean_reversion=0.08,
            volatility=0.01,
        )


def test_flat_curve_forward_slope():
    curve = SmoothDiscountCurve(
        tenors=(0.0, 1.0, 2.0, 3.0),
        zero_rates=(0.03,) * 4,
    )
    result = calculate_term_structure_diagnostics(
        curve,
        mean_reversion=0.08,
        volatility=0.01,
    )
    np.testing.assert_allclose(result.forward_slopes, 0.0, atol=1e-12)
