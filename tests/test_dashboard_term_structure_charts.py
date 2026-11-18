"""Tests for Plotly term-structure chart construction."""

import numpy as np
import pytest

from app.term_structure_charts import (
    discount_factor_chart,
    forward_curve_chart,
    forward_slope_chart,
    hull_white_drift_chart,
    parameter_comparison_chart,
)
from pyccr.dashboard.term_structure import calculate_term_structure_diagnostics
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve


@pytest.fixture
def curve():
    return SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
        zero_rates=(0.02, 0.022, 0.024, 0.026, 0.028, 0.03),
    )


@pytest.fixture
def baseline(curve):
    return calculate_term_structure_diagnostics(
        curve, mean_reversion=0.08, volatility=0.01
    )


@pytest.fixture
def alternative(curve):
    return calculate_term_structure_diagnostics(
        curve, mean_reversion=0.15, volatility=0.02
    )


def test_discount_chart_has_one_trace(baseline):
    assert len(discount_factor_chart(baseline).data) == 1


def test_discount_chart_values(baseline):
    np.testing.assert_allclose(
        discount_factor_chart(baseline).data[0].y,
        baseline.discount_factors,
    )


def test_forward_chart_has_two_traces(baseline):
    assert len(forward_curve_chart(baseline).data) == 2


def test_forward_chart_rate_units(baseline):
    fig = forward_curve_chart(baseline)
    np.testing.assert_allclose(fig.data[0].y, 100 * baseline.zero_rates)
    np.testing.assert_allclose(fig.data[1].y, 100 * baseline.instantaneous_forwards)


def test_slope_chart_has_one_data_trace(baseline):
    assert len(forward_slope_chart(baseline).data) == 1


def test_slope_chart_basis_point_units(baseline):
    np.testing.assert_allclose(
        forward_slope_chart(baseline).data[0].y,
        10000 * baseline.forward_slopes,
    )


def test_drift_chart_has_four_traces(baseline):
    assert len(hull_white_drift_chart(baseline).data) == 4


def test_drift_chart_components(baseline):
    fig = hull_white_drift_chart(baseline)
    components = (
        baseline.forward_slopes,
        baseline.mean_reversion_contribution,
        baseline.volatility_contribution,
        baseline.theta,
    )
    for trace, component in zip(fig.data, components, strict=True):
        np.testing.assert_allclose(trace.y, 10000 * component)


def test_comparison_has_four_traces(baseline, alternative):
    assert len(parameter_comparison_chart(baseline, alternative).data) == 4


def test_comparison_forward_unchanged(baseline, alternative):
    fig = parameter_comparison_chart(baseline, alternative)
    np.testing.assert_array_equal(fig.data[0].y, fig.data[2].y)


def test_comparison_drift_changes(baseline, alternative):
    fig = parameter_comparison_chart(baseline, alternative)
    assert not np.array_equal(fig.data[1].y, fig.data[3].y)


def test_comparison_rejects_different_grid(curve, baseline):
    different = calculate_term_structure_diagnostics(
        curve, mean_reversion=0.15, volatility=0.02, n_points=31
    )
    with pytest.raises(ValueError, match="same time grid"):
        parameter_comparison_chart(baseline, different)
