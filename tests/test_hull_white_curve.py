"""Regression tests for the curve-consistent Hull-White model."""

import numpy as np
import pytest

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel


@pytest.fixture
def curve():
    return SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        zero_rates=(0.025, 0.027, 0.029, 0.030, 0.032, 0.034, 0.035, 0.036),
    )


@pytest.fixture
def model(curve):
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=0.01,
    )


def test_initial_curve_reproduction(model, curve):
    maturities = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 20.0])
    initial_rate = curve.instantaneous_forward(0.0)

    np.testing.assert_allclose(
        model.bond_price(0.0, maturities, initial_rate),
        curve.discount_factor(maturities),
        rtol=1e-12,
        atol=1e-12,
    )


def test_b_factor_at_maturity(model):
    assert float(model.b_factor(2.0, 2.0)) == pytest.approx(0.0)


def test_bond_price_at_maturity(model):
    assert float(model.bond_price(2.0, 2.0, 0.03)) == pytest.approx(1.0)


def test_b_factor_is_positive(model):
    assert np.all(model.b_factor(2.0, [3.0, 5.0, 10.0]) > 0.0)


def test_bond_price_decreases_with_short_rate(model):
    assert float(model.bond_price(1.0, 5.0, 0.02)) > float(
        model.bond_price(1.0, 5.0, 0.05)
    )


def test_zero_volatility_has_no_variance_adjustment(curve):
    deterministic = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=0.0,
    )
    assert float(deterministic.variance_adjustment(2.0, 10.0)) == 0.0


def test_variance_adjustment_is_nonnegative(model):
    assert np.all(model.variance_adjustment(2.0, [3.0, 5.0, 10.0]) >= 0.0)


def test_theta_is_finite(model):
    assert np.all(np.isfinite(model.theta(np.linspace(0.0, 20.0, 101))))


@pytest.mark.parametrize(
    ("mean_reversion", "volatility"),
    [
        (0.0, 0.01),
        (-0.01, 0.01),
        (0.08, -0.01),
        (np.nan, 0.01),
    ],
)
def test_invalid_parameters_rejected(curve, mean_reversion, volatility):
    with pytest.raises(ValueError):
        HullWhiteCurveModel(
            curve=curve,
            mean_reversion=mean_reversion,
            volatility=volatility,
        )


def test_maturity_before_valuation_rejected(model):
    with pytest.raises(ValueError):
        model.bond_price(5.0, 2.0, 0.03)


def test_maturity_beyond_curve_horizon_rejected(model):
    with pytest.raises(ValueError):
        model.bond_price(1.0, 35.0, 0.03)


def test_invalid_short_rate_rejected(model):
    with pytest.raises(ValueError):
        model.bond_price(1.0, 5.0, np.nan)
