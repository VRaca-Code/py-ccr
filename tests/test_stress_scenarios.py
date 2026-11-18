"""Deterministic stress-shock validation."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.market.discount_curve import DiscountCurve
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.stress import (
    shock_correlation,
    shock_discount_curve,
    shock_fx_spot,
    shock_hazard_curve,
    shock_smooth_discount_curve,
    shock_volatility,
)


@pytest.fixture
def discount_curve():
    return DiscountCurve(
        tenors=(0.5, 1.0, 2.0),
        zero_rates=(0.01, 0.02, 0.03),
    )


@pytest.fixture
def smooth_curve():
    return SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0),
        zero_rates=(0.01, 0.015, 0.02, 0.03),
    )


@pytest.mark.parametrize("shift_bps", [-100.0, 0.0, 25.0, 200.0])
def test_parallel_discount_curve_shift(discount_curve, shift_bps):
    shifted = shock_discount_curve(discount_curve, shift_bps)

    assert shifted is not discount_curve
    assert shifted.tenors == discount_curve.tenors
    np.testing.assert_allclose(
        shifted.zero_rates,
        np.asarray(discount_curve.zero_rates) + shift_bps * 1e-4,
        atol=1e-15,
    )
    assert discount_curve.zero_rates == (0.01, 0.02, 0.03)


@pytest.mark.parametrize("shift_bps", [-50.0, 0.0, 100.0])
def test_smooth_curve_parallel_shift(smooth_curve, shift_bps):
    shifted = shock_smooth_discount_curve(smooth_curve, shift_bps)
    assert shifted is not smooth_curve
    assert shifted.tenors == smooth_curve.tenors

    times = np.array([0.0, 0.25, 0.75, 1.5, 2.0])
    np.testing.assert_allclose(
        shifted.zero_rate(times),
        smooth_curve.zero_rate(times) + shift_bps * 1e-4,
        atol=1e-14,
    )


@pytest.mark.parametrize("shift_bps", [-100.0, 0.0, 100.0])
def test_discount_factor_shift_identity(discount_curve, shift_bps):
    times = np.array([0.25, 0.75, 1.5, 3.0])
    shocked = shock_discount_curve(discount_curve, shift_bps)

    np.testing.assert_allclose(
        shocked.discount_factor(times),
        discount_curve.discount_factor(times) * np.exp(-times * shift_bps * 1e-4),
        rtol=1e-13,
    )


@pytest.mark.parametrize(
    ("relative_shock", "expected"),
    [(-0.2, 0.88), (0.0, 1.1), (0.1, 1.21)],
)
def test_relative_fx_spot_shock(relative_shock, expected):
    assert shock_fx_spot(1.1, relative_shock) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("absolute_shock", "expected"),
    [(-0.05, 0.15), (0.0, 0.2), (0.1, 0.3)],
)
def test_absolute_volatility_shock(absolute_shock, expected):
    assert shock_volatility(0.2, absolute_shock) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("shift_bps", "expected"),
    [(-100.0, 0.01), (0.0, 0.02), (200.0, 0.04)],
)
def test_hazard_rate_shift(shift_bps, expected):
    base = ConstantHazardCurve(0.02)
    shocked = shock_hazard_curve(base, shift_bps)

    assert shocked is not base
    assert shocked.hazard_rate == pytest.approx(expected)
    assert base.hazard_rate == 0.02


def test_higher_hazard_lowers_survival_probability():
    base = ConstantHazardCurve(0.02)
    shocked = shock_hazard_curve(base, 100.0)
    times = np.array([0.0, 1.0, 3.0])

    assert np.all(
        shocked.survival_probability(times[1:]) < base.survival_probability(times[1:])
    )


@pytest.mark.parametrize(
    ("shock", "expected"),
    [(-0.3, -0.1), (0.0, 0.2), (0.4, 0.6)],
)
def test_two_factor_correlation_shock(shock, expected):
    actual = shock_correlation(0.2, shock)
    assert actual == pytest.approx(expected)

    matrix = np.array([[1.0, actual], [actual, 1.0]])
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (shock_fx_spot, (1.1, -1.0)),
        (shock_fx_spot, (1.1, -1.5)),
        (shock_fx_spot, (0.0, 0.1)),
        (shock_volatility, (0.2, -0.3)),
        (shock_volatility, (-0.1, 0.1)),
        (shock_correlation, (0.2, 0.8)),
        (shock_correlation, (-1.0, 0.2)),
        (shock_correlation, (1.0, -0.2)),
    ],
)
def test_invalid_scalar_shocks(function, args):
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), -float("inf"), True, None],
)
def test_nonfinite_or_invalid_fx_shock(invalid):
    with pytest.raises(ValueError):
        shock_fx_spot(1.1, invalid)


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), -float("inf"), True, None],
)
def test_nonfinite_or_invalid_volatility_shock(invalid):
    with pytest.raises(ValueError):
        shock_volatility(0.2, invalid)


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), -float("inf"), True, None],
)
def test_nonfinite_or_invalid_correlation_shock(invalid):
    with pytest.raises(ValueError):
        shock_correlation(0.2, invalid)


def test_negative_shocked_hazard_is_rejected():
    with pytest.raises(ValueError):
        shock_hazard_curve(ConstantHazardCurve(0.01), -200.0)


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), -float("inf"), True, None],
)
def test_invalid_curve_shift(discount_curve, invalid):
    with pytest.raises(ValueError):
        shock_discount_curve(discount_curve, invalid)


def test_wrong_object_types_are_rejected(discount_curve):
    with pytest.raises(TypeError):
        shock_smooth_discount_curve(discount_curve, 10.0)
    with pytest.raises(TypeError):
        shock_discount_curve(None, 10.0)
    with pytest.raises(TypeError):
        shock_hazard_curve(discount_curve, 10.0)


def test_hazard_shock_is_not_a_cds_spread_conversion():
    base = ConstantHazardCurve(0.02)
    result = shock_hazard_curve(base, 50.0)
    assert result.hazard_rate == pytest.approx(0.025)


def test_sequential_rate_shocks_are_additive(discount_curve):
    first = shock_discount_curve(discount_curve, 100.0)
    second = shock_discount_curve(first, -40.0)
    direct = shock_discount_curve(discount_curve, 60.0)

    np.testing.assert_allclose(
        second.zero_rates,
        direct.zero_rates,
        atol=1e-15,
    )
