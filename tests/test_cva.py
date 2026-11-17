"""Tests for endpoint-discretized unilateral CVA."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.cva import calculate_unilateral_cva


def _calculate(
    times=(0.0, 1.0, 2.0, 3.0),
    expected_exposure=(0.0, 100.0, 100.0, 100.0),
    discount_factors=(1.0, 1.0, 1.0, 1.0),
    hazard_rate=0.02,
    recovery_rate=0.4,
):
    return calculate_unilateral_cva(
        times=times,
        expected_exposure=expected_exposure,
        discount_factors=discount_factors,
        credit_curve=ConstantHazardCurve(hazard_rate),
        recovery_rate=recovery_rate,
    )


def test_constant_exposure_analytical_benchmark():
    result = _calculate()
    expected = 0.6 * 100.0 * (1.0 - np.exp(-0.02 * 3.0))

    np.testing.assert_allclose(result.cva, expected, rtol=1e-13, atol=1e-13)


def test_interval_contributions_match_formula():
    times = np.array([0.0, 0.5, 1.5, 3.0])
    exposures = np.array([0.0, 50.0, 120.0, 80.0])
    discounts = np.array([1.0, 0.99, 0.96, 0.91])
    hazard = 0.035
    recovery = 0.25

    result = _calculate(
        times=times,
        expected_exposure=exposures,
        discount_factors=discounts,
        hazard_rate=hazard,
        recovery_rate=recovery,
    )

    survival = np.exp(-hazard * times)
    marginal = survival[:-1] - survival[1:]
    discounted_ee = discounts[1:] * exposures[1:]
    expected_contributions = (1.0 - recovery) * discounted_ee * marginal

    np.testing.assert_allclose(
        result.marginal_default_probabilities,
        marginal,
        rtol=1e-13,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.discounted_expected_exposures,
        discounted_ee,
    )
    np.testing.assert_allclose(
        result.contributions,
        expected_contributions,
        rtol=1e-13,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.cva,
        expected_contributions.sum(),
        rtol=1e-13,
        atol=1e-14,
    )


def test_zero_hazard_produces_zero_cva():
    result = _calculate(hazard_rate=0.0)
    assert result.cva == 0.0
    np.testing.assert_array_equal(result.contributions, np.zeros(3))


def test_full_recovery_produces_zero_cva():
    result = _calculate(recovery_rate=1.0)
    assert result.cva == 0.0
    np.testing.assert_array_equal(result.contributions, np.zeros(3))


def test_zero_exposure_produces_zero_cva():
    result = _calculate(expected_exposure=(0.0, 0.0, 0.0, 0.0))
    assert result.cva == 0.0


def test_zero_recovery_matches_expected_loss():
    result = _calculate(recovery_rate=0.0)
    expected = 100.0 * (1.0 - np.exp(-0.06))
    np.testing.assert_allclose(result.cva, expected)


def test_exposure_scaling_is_linear():
    baseline = _calculate()
    doubled = _calculate(expected_exposure=(0.0, 200.0, 200.0, 200.0))
    np.testing.assert_allclose(doubled.cva, 2.0 * baseline.cva)


def test_lower_recovery_increases_cva():
    low_recovery = _calculate(recovery_rate=0.2)
    high_recovery = _calculate(recovery_rate=0.7)
    assert low_recovery.cva > high_recovery.cva


def test_higher_discount_factors_increase_cva():
    lower = _calculate(discount_factors=(1.0, 0.95, 0.90, 0.85))
    higher = _calculate(discount_factors=(1.0, 0.99, 0.98, 0.97))
    assert higher.cva > lower.cva


def test_hazard_monotonicity_for_constant_exposure():
    hazards = [0.0, 0.01, 0.02, 0.05, 0.10]
    results = [_calculate(hazard_rate=hazard).cva for hazard in hazards]
    assert np.all(np.diff(results) > 0.0)


def test_two_point_grid():
    result = _calculate(
        times=(0.0, 1.0),
        expected_exposure=(0.0, 100.0),
        discount_factors=(1.0, 0.98),
    )
    assert result.contributions.shape == (1,)
    assert result.marginal_default_probabilities.shape == (1,)
    assert result.discounted_expected_exposures.shape == (1,)


def test_time_zero_exposure_not_counted():
    baseline = _calculate()
    changed = _calculate(expected_exposure=(100000.0, 100.0, 100.0, 100.0))
    np.testing.assert_allclose(changed.cva, baseline.cva)


def test_result_arrays_are_independent_of_input_arrays():
    times = np.array([0.0, 1.0, 2.0])
    exposures = np.array([0.0, 100.0, 100.0])
    discounts = np.ones(3)

    result = _calculate(
        times=times,
        expected_exposure=exposures,
        discount_factors=discounts,
    )

    times[1] = 99.0
    exposures[1] = 999.0
    discounts[1] = 0.5

    np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        result.discounted_expected_exposures,
        [100.0, 100.0],
    )


@pytest.mark.parametrize(
    "times",
    [
        [],
        [0.0],
        [0.1, 1.0],
        [-1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [[0.0, 1.0], [2.0, 3.0]],
    ],
)
def test_invalid_time_grids_rejected(times):
    with pytest.raises(ValueError, match="times"):
        _calculate(times=times)


@pytest.mark.parametrize(
    "exposures",
    [
        [0.0, 100.0],
        [0.0, -1.0, 100.0, 100.0],
        [0.0, float("nan"), 100.0, 100.0],
        [0.0, float("inf"), 100.0, 100.0],
        [[0.0, 1.0], [2.0, 3.0]],
    ],
)
def test_invalid_expected_exposures_rejected(exposures):
    with pytest.raises(ValueError, match="expected_exposure"):
        _calculate(expected_exposure=exposures)


@pytest.mark.parametrize(
    "discounts",
    [
        [1.0, 0.99],
        [1.0, 0.0, 0.98, 0.97],
        [1.0, -0.1, 0.98, 0.97],
        [1.0, float("nan"), 0.98, 0.97],
        [1.0, float("inf"), 0.98, 0.97],
        [[1.0, 0.99], [0.98, 0.97]],
    ],
)
def test_invalid_discount_factors_rejected(discounts):
    with pytest.raises(ValueError, match="discount_factors"):
        _calculate(discount_factors=discounts)


@pytest.mark.parametrize(
    "recovery",
    [-0.1, 1.1, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_recovery_rates_rejected(recovery):
    with pytest.raises(ValueError, match="recovery_rate"):
        _calculate(recovery_rate=recovery)


def test_invalid_credit_curve_type_rejected():
    with pytest.raises(TypeError, match="credit_curve"):
        calculate_unilateral_cva(
            times=[0.0, 1.0],
            expected_exposure=[0.0, 100.0],
            discount_factors=[1.0, 0.98],
            credit_curve=None,
        )


def test_cva_contributions_are_nonnegative():
    result = _calculate()
    assert result.cva >= 0.0
    assert np.all(result.contributions >= 0.0)
    assert np.all(result.marginal_default_probabilities >= 0.0)
