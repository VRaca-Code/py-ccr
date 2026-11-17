"""Tests for endpoint-discretized unilateral DVA."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.xva.cva import calculate_unilateral_cva
from pyccr.xva.dva import DVAResult, calculate_unilateral_dva


def _calculate(
    times=(0.0, 1.0, 2.0, 3.0),
    expected_negative_exposure=(0.0, 100.0, 100.0, 100.0),
    discount_factors=(1.0, 1.0, 1.0, 1.0),
    hazard_rate=0.02,
    recovery_rate=0.4,
):
    return calculate_unilateral_dva(
        times=times,
        expected_negative_exposure=expected_negative_exposure,
        discount_factors=discount_factors,
        own_credit_curve=ConstantHazardCurve(hazard_rate),
        recovery_rate=recovery_rate,
    )


def test_constant_exposure_analytical_benchmark():
    result = _calculate()
    expected = 60.0 * (1.0 - np.exp(-0.06))
    assert isinstance(result, DVAResult)
    np.testing.assert_allclose(result.dva, expected, rtol=1e-13)


def test_interval_contributions_match_formula():
    times = np.array([0.0, 0.5, 1.5, 3.0])
    ene = np.array([0.0, 50.0, 120.0, 80.0])
    discounts = np.array([1.0, 0.99, 0.96, 0.91])
    hazard = 0.035
    recovery = 0.25

    result = _calculate(
        times=times,
        expected_negative_exposure=ene,
        discount_factors=discounts,
        hazard_rate=hazard,
        recovery_rate=recovery,
    )

    survival = np.exp(-hazard * times)
    marginal = survival[:-1] - survival[1:]
    discounted = discounts[1:] * ene[1:]
    contributions = (1.0 - recovery) * discounted * marginal

    np.testing.assert_allclose(
        result.marginal_default_probabilities, marginal, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.discounted_expected_negative_exposures, discounted
    )
    np.testing.assert_allclose(result.contributions, contributions, rtol=1e-13)
    np.testing.assert_allclose(result.dva, contributions.sum(), rtol=1e-13)


@pytest.mark.parametrize(
    ("hazard", "recovery", "exposures"),
    [
        (0.0, 0.4, (0.0, 100.0, 100.0, 100.0)),
        (0.02, 1.0, (0.0, 100.0, 100.0, 100.0)),
        (0.02, 0.4, (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_zero_dva_cases(hazard, recovery, exposures):
    result = _calculate(
        hazard_rate=hazard,
        recovery_rate=recovery,
        expected_negative_exposure=exposures,
    )
    assert result.dva == 0.0
    np.testing.assert_array_equal(result.contributions, np.zeros(3))


def test_zero_recovery_matches_expected_loss():
    result = _calculate(recovery_rate=0.0)
    np.testing.assert_allclose(result.dva, 100.0 * (1.0 - np.exp(-0.06)))


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_exposure_scaling_is_linear(scale):
    baseline = _calculate()
    scaled = _calculate(
        expected_negative_exposure=(0.0, 100 * scale, 100 * scale, 100 * scale)
    )
    np.testing.assert_allclose(scaled.dva, scale * baseline.dva)


def test_lower_recovery_increases_dva():
    assert _calculate(recovery_rate=0.2).dva > _calculate(recovery_rate=0.7).dva


def test_higher_discount_factors_increase_dva():
    lower = _calculate(discount_factors=(1.0, 0.95, 0.90, 0.85))
    higher = _calculate(discount_factors=(1.0, 0.99, 0.98, 0.97))
    assert higher.dva > lower.dva


def test_hazard_monotonicity_for_constant_exposure():
    hazards = [0.0, 0.01, 0.02, 0.05, 0.10]
    results = [_calculate(hazard_rate=h).dva for h in hazards]
    assert np.all(np.diff(results) > 0.0)


def test_two_point_grid():
    result = _calculate(
        times=(0.0, 1.0),
        expected_negative_exposure=(0.0, 100.0),
        discount_factors=(1.0, 0.98),
    )
    assert result.contributions.shape == (1,)
    assert result.marginal_default_probabilities.shape == (1,)
    assert result.discounted_expected_negative_exposures.shape == (1,)


def test_time_zero_exposure_is_excluded():
    baseline = _calculate()
    changed = _calculate(expected_negative_exposure=(100000.0, 100.0, 100.0, 100.0))
    np.testing.assert_allclose(changed.dva, baseline.dva)


def test_result_arrays_do_not_alias_inputs():
    times = np.array([0.0, 1.0, 2.0])
    exposures = np.array([0.0, 100.0, 100.0])
    discounts = np.ones(3)

    result = _calculate(
        times=times,
        expected_negative_exposure=exposures,
        discount_factors=discounts,
    )

    times[1] = 99.0
    exposures[1] = 999.0
    discounts[1] = 0.5

    np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        result.discounted_expected_negative_exposures, [100.0, 100.0]
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
def test_invalid_expected_negative_exposures_rejected(exposures):
    with pytest.raises(ValueError, match="expected_negative_exposure"):
        _calculate(expected_negative_exposure=exposures)


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


def test_invalid_own_credit_curve_type_rejected():
    with pytest.raises(TypeError, match="own_credit_curve"):
        calculate_unilateral_dva(
            times=[0.0, 1.0],
            expected_negative_exposure=[0.0, 100.0],
            discount_factors=[1.0, 0.98],
            own_credit_curve=None,
        )


def test_dva_contributions_are_nonnegative():
    result = _calculate()
    assert result.dva >= 0.0
    assert np.all(result.contributions >= 0.0)
    assert np.all(result.marginal_default_probabilities >= 0.0)


def test_dva_equals_cva_for_identical_positive_exposure_inputs():
    times = np.array([0.0, 0.5, 1.0, 2.0])
    exposure = np.array([0.0, 50.0, 120.0, 80.0])
    discounts = np.exp(-0.03 * times)
    curve = ConstantHazardCurve(0.025)

    dva = calculate_unilateral_dva(
        times, exposure, discounts, curve, recovery_rate=0.35
    )
    cva = calculate_unilateral_cva(
        times, exposure, discounts, curve, recovery_rate=0.35
    )

    np.testing.assert_allclose(dva.dva, cva.cva, rtol=1e-14)
    np.testing.assert_allclose(dva.contributions, cva.contributions)


def test_negative_portfolio_values_feed_dva_via_ene():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array(
        [
            [0.0, -100.0, -200.0],
            [0.0, -300.0, 100.0],
        ]
    )

    profiles = calculate_exposure_profiles(times, values)

    np.testing.assert_allclose(profiles.ene, [0.0, 200.0, 100.0])
    np.testing.assert_allclose(profiles.ee, [0.0, 0.0, 50.0])

    result = calculate_unilateral_dva(
        times=times,
        expected_negative_exposure=profiles.ene,
        discount_factors=np.ones(3),
        own_credit_curve=ConstantHazardCurve(0.02),
    )

    expected = 0.6 * (
        200.0 * (1.0 - np.exp(-0.02)) + 100.0 * (np.exp(-0.02) - np.exp(-0.04))
    )
    np.testing.assert_allclose(result.dva, expected, rtol=1e-13)


def test_positive_only_portfolio_has_zero_dva():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array(
        [
            [10.0, 100.0, 200.0],
            [20.0, 300.0, 400.0],
        ]
    )
    profiles = calculate_exposure_profiles(times, values)

    result = _calculate(
        times=times,
        expected_negative_exposure=profiles.ene,
        discount_factors=np.ones(3),
    )
    assert result.dva == 0.0
