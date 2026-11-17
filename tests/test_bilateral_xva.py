"""Tests for standalone unilateral CVA/DVA aggregation."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.xva.bilateral import (
    BilateralXVAResult,
    calculate_bilateral_xva,
)
from pyccr.xva.cva import calculate_unilateral_cva
from pyccr.xva.dva import calculate_unilateral_dva


def _calculate(
    times=(0.0, 1.0, 2.0, 3.0),
    ee=(0.0, 100.0, 100.0, 100.0),
    ene=(0.0, 50.0, 50.0, 50.0),
    discounts=(1.0, 1.0, 1.0, 1.0),
    counterparty_hazard=0.02,
    own_hazard=0.03,
    counterparty_recovery=0.4,
    own_recovery=0.4,
):
    return calculate_bilateral_xva(
        times=times,
        expected_exposure=ee,
        expected_negative_exposure=ene,
        discount_factors=discounts,
        counterparty_credit_curve=ConstantHazardCurve(counterparty_hazard),
        own_credit_curve=ConstantHazardCurve(own_hazard),
        counterparty_recovery_rate=counterparty_recovery,
        own_recovery_rate=own_recovery,
    )


def test_analytical_benchmark():
    result = _calculate()

    expected_cva = 60.0 * (1.0 - np.exp(-0.06))
    expected_dva = 30.0 * (1.0 - np.exp(-0.09))

    assert isinstance(result, BilateralXVAResult)

    np.testing.assert_allclose(result.cva, expected_cva)
    np.testing.assert_allclose(result.dva, expected_dva)
    np.testing.assert_allclose(result.net_xva, expected_dva - expected_cva)


def test_default_benchmark_has_negative_net_xva():
    result = _calculate()
    assert result.cva > result.dva
    assert result.net_xva < 0.0


def test_positive_net_xva_when_dva_exceeds_cva():
    result = _calculate(
        ee=(0.0, 10.0, 10.0, 10.0),
        ene=(0.0, 200.0, 200.0, 200.0),
    )
    assert result.dva > result.cva
    assert result.net_xva > 0.0


def test_symmetric_exposures_and_credit_risk_cancel():
    result = _calculate(
        ene=(0.0, 100.0, 100.0, 100.0),
        own_hazard=0.02,
    )
    np.testing.assert_allclose(result.cva, result.dva)
    np.testing.assert_allclose(result.net_xva, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.net_contributions, np.zeros(3), atol=1e-14)


def test_individual_results_match_unilateral_calculators():
    times = np.array([0.0, 0.5, 1.5, 3.0])
    ee = np.array([0.0, 40.0, 120.0, 90.0])
    ene = np.array([0.0, 100.0, 30.0, 150.0])
    discounts = np.exp(-0.025 * times)

    cp_curve = ConstantHazardCurve(0.04)
    own_curve = ConstantHazardCurve(0.015)

    result = calculate_bilateral_xva(
        times,
        ee,
        ene,
        discounts,
        cp_curve,
        own_curve,
        counterparty_recovery_rate=0.35,
        own_recovery_rate=0.5,
    )

    cva = calculate_unilateral_cva(times, ee, discounts, cp_curve, recovery_rate=0.35)
    dva = calculate_unilateral_dva(times, ene, discounts, own_curve, recovery_rate=0.5)

    np.testing.assert_allclose(result.cva, cva.cva)
    np.testing.assert_allclose(result.dva, dva.dva)
    np.testing.assert_allclose(result.cva_contributions, cva.contributions)
    np.testing.assert_allclose(result.dva_contributions, dva.contributions)
    np.testing.assert_allclose(result.cva_result.contributions, cva.contributions)
    np.testing.assert_allclose(result.dva_result.contributions, dva.contributions)


def test_net_contributions_are_signed_difference():
    result = _calculate()

    np.testing.assert_allclose(
        result.net_contributions,
        result.dva_contributions - result.cva_contributions,
    )
    np.testing.assert_allclose(
        result.net_contributions.sum(),
        result.net_xva,
        atol=1e-14,
    )


def test_result_time_grid_and_interval_shapes():
    result = _calculate()

    np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0, 3.0])

    for contributions in (
        result.cva_contributions,
        result.dva_contributions,
        result.net_contributions,
    ):
        assert contributions.shape == (3,)


@pytest.mark.parametrize(
    ("counterparty_hazard", "own_hazard"),
    [
        (0.0, 0.0),
        (0.0, 0.03),
        (0.02, 0.0),
    ],
)
def test_zero_hazard_cases(counterparty_hazard, own_hazard):
    result = _calculate(
        counterparty_hazard=counterparty_hazard,
        own_hazard=own_hazard,
    )

    if counterparty_hazard == 0.0:
        assert result.cva == 0.0

    if own_hazard == 0.0:
        assert result.dva == 0.0

    np.testing.assert_allclose(result.net_xva, result.dva - result.cva)


def test_full_counterparty_recovery_removes_cva():
    result = _calculate(counterparty_recovery=1.0)
    assert result.cva == 0.0
    np.testing.assert_allclose(result.net_xva, result.dva)


def test_full_own_recovery_removes_dva():
    result = _calculate(own_recovery=1.0)
    assert result.dva == 0.0
    np.testing.assert_allclose(result.net_xva, -result.cva)


def test_distinct_recovery_rates():
    result = _calculate(
        counterparty_recovery=0.2,
        own_recovery=0.8,
    )
    expected_cva = 80.0 * (1.0 - np.exp(-0.06))
    expected_dva = 10.0 * (1.0 - np.exp(-0.09))

    np.testing.assert_allclose(result.cva, expected_cva)
    np.testing.assert_allclose(result.dva, expected_dva)


def test_counterparty_hazard_changes_only_cva():
    low = _calculate(counterparty_hazard=0.01)
    high = _calculate(counterparty_hazard=0.04)

    assert high.cva > low.cva
    np.testing.assert_allclose(high.dva, low.dva)
    assert high.net_xva < low.net_xva


def test_own_hazard_changes_only_dva():
    low = _calculate(own_hazard=0.01)
    high = _calculate(own_hazard=0.04)

    assert high.dva > low.dva
    np.testing.assert_allclose(high.cva, low.cva)
    assert high.net_xva > low.net_xva


def test_counterparty_recovery_changes_only_cva():
    low = _calculate(counterparty_recovery=0.2)
    high = _calculate(counterparty_recovery=0.8)

    assert low.cva > high.cva
    np.testing.assert_allclose(low.dva, high.dva)


def test_own_recovery_changes_only_dva():
    low = _calculate(own_recovery=0.2)
    high = _calculate(own_recovery=0.8)

    assert low.dva > high.dva
    np.testing.assert_allclose(low.cva, high.cva)


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_scaling_both_exposures_scales_all_adjustments(scale):
    baseline = _calculate()

    scaled = _calculate(
        ee=(0.0, 100 * scale, 100 * scale, 100 * scale),
        ene=(0.0, 50 * scale, 50 * scale, 50 * scale),
    )

    np.testing.assert_allclose(scaled.cva, scale * baseline.cva)
    np.testing.assert_allclose(scaled.dva, scale * baseline.dva)
    np.testing.assert_allclose(scaled.net_xva, scale * baseline.net_xva)


def test_two_point_grid():
    result = _calculate(
        times=(0.0, 1.0),
        ee=(0.0, 100.0),
        ene=(0.0, 50.0),
        discounts=(1.0, 0.98),
    )

    assert result.net_contributions.shape == (1,)
    np.testing.assert_allclose(result.net_contributions[0], result.net_xva)


def test_time_zero_exposures_do_not_affect_xva():
    baseline = _calculate()

    changed = _calculate(
        ee=(100000.0, 100.0, 100.0, 100.0),
        ene=(200000.0, 50.0, 50.0, 50.0),
    )

    np.testing.assert_allclose(changed.net_xva, baseline.net_xva)


def test_result_times_do_not_alias_input():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    result = _calculate(times=times)

    times[1] = 99.0

    np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0, 3.0])


def test_integration_with_exposure_profiles():
    times = np.array([0.0, 1.0, 2.0])

    portfolio_values = np.array(
        [
            [0.0, 100.0, -200.0],
            [0.0, -300.0, 400.0],
        ]
    )

    profiles = calculate_exposure_profiles(times, portfolio_values)

    np.testing.assert_allclose(profiles.ee, [0.0, 50.0, 200.0])
    np.testing.assert_allclose(profiles.ene, [0.0, 150.0, 100.0])

    result = calculate_bilateral_xva(
        times=times,
        expected_exposure=profiles.ee,
        expected_negative_exposure=profiles.ene,
        discount_factors=np.ones(3),
        counterparty_credit_curve=ConstantHazardCurve(0.02),
        own_credit_curve=ConstantHazardCurve(0.03),
    )

    assert result.cva > 0.0
    assert result.dva > 0.0

    np.testing.assert_allclose(result.net_xva, result.dva - result.cva)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [0.0],
        [0.1, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, float("nan")],
        [0.0, float("inf")],
    ],
)
def test_invalid_times_rejected(times):
    with pytest.raises(ValueError, match="times"):
        _calculate(times=times)


@pytest.mark.parametrize(
    "ee",
    [
        [0.0, 100.0],
        [0.0, -1.0, 100.0, 100.0],
        [0.0, float("nan"), 100.0, 100.0],
        [0.0, float("inf"), 100.0, 100.0],
    ],
)
def test_invalid_ee_rejected(ee):
    with pytest.raises(ValueError, match="expected_exposure"):
        _calculate(ee=ee)


@pytest.mark.parametrize(
    "ene",
    [
        [0.0, 50.0],
        [0.0, -1.0, 50.0, 50.0],
        [0.0, float("nan"), 50.0, 50.0],
        [0.0, float("inf"), 50.0, 50.0],
    ],
)
def test_invalid_ene_rejected(ene):
    with pytest.raises(ValueError, match="expected_negative_exposure"):
        _calculate(ene=ene)


@pytest.mark.parametrize(
    "discounts",
    [
        [1.0, 0.99],
        [1.0, 0.0, 0.98, 0.97],
        [1.0, -0.1, 0.98, 0.97],
        [1.0, float("nan"), 0.98, 0.97],
        [1.0, float("inf"), 0.98, 0.97],
    ],
)
def test_invalid_discounts_rejected(discounts):
    with pytest.raises(ValueError, match="discount_factors"):
        _calculate(discounts=discounts)


@pytest.mark.parametrize(
    "recovery",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_counterparty_recovery_rejected(recovery):
    with pytest.raises(ValueError, match="recovery_rate"):
        _calculate(counterparty_recovery=recovery)


@pytest.mark.parametrize(
    "recovery",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_own_recovery_rejected(recovery):
    with pytest.raises(ValueError, match="recovery_rate"):
        _calculate(own_recovery=recovery)


def test_invalid_counterparty_credit_curve_rejected():
    with pytest.raises(TypeError, match="credit_curve"):
        calculate_bilateral_xva(
            times=[0.0, 1.0],
            expected_exposure=[0.0, 100.0],
            expected_negative_exposure=[0.0, 50.0],
            discount_factors=[1.0, 1.0],
            counterparty_credit_curve=None,
            own_credit_curve=ConstantHazardCurve(0.03),
        )


def test_invalid_own_credit_curve_rejected():
    with pytest.raises(TypeError, match="own_credit_curve"):
        calculate_bilateral_xva(
            times=[0.0, 1.0],
            expected_exposure=[0.0, 100.0],
            expected_negative_exposure=[0.0, 50.0],
            discount_factors=[1.0, 1.0],
            counterparty_credit_curve=ConstantHazardCurve(0.02),
            own_credit_curve=None,
        )


def test_cva_and_dva_contributions_nonnegative():
    result = _calculate()

    assert np.all(result.cva_contributions >= 0.0)
    assert np.all(result.dva_contributions >= 0.0)


def test_net_contributions_may_be_negative():
    result = _calculate()
    assert np.any(result.net_contributions < 0.0)


def test_all_zero_exposures_produce_zero_adjustments():
    result = _calculate(
        ee=(0.0, 0.0, 0.0, 0.0),
        ene=(0.0, 0.0, 0.0, 0.0),
    )

    assert result.cva == 0.0
    assert result.dva == 0.0
    assert result.net_xva == 0.0
