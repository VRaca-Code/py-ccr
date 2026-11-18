"""Tests for additive standalone CVA/DVA/FVA reporting."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.xva.bilateral import calculate_bilateral_xva
from pyccr.xva.combined import CombinedXVAResult, calculate_combined_xva
from pyccr.xva.fva import calculate_fva


def calculate(**overrides):
    params = {
        "times": [0.0, 0.5, 1.0],
        "expected_exposure": [0.0, 100.0, 100.0],
        "expected_negative_exposure": [0.0, 40.0, 40.0],
        "discount_factors": [1.0, 1.0, 1.0],
        "counterparty_credit_curve": ConstantHazardCurve(0.02),
        "own_credit_curve": ConstantHazardCurve(0.03),
        "borrowing_spread": 0.02,
        "lending_spread": 0.01,
    }
    params.update(overrides)
    return calculate_combined_xva(**params)


def test_result_type_and_shapes():
    result = calculate()
    assert isinstance(result, CombinedXVAResult)
    assert result.times.shape == (3,)
    for values in (
        result.cva_contributions,
        result.dva_contributions,
        result.fca_contributions,
        result.fba_contributions,
        result.net_contributions,
    ):
        assert values.shape == (2,)


def test_manual_component_parity():
    result = calculate()
    bilateral = calculate_bilateral_xva(
        [0.0, 0.5, 1.0],
        [0.0, 100.0, 100.0],
        [0.0, 40.0, 40.0],
        [1.0, 1.0, 1.0],
        ConstantHazardCurve(0.02),
        ConstantHazardCurve(0.03),
    )
    funding = calculate_fva(
        [0.0, 0.5, 1.0],
        [0.0, 100.0, 100.0],
        [0.0, 40.0, 40.0],
        [1.0, 1.0, 1.0],
        0.02,
        0.01,
    )

    assert result.cva == pytest.approx(bilateral.cva)
    assert result.dva == pytest.approx(bilateral.dva)
    assert result.fca == pytest.approx(funding.fca)
    assert result.fba == pytest.approx(funding.fba)
    assert result.fva == pytest.approx(funding.fva)
    assert result.net_xva == pytest.approx(bilateral.net_xva + funding.fva)


def test_funding_analytical_benchmark():
    result = calculate()
    assert result.fca == pytest.approx(2.0)
    assert result.fba == pytest.approx(0.4)
    assert result.fva == pytest.approx(-1.6)


def test_credit_analytical_benchmark():
    result = calculate()
    cp_pd = 1.0 - np.exp(-0.02)
    own_pd = 1.0 - np.exp(-0.03)
    assert result.cva == pytest.approx(0.6 * 100.0 * cp_pd)
    assert result.dva == pytest.approx(0.6 * 40.0 * own_pd)


def test_signed_net_xva_identity():
    result = calculate()
    assert result.net_xva == pytest.approx(
        result.dva - result.cva + result.fba - result.fca
    )


def test_interval_contributions_reconcile():
    result = calculate()
    expected = (
        result.dva_contributions
        - result.cva_contributions
        + result.fba_contributions
        - result.fca_contributions
    )
    np.testing.assert_allclose(result.net_contributions, expected)
    assert result.net_xva == pytest.approx(np.sum(result.net_contributions))


def test_zero_spreads_recover_bilateral_xva():
    result = calculate(
        borrowing_spread=0.0,
        lending_spread=0.0,
    )
    assert result.fva == 0.0
    assert result.net_xva == pytest.approx(result.bilateral_result.net_xva)
    np.testing.assert_allclose(
        result.net_contributions,
        result.bilateral_result.net_contributions,
    )


def test_zero_hazards_recover_fva():
    result = calculate(
        counterparty_credit_curve=ConstantHazardCurve(0.0),
        own_credit_curve=ConstantHazardCurve(0.0),
    )
    assert result.cva == 0.0
    assert result.dva == 0.0
    assert result.net_xva == pytest.approx(result.fva)


def test_zero_exposures_produce_zero_adjustments():
    result = calculate(
        expected_exposure=[0.0, 0.0, 0.0],
        expected_negative_exposure=[0.0, 0.0, 0.0],
    )
    for value in (
        result.cva,
        result.dva,
        result.fca,
        result.fba,
        result.fva,
        result.net_xva,
    ):
        assert value == 0.0


def test_zero_ene_removes_dva_and_fba():
    result = calculate(
        expected_negative_exposure=[0.0, 0.0, 0.0],
    )
    assert result.dva == 0.0
    assert result.fba == 0.0
    assert result.net_xva <= 0.0


def test_zero_ee_removes_cva_and_fca():
    result = calculate(
        expected_exposure=[0.0, 0.0, 0.0],
    )
    assert result.cva == 0.0
    assert result.fca == 0.0
    assert result.net_xva >= 0.0


def test_borrowing_spread_increase_reduces_net_xva():
    base = calculate()
    stressed = calculate(borrowing_spread=0.03)
    assert stressed.net_xva - base.net_xva == pytest.approx(-1.0)
    assert stressed.cva == base.cva
    assert stressed.dva == base.dva


def test_lending_spread_increase_raises_net_xva():
    base = calculate()
    stressed = calculate(lending_spread=0.02)
    assert stressed.net_xva - base.net_xva == pytest.approx(0.4)
    assert stressed.cva == base.cva
    assert stressed.dva == base.dva


def test_credit_hazard_change_leaves_funding_unchanged():
    base = calculate()
    stressed = calculate(
        counterparty_credit_curve=ConstantHazardCurve(0.05),
    )
    assert stressed.fca == base.fca
    assert stressed.fba == base.fba
    assert stressed.cva > base.cva


def test_recovery_changes_credit_only():
    base = calculate()
    changed = calculate(
        counterparty_recovery_rate=0.8,
        own_recovery_rate=0.8,
    )
    assert changed.cva < base.cva
    assert changed.dva < base.dva
    assert changed.fca == base.fca
    assert changed.fba == base.fba


def test_interval_funding_spreads():
    result = calculate(
        borrowing_spread=[0.01, 0.03],
        lending_spread=[0.0, 0.02],
    )
    assert result.fca == pytest.approx(2.0)
    assert result.fba == pytest.approx(0.4)


def test_discount_factors_shared_by_components():
    result = calculate(discount_factors=[1.0, 0.9, 0.8])
    assert result.fca == pytest.approx(1.7)
    assert result.fba == pytest.approx(0.34)
    np.testing.assert_allclose(
        result.fva_result.discounted_expected_exposures,
        [90.0, 80.0],
    )
    np.testing.assert_allclose(
        result.bilateral_result.cva_result.discounted_expected_exposures,
        [90.0, 80.0],
    )


def test_integration_with_exposure_profiles():
    times = np.array([0.0, 0.5, 1.0])
    values = np.array(
        [
            [0.0, 100.0, -20.0],
            [0.0, -40.0, 60.0],
        ]
    )
    profiles = calculate_exposure_profiles(times, values)

    result = calculate_combined_xva(
        profiles.times,
        profiles.ee,
        profiles.ene,
        np.ones(3),
        ConstantHazardCurve(0.02),
        ConstantHazardCurve(0.03),
        borrowing_spread=0.02,
        lending_spread=0.01,
    )
    assert result.fca == pytest.approx(0.8)
    assert result.fba == pytest.approx(0.15)
    assert result.net_xva == pytest.approx(result.bilateral_result.net_xva - 0.65)


@pytest.mark.parametrize(
    "times",
    [
        [0.0],
        [0.1, 1.0, 2.0],
        [0.0, 0.0, 1.0],
        [0.0, np.nan, 1.0],
    ],
)
def test_invalid_times(times):
    with pytest.raises(ValueError, match="times"):
        calculate(times=times)


@pytest.mark.parametrize(
    "exposure",
    [
        [0.0, 100.0],
        [0.0, -1.0, 1.0],
        [0.0, np.nan, 1.0],
    ],
)
def test_invalid_ee(exposure):
    with pytest.raises(ValueError, match="expected_exposure"):
        calculate(expected_exposure=exposure)


@pytest.mark.parametrize(
    "exposure",
    [
        [0.0, 40.0],
        [0.0, -1.0, 1.0],
        [0.0, np.inf, 1.0],
    ],
)
def test_invalid_ene(exposure):
    with pytest.raises(ValueError, match="expected_negative_exposure"):
        calculate(expected_negative_exposure=exposure)


@pytest.mark.parametrize(
    "discounts",
    [
        [1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, np.nan, 1.0],
    ],
)
def test_invalid_discount_factors(discounts):
    with pytest.raises(ValueError, match="discount_factors"):
        calculate(discount_factors=discounts)


def test_invalid_borrowing_spread():
    with pytest.raises(ValueError, match="borrowing_spread"):
        calculate(borrowing_spread=-0.01)


def test_invalid_lending_spread():
    with pytest.raises(ValueError, match="lending_spread"):
        calculate(lending_spread=np.nan)


def test_invalid_credit_curve():
    with pytest.raises(TypeError, match="credit_curve must be"):
        calculate(counterparty_credit_curve="invalid")


def test_invalid_recovery():
    with pytest.raises(ValueError, match="recovery_rate must lie"):
        calculate(counterparty_recovery_rate=1.5)


def test_input_arrays_not_mutated():
    times = np.array([0.0, 0.5, 1.0])
    ee = np.array([0.0, 100.0, 100.0])
    ene = np.array([0.0, 40.0, 40.0])
    df = np.ones(3)
    borrow = np.array([0.02, 0.02])
    lend = np.array([0.01, 0.01])
    arrays = (times, ee, ene, df, borrow, lend)
    originals = [value.copy() for value in arrays]

    calculate_combined_xva(
        times,
        ee,
        ene,
        df,
        ConstantHazardCurve(0.02),
        ConstantHazardCurve(0.03),
        borrowing_spread=borrow,
        lending_spread=lend,
    )

    for actual, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_result_times_do_not_alias_input():
    times = np.array([0.0, 0.5, 1.0])
    result = calculate(times=times)
    times[1] = 0.75
    assert result.times[1] == 0.5


def test_result_is_frozen():
    result = calculate()
    with pytest.raises(FrozenInstanceError):
        result.net_xva = 123.0
