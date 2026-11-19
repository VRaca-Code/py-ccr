"""Tests for collateral-aware XVA integration."""

from dataclasses import replace

import numpy as np
import pytest

from pyccr.collateral.csa import CSAConfig
from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.collateral_profiles import (
    calculate_collateral_exposure_profiles,
)
from pyccr.xva.bilateral import calculate_bilateral_xva
from pyccr.xva.collateral_xva import (
    CollateralXVAResult,
    calculate_collateral_xva,
)
from pyccr.xva.cva import calculate_unilateral_cva
from pyccr.xva.dva import calculate_unilateral_dva
from pyccr.xva.first_to_default import calculate_first_to_default_xva


@pytest.fixture
def example():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    values = np.array(
        [
            [0.0, 10.0, 14.0, 7.0, 3.0],
            [0.0, -8.0, -12.0, -5.0, -2.0],
            [0.0, 5.0, -3.0, 9.0, 11.0],
        ]
    )

    csa = CSAConfig(
        counterparty_threshold=2.0,
        own_threshold=3.0,
        minimum_transfer_amount=1.0,
        call_frequency_steps=1,
        settlement_lag_steps=0,
    )

    exposure = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=1,
        received_im=1.5,
        posted_im=2.0,
    )

    cp_curve = ConstantHazardCurve(0.03)
    own_curve = ConstantHazardCurve(0.02)

    df = np.exp(-0.04 * exposure.profiles.times)

    return exposure, df, cp_curve, own_curve


def test_result_type_and_default_grid(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(exposure, df, cp, own)

    assert isinstance(result, CollateralXVAResult)
    np.testing.assert_array_equal(result.times, exposure.profiles.times)
    np.testing.assert_array_equal(result.discount_factors, df)
    assert result.times.size == 4


def test_matches_existing_unilateral_calculators(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(exposure, df, cp, own)
    p = exposure.profiles

    expected_cva = calculate_unilateral_cva(p.times, p.ee, df, cp)
    expected_dva = calculate_unilateral_dva(p.times, p.ene, df, own)

    assert result.cva_result.cva == expected_cva.cva
    assert result.dva_result.dva == expected_dva.dva

    np.testing.assert_array_equal(
        result.cva_result.contributions,
        expected_cva.contributions,
    )
    np.testing.assert_array_equal(
        result.dva_result.contributions,
        expected_dva.contributions,
    )


def test_matches_existing_bilateral_calculator(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(exposure, df, cp, own)
    p = exposure.profiles

    expected = calculate_bilateral_xva(p.times, p.ee, p.ene, df, cp, own)

    assert result.bilateral_result.cva == expected.cva
    assert result.bilateral_result.dva == expected.dva
    assert result.bilateral_result.net_xva == expected.net_xva

    np.testing.assert_array_equal(
        result.bilateral_result.net_contributions,
        expected.net_contributions,
    )


def test_matches_existing_first_to_default_calculator(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(exposure, df, cp, own)
    p = exposure.profiles

    expected = calculate_first_to_default_xva(p.times, p.ee, p.ene, df, cp, own)

    assert result.first_to_default_result.cva == expected.cva
    assert result.first_to_default_result.dva == expected.dva
    assert result.first_to_default_result.net_xva == expected.net_xva

    np.testing.assert_array_equal(
        result.first_to_default_result.net_contributions,
        expected.net_contributions,
    )


def test_mpor_shortens_default_horizon(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(exposure, df, cp, own)

    assert result.times[-1] == 0.75
    assert exposure.closeout_times[-1] == 1.0


def test_full_simulation_grid_discounts_are_rejected(example):
    exposure, _, cp, own = example

    full_grid_df = np.exp(-0.04 * np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    with pytest.raises(ValueError, match="default-time grid"):
        calculate_collateral_xva(exposure, full_grid_df, cp, own)


def test_single_default_observation_rejected():
    times = np.array([0.0, 1.0])
    values = np.array([[0.0, 10.0]])

    exposure = calculate_collateral_exposure_profiles(
        times,
        values,
        CSAConfig(),
        mpor_steps=1,
    )

    with pytest.raises(ValueError, match="at least two"):
        calculate_collateral_xva(
            exposure,
            np.array([1.0]),
            ConstantHazardCurve(0.02),
            ConstantHazardCurve(0.01),
        )


@pytest.mark.parametrize(
    "bad_df",
    [
        [1.0],
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [1.0, 0.9, 0.8, 0.0],
        [1.0, 0.9, 0.8, -0.1],
        [1.0, 0.9, 0.8, np.nan],
        [1.0, 0.9, 0.8, np.inf],
        [[1.0, 0.9, 0.8, 0.7]],
    ],
)
def test_invalid_discount_factors_rejected(example, bad_df):
    exposure, _, cp, own = example

    with pytest.raises(ValueError):
        calculate_collateral_xva(exposure, bad_df, cp, own)


def test_zero_hazard_produces_zero_xva(example):
    exposure, df, _, _ = example

    zero = ConstantHazardCurve(0.0)

    result = calculate_collateral_xva(exposure, df, zero, zero)

    assert result.cva_result.cva == 0.0
    assert result.dva_result.dva == 0.0
    assert result.bilateral_result.net_xva == 0.0
    assert result.first_to_default_result.net_xva == 0.0


def test_full_recovery_produces_zero_xva(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(
        exposure,
        df,
        cp,
        own,
        counterparty_recovery_rate=1.0,
        own_recovery_rate=1.0,
    )

    assert result.cva_result.cva == 0.0
    assert result.dva_result.dva == 0.0
    assert result.bilateral_result.net_xva == 0.0
    assert result.first_to_default_result.net_xva == 0.0


def test_asymmetric_im_reduces_corresponding_exposure():
    times = np.array([0.0, 0.5, 1.0])
    values = np.array(
        [
            [0.0, 10.0, 10.0],
            [0.0, -10.0, -10.0],
        ]
    )

    csa = CSAConfig(
        counterparty_threshold=100.0,
        own_threshold=100.0,
    )

    baseline = calculate_collateral_exposure_profiles(times, values, csa)

    with_received_im = calculate_collateral_exposure_profiles(
        times, values, csa, received_im=4.0
    )

    with_posted_im = calculate_collateral_exposure_profiles(
        times, values, csa, posted_im=4.0
    )

    cp = ConstantHazardCurve(0.03)
    own = ConstantHazardCurve(0.02)
    df = np.ones_like(times)

    base = calculate_collateral_xva(baseline, df, cp, own)
    received = calculate_collateral_xva(with_received_im, df, cp, own)
    posted = calculate_collateral_xva(with_posted_im, df, cp, own)

    assert received.cva_result.cva < base.cva_result.cva
    assert received.dva_result.dva == base.dva_result.dva

    assert posted.dva_result.dva < base.dva_result.dva
    assert posted.cva_result.cva == base.cva_result.cva


def test_result_arrays_do_not_alias_inputs(example):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(exposure, df, cp, own)

    original_times = result.times.copy()
    original_df = result.discount_factors.copy()

    df[:] = 0.5
    exposure.profiles.times[:] = 99.0

    np.testing.assert_array_equal(result.times, original_times)
    np.testing.assert_array_equal(result.discount_factors, original_df)


def test_invalid_exposure_type_rejected(example):
    _, df, cp, own = example

    with pytest.raises(TypeError):
        calculate_collateral_xva(object(), df, cp, own)


def test_invalid_closeout_alignment_rejected(example):
    exposure, df, cp, own = example

    bad = replace(
        exposure,
        closeout_times=np.array([0.0, 0.25]),
    )

    with pytest.raises(ValueError, match="closeout times"):
        calculate_collateral_xva(bad, df, cp, own)


def test_closeout_preceding_default_rejected(example):
    exposure, df, cp, own = example

    bad = replace(
        exposure,
        closeout_times=exposure.profiles.times - 0.1,
    )

    with pytest.raises(ValueError, match="cannot precede"):
        calculate_collateral_xva(bad, df, cp, own)


@pytest.mark.parametrize(
    "cp_recovery,own_recovery",
    [
        (0.0, 0.0),
        (0.2, 0.8),
        (0.4, 0.4),
        (1.0, 1.0),
    ],
)
def test_distinct_recovery_rates(example, cp_recovery, own_recovery):
    exposure, df, cp, own = example

    result = calculate_collateral_xva(
        exposure,
        df,
        cp,
        own,
        counterparty_recovery_rate=cp_recovery,
        own_recovery_rate=own_recovery,
    )

    p = exposure.profiles

    expected = calculate_bilateral_xva(
        p.times,
        p.ee,
        p.ene,
        df,
        cp,
        own,
        counterparty_recovery_rate=cp_recovery,
        own_recovery_rate=own_recovery,
    )

    assert result.bilateral_result.cva == expected.cva
    assert result.bilateral_result.dva == expected.dva


def test_negative_exposure_cannot_increase_from_received_im():
    times = np.array([0.0, 0.5, 1.0])
    values = np.array(
        [
            [0.0, -10.0, -10.0],
            [0.0, 10.0, 10.0],
        ]
    )

    csa = CSAConfig(
        counterparty_threshold=100.0,
        own_threshold=100.0,
    )

    a = calculate_collateral_exposure_profiles(times, values, csa)

    b = calculate_collateral_exposure_profiles(times, values, csa, received_im=3.0)

    cp = ConstantHazardCurve(0.02)
    own = ConstantHazardCurve(0.02)
    df = np.ones_like(times)

    a_xva = calculate_collateral_xva(a, df, cp, own)
    b_xva = calculate_collateral_xva(b, df, cp, own)

    assert b_xva.dva_result.dva == a_xva.dva_result.dva
    assert b_xva.cva_result.cva <= a_xva.cva_result.cva
