"""Tests for the collateral-aware exposure profile adapter."""

import numpy as np
import pytest

from pyccr.collateral.csa import CSAConfig
from pyccr.collateral.initial_margin import (
    calculate_initial_margin_exposures,
)
from pyccr.collateral.margin import simulate_variation_margin
from pyccr.collateral.mpor import calculate_mpor_exposures
from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.collateral_profiles import (
    CollateralExposureResult,
    calculate_collateral_exposure_profiles,
)
from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.xva.bilateral import calculate_bilateral_xva
from pyccr.xva.cva import calculate_unilateral_cva
from pyccr.xva.dva import calculate_unilateral_dva
from pyccr.xva.first_to_default import calculate_first_to_default_xva


@pytest.fixture
def example():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = np.array(
        [
            [100.0, 115.0, 130.0, 140.0, 150.0],
            [-100.0, -115.0, -130.0, -140.0, -150.0],
        ]
    )
    csa = CSAConfig(
        counterparty_threshold=1e6,
        own_threshold=1e6,
    )
    return times, values, csa


def test_result_type_and_default_time_alignment(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(times, values, csa, mpor_steps=2)
    assert isinstance(result, CollateralExposureResult)
    np.testing.assert_array_equal(result.profiles.times, times[:3])
    np.testing.assert_array_equal(result.closeout_times, times[2:])


def test_asymmetric_initial_margin_exposure(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=20.0,
        posted_im=25.0,
    )
    np.testing.assert_allclose(
        result.profiles.positive_exposures,
        [[110.0, 120.0, 130.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        result.profiles.negative_exposures,
        [[0.0, 0.0, 0.0], [105.0, 115.0, 125.0]],
    )


def test_ee_ene_pfe_epe(example):
    times, values, csa = example
    profiles = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=20.0,
        posted_im=25.0,
    ).profiles
    np.testing.assert_allclose(profiles.ee, [55.0, 60.0, 65.0])
    np.testing.assert_allclose(profiles.ene, [52.5, 57.5, 62.5])
    np.testing.assert_allclose(profiles.pfe, [104.5, 114.0, 123.5])
    assert profiles.epe == pytest.approx(60.0)


def test_zero_collateral_and_zero_mpor_match_baseline(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(times, values, csa)
    baseline = calculate_exposure_profiles(times, values)

    for field in (
        "positive_exposures",
        "negative_exposures",
        "ee",
        "ene",
        "pfe",
    ):
        np.testing.assert_allclose(
            getattr(result.profiles, field),
            getattr(baseline, field),
        )

    assert result.profiles.epe == pytest.approx(baseline.epe)


def test_zero_im_matches_mpor(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(times, values, csa, mpor_steps=2)
    np.testing.assert_array_equal(
        result.profiles.positive_exposures,
        result.mpor_result.positive_exposure,
    )
    np.testing.assert_array_equal(
        result.profiles.negative_exposures,
        result.mpor_result.negative_exposure,
    )


def test_adapter_reuses_existing_collateral_engines(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=1,
        received_im=10.0,
        posted_im=15.0,
    )

    margin = simulate_variation_margin(values, csa)
    mpor = calculate_mpor_exposures(values, margin.vm_balance, mpor_steps=1)
    im = calculate_initial_margin_exposures(mpor, received_im=10.0, posted_im=15.0)

    np.testing.assert_array_equal(result.margin_result.vm_balance, margin.vm_balance)
    np.testing.assert_array_equal(
        result.initial_margin_result.positive_exposure,
        im.positive_exposure,
    )
    np.testing.assert_array_equal(
        result.initial_margin_result.negative_exposure,
        im.negative_exposure,
    )


def test_im_cannot_increase_ee_ene_or_pfe(example):
    times, values, csa = example
    baseline = calculate_collateral_exposure_profiles(
        times, values, csa, mpor_steps=2
    ).profiles
    adjusted = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=20.0,
        posted_im=25.0,
    ).profiles

    assert np.all(adjusted.ee <= baseline.ee)
    assert np.all(adjusted.ene <= baseline.ene)
    assert np.all(adjusted.pfe <= baseline.pfe)


def test_shortened_horizon_is_explicit(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(times, values, csa, mpor_steps=2)
    assert result.profiles.times[-1] == pytest.approx(0.5)
    assert times[-1] == pytest.approx(1.0)
    assert result.profiles.times.size == 3


def test_single_default_time_observation_is_supported(example):
    times, values, csa = example
    result = calculate_collateral_exposure_profiles(times, values, csa, mpor_steps=4)
    assert result.profiles.times.size == 1
    assert result.profiles.epe == pytest.approx(result.profiles.ee[0])


def test_invalid_mpor_is_rejected(example):
    times, values, csa = example
    with pytest.raises((TypeError, ValueError)):
        calculate_collateral_exposure_profiles(times, values, csa, mpor_steps=5)


@pytest.mark.parametrize(
    "invalid_times",
    [
        [0.1, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.25, 0.25, 0.75, 1.0],
        [0.0, 0.25, np.nan, 0.75, 1.0],
    ],
)
def test_invalid_time_grid(example, invalid_times):
    _, values, csa = example
    with pytest.raises(ValueError):
        calculate_collateral_exposure_profiles(invalid_times, values, csa)


@pytest.mark.parametrize(
    "invalid_values",
    [
        np.array([1.0, 2.0, 3.0]),
        np.empty((0, 5)),
        np.full((2, 5), np.nan),
        np.zeros((2, 4)),
    ],
)
def test_invalid_portfolio_values(example, invalid_values):
    times, _, csa = example
    with pytest.raises(ValueError):
        calculate_collateral_exposure_profiles(times, invalid_values, csa)


@pytest.mark.parametrize(
    "confidence_level",
    [-0.1, 0.0, 1.0, 1.1, np.nan],
)
def test_invalid_confidence_level(example, confidence_level):
    times, values, csa = example
    with pytest.raises(ValueError):
        calculate_collateral_exposure_profiles(
            times,
            values,
            csa,
            confidence_level=confidence_level,
        )


def test_invalid_csa_type(example):
    times, values, _ = example
    with pytest.raises(TypeError):
        calculate_collateral_exposure_profiles(times, values, None)


def test_input_arrays_are_not_mutated(example):
    times, values, csa = example
    original_times = times.copy()
    original_values = values.copy()

    calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=20.0,
        posted_im=25.0,
    )

    np.testing.assert_array_equal(times, original_times)
    np.testing.assert_array_equal(values, original_values)


def test_full_grid_initial_margin(example):
    times, values, csa = example
    received = np.full_like(values, 20.0)
    posted = np.full_like(values, 25.0)

    result = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=received,
        posted_im=posted,
    )

    np.testing.assert_allclose(result.profiles.ee, [55.0, 60.0, 65.0])
    np.testing.assert_allclose(result.profiles.ene, [52.5, 57.5, 62.5])


def test_unilateral_and_bilateral_xva_compatibility(example):
    times, values, csa = example
    profiles = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=20.0,
        posted_im=25.0,
    ).profiles

    discount = np.exp(-0.03 * profiles.times)
    cp_curve = ConstantHazardCurve(0.02)
    own_curve = ConstantHazardCurve(0.015)

    cva = calculate_unilateral_cva(profiles.times, profiles.ee, discount, cp_curve)
    dva = calculate_unilateral_dva(profiles.times, profiles.ene, discount, own_curve)
    bilateral = calculate_bilateral_xva(
        profiles.times,
        profiles.ee,
        profiles.ene,
        discount,
        cp_curve,
        own_curve,
    )

    assert bilateral.cva == pytest.approx(cva.cva)
    assert bilateral.dva == pytest.approx(dva.dva)
    assert bilateral.net_xva == pytest.approx(dva.dva - cva.cva)
    assert cva.cva == pytest.approx(0.368870845568)
    assert dva.dva == pytest.approx(0.265922247964)


def test_first_to_default_xva_compatibility(example):
    times, values, csa = example
    profiles = calculate_collateral_exposure_profiles(
        times,
        values,
        csa,
        mpor_steps=2,
        received_im=20.0,
        posted_im=25.0,
    ).profiles

    discount = np.exp(-0.03 * profiles.times)

    result = calculate_first_to_default_xva(
        profiles.times,
        profiles.ee,
        profiles.ene,
        discount,
        ConstantHazardCurve(0.02),
        ConstantHazardCurve(0.015),
    )

    assert result.cva == pytest.approx(0.367468344619)
    assert result.dva == pytest.approx(0.264573624223)
    assert result.net_xva == pytest.approx(result.dva - result.cva)
