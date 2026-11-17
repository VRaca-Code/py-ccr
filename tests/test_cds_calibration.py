"""Tests for flat CDS credit-triangle hazard calibration."""

import numpy as np
import pytest

from pyccr.credit.cds_calibration import (
    calibrate_flat_hazard_from_cds,
    implied_cds_spread_bps,
)
from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.cva import calculate_unilateral_cva


def test_120_bps_at_40_percent_recovery():
    curve = calibrate_flat_hazard_from_cds(120.0, 0.4)
    assert isinstance(curve, ConstantHazardCurve)
    assert curve.hazard_rate == pytest.approx(0.02)


def test_default_recovery_is_40_percent():
    curve = calibrate_flat_hazard_from_cds(120.0)
    assert curve.hazard_rate == pytest.approx(0.02)


@pytest.mark.parametrize(
    "spread_bps,recovery,expected_hazard",
    [
        (0.0, 0.4, 0.0),
        (60.0, 0.4, 0.01),
        (120.0, 0.4, 0.02),
        (300.0, 0.4, 0.05),
        (100.0, 0.0, 0.01),
        (100.0, 0.5, 0.02),
        (100.0, 0.8, 0.05),
        (250.0, 0.75, 0.10),
    ],
)
def test_analytical_credit_triangle(spread_bps, recovery, expected_hazard):
    curve = calibrate_flat_hazard_from_cds(spread_bps, recovery)
    assert curve.hazard_rate == pytest.approx(expected_hazard)


@pytest.mark.parametrize(
    "spread_bps",
    [0.0, 1.0, 25.0, 100.0, 120.0, 500.0, 1500.0],
)
@pytest.mark.parametrize(
    "recovery",
    [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
)
def test_spread_round_trip(spread_bps, recovery):
    curve = calibrate_flat_hazard_from_cds(spread_bps, recovery)
    reconstructed = implied_cds_spread_bps(curve, recovery)
    assert reconstructed == pytest.approx(spread_bps, rel=1e-13, abs=1e-12)


def test_zero_spread_gives_certain_survival():
    curve = calibrate_flat_hazard_from_cds(0.0)
    np.testing.assert_array_equal(
        curve.survival_probability([0.0, 1.0, 5.0]),
        [1.0, 1.0, 1.0],
    )


def test_calibrated_default_probability():
    curve = calibrate_flat_hazard_from_cds(120.0, 0.4)
    expected = 1.0 - np.exp(-0.02 * 3.0)
    assert float(curve.default_probability(3.0)) == (pytest.approx(expected))


def test_higher_spread_increases_hazard():
    low = calibrate_flat_hazard_from_cds(80.0, 0.4)
    high = calibrate_flat_hazard_from_cds(200.0, 0.4)
    assert high.hazard_rate > low.hazard_rate


def test_higher_recovery_increases_implied_hazard():
    low = calibrate_flat_hazard_from_cds(120.0, 0.2)
    high = calibrate_flat_hazard_from_cds(120.0, 0.6)
    assert high.hazard_rate > low.hazard_rate


def test_higher_recovery_reduces_implied_spread():
    curve = ConstantHazardCurve(0.02)
    low = implied_cds_spread_bps(curve, 0.2)
    high = implied_cds_spread_bps(curve, 0.6)
    assert high < low


def test_implied_spread_at_full_recovery_is_zero():
    curve = ConstantHazardCurve(0.02)
    assert implied_cds_spread_bps(curve, 1.0) == 0.0


def test_zero_hazard_has_zero_implied_spread():
    curve = ConstantHazardCurve(0.0)
    assert implied_cds_spread_bps(curve, 0.4) == 0.0


@pytest.mark.parametrize(
    "invalid_spread",
    [
        -0.01,
        -100.0,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_cds_spreads_rejected(invalid_spread):
    with pytest.raises(ValueError, match="cds_spread_bps"):
        calibrate_flat_hazard_from_cds(invalid_spread)


@pytest.mark.parametrize(
    "invalid_recovery",
    [
        -0.01,
        1.0,
        1.01,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_calibration_recovery_rejected(
    invalid_recovery,
):
    with pytest.raises(ValueError, match="recovery_rate"):
        calibrate_flat_hazard_from_cds(120.0, invalid_recovery)


@pytest.mark.parametrize(
    "invalid_recovery",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_implied_spread_recovery_rejected(
    invalid_recovery,
):
    with pytest.raises(ValueError, match="recovery_rate"):
        implied_cds_spread_bps(
            ConstantHazardCurve(0.02),
            invalid_recovery,
        )


def test_implied_spread_rejects_wrong_curve_type():
    with pytest.raises(TypeError, match="credit_curve"):
        implied_cds_spread_bps(None)


def test_calibration_result_is_immutable():
    from dataclasses import FrozenInstanceError

    curve = calibrate_flat_hazard_from_cds(120.0)

    with pytest.raises(FrozenInstanceError):
        curve.hazard_rate = 0.05


def test_calibrated_curve_integrates_with_cva():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    exposures = np.array([0.0, 100.0, 100.0, 100.0])
    discounts = np.ones_like(times)

    calibrated = calibrate_flat_hazard_from_cds(120.0, recovery_rate=0.4)

    result = calculate_unilateral_cva(
        times=times,
        expected_exposure=exposures,
        discount_factors=discounts,
        credit_curve=calibrated,
        recovery_rate=0.4,
    )

    expected = 60.0 * (1.0 - np.exp(-0.06))

    assert result.cva == pytest.approx(expected, rel=1e-13)


def test_calibrated_zero_spread_gives_zero_cva():
    result = calculate_unilateral_cva(
        times=[0.0, 1.0, 2.0],
        expected_exposure=[0.0, 100.0, 100.0],
        discount_factors=[1.0, 0.98, 0.95],
        credit_curve=calibrate_flat_hazard_from_cds(0.0),
    )
    assert result.cva == 0.0
