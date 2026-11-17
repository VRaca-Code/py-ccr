"""Tests for independent first-to-default bilateral XVA."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.cva import calculate_unilateral_cva
from pyccr.xva.dva import calculate_unilateral_dva
from pyccr.xva.first_to_default import (
    FirstToDefaultXVAResult,
    calculate_first_to_default_xva,
)

TIMES = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
EE = np.array([0.0, 100.0, 100.0, 100.0, 100.0])
ENE = np.array([0.0, 50.0, 50.0, 50.0, 50.0])
DF = np.ones_like(TIMES)


def calculate(
    *,
    times=TIMES,
    ee=EE,
    ene=ENE,
    df=DF,
    cp_hazard=0.02,
    own_hazard=0.03,
    cp_recovery=0.4,
    own_recovery=0.4,
):
    return calculate_first_to_default_xva(
        times=times,
        expected_exposure=ee,
        expected_negative_exposure=ene,
        discount_factors=df,
        counterparty_credit_curve=ConstantHazardCurve(cp_hazard),
        own_credit_curve=ConstantHazardCurve(own_hazard),
        counterparty_recovery_rate=cp_recovery,
        own_recovery_rate=own_recovery,
    )


def test_result_type_and_shapes():
    result = calculate()
    assert isinstance(result, FirstToDefaultXVAResult)
    assert result.times.shape == TIMES.shape

    for values in (
        result.counterparty_first_default_probabilities,
        result.own_first_default_probabilities,
        result.discounted_expected_exposures,
        result.discounted_expected_negative_exposures,
        result.cva_contributions,
        result.dva_contributions,
        result.net_contributions,
    ):
        assert values.shape == (TIMES.size - 1,)


def test_analytical_benchmark():
    result = calculate()
    total_probability = 1.0 - np.exp(-0.05 * 3.0)

    expected_cva = 0.6 * 100.0 * 0.02 / 0.05 * total_probability
    expected_dva = 0.6 * 50.0 * 0.03 / 0.05 * total_probability

    assert result.cva == pytest.approx(expected_cva, rel=1e-13)
    assert result.dva == pytest.approx(expected_dva, rel=1e-13)
    assert result.net_xva == pytest.approx(expected_dva - expected_cva, rel=1e-13)


def test_exact_interval_probabilities():
    result = calculate()
    survival = np.exp(-0.05 * TIMES)
    total = survival[:-1] - survival[1:]

    np.testing.assert_allclose(
        result.counterparty_first_default_probabilities,
        0.02 / 0.05 * total,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        result.own_first_default_probabilities,
        0.03 / 0.05 * total,
        rtol=1e-13,
    )


def test_competing_probabilities_sum_to_first_default():
    result = calculate()
    combined = (
        result.counterparty_first_default_probabilities
        + result.own_first_default_probabilities
    )

    np.testing.assert_allclose(
        combined.sum(),
        1.0 - np.exp(-0.05 * TIMES[-1]),
        rtol=1e-13,
    )


def test_interval_contributions_reconcile():
    result = calculate()

    assert result.cva_contributions.sum() == pytest.approx(result.cva)
    assert result.dva_contributions.sum() == pytest.approx(result.dva)
    assert result.net_contributions.sum() == pytest.approx(result.net_xva)

    np.testing.assert_allclose(
        result.net_contributions,
        result.dva_contributions - result.cva_contributions,
    )


def test_discounted_endpoint_exposures():
    df = np.exp(-0.04 * TIMES)
    result = calculate(df=df)

    np.testing.assert_allclose(
        result.discounted_expected_exposures,
        df[1:] * EE[1:],
    )
    np.testing.assert_allclose(
        result.discounted_expected_negative_exposures,
        df[1:] * ENE[1:],
    )


@pytest.mark.parametrize(
    "cp_hazard,own_hazard",
    [
        (0.0, 0.0),
        (0.0, 0.03),
        (0.02, 0.0),
    ],
)
def test_zero_hazard_cases(cp_hazard, own_hazard):
    result = calculate(
        cp_hazard=cp_hazard,
        own_hazard=own_hazard,
    )

    if cp_hazard == 0.0:
        assert result.cva == 0.0
        assert np.all(result.counterparty_first_default_probabilities == 0.0)

    if own_hazard == 0.0:
        assert result.dva == 0.0
        assert np.all(result.own_first_default_probabilities == 0.0)


def test_zero_own_hazard_matches_unilateral_cva():
    result = calculate(own_hazard=0.0)

    unilateral = calculate_unilateral_cva(
        times=TIMES,
        expected_exposure=EE,
        discount_factors=DF,
        credit_curve=ConstantHazardCurve(0.02),
    )

    assert result.cva == pytest.approx(unilateral.cva, rel=1e-13)


def test_zero_counterparty_hazard_matches_unilateral_dva():
    result = calculate(cp_hazard=0.0)

    unilateral = calculate_unilateral_dva(
        times=TIMES,
        expected_negative_exposure=ENE,
        discount_factors=DF,
        own_credit_curve=ConstantHazardCurve(0.03),
    )

    assert result.dva == pytest.approx(unilateral.dva, rel=1e-13)


def test_competing_defaults_reduce_both_adjustments():
    result = calculate()

    cva = calculate_unilateral_cva(TIMES, EE, DF, ConstantHazardCurve(0.02))
    dva = calculate_unilateral_dva(TIMES, ENE, DF, ConstantHazardCurve(0.03))

    assert 0.0 < result.cva < cva.cva
    assert 0.0 < result.dva < dva.dva


@pytest.mark.parametrize("recovery", [0.0, 0.25, 0.4, 0.8, 1.0])
def test_counterparty_recovery_scaling(recovery):
    result = calculate(cp_recovery=recovery)
    base = calculate(cp_recovery=0.0)

    assert result.cva == pytest.approx((1.0 - recovery) * base.cva)


@pytest.mark.parametrize("recovery", [0.0, 0.25, 0.4, 0.8, 1.0])
def test_own_recovery_scaling(recovery):
    result = calculate(own_recovery=recovery)
    base = calculate(own_recovery=0.0)

    assert result.dva == pytest.approx((1.0 - recovery) * base.dva)


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0, 10.0])
def test_positive_exposure_scaling(scale):
    result = calculate(ee=scale * EE)
    base = calculate()

    assert result.cva == pytest.approx(scale * base.cva)
    assert result.dva == pytest.approx(base.dva)


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0, 10.0])
def test_negative_exposure_scaling(scale):
    result = calculate(ene=scale * ENE)
    base = calculate()

    assert result.dva == pytest.approx(scale * base.dva)
    assert result.cva == pytest.approx(base.cva)


def test_symmetric_inputs_cancel():
    equal_exposure = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    result = calculate(
        ee=equal_exposure,
        ene=equal_exposure,
        cp_hazard=0.02,
        own_hazard=0.02,
    )

    assert result.cva == pytest.approx(result.dva)
    assert result.net_xva == pytest.approx(0.0, abs=1e-13)


def test_tiny_hazards_stable():
    result = calculate(cp_hazard=1e-12, own_hazard=2e-12)

    assert np.isfinite(result.cva)
    assert np.isfinite(result.dva)
    assert result.cva > 0.0
    assert result.dva > 0.0

    assert result.cva == pytest.approx(
        0.6 * 100.0 * 1e-12 * TIMES[-1],
        rel=1e-10,
    )


def test_very_large_hazards_finite():
    result = calculate(cp_hazard=500.0, own_hazard=500.0)

    assert np.isfinite(result.cva)
    assert np.isfinite(result.dva)
    assert np.all(result.counterparty_first_default_probabilities >= 0.0)
    assert np.all(result.own_first_default_probabilities >= 0.0)


def test_input_arrays_are_copied():
    times = TIMES.copy()
    result = calculate(times=times)

    times[1] = 99.0
    assert result.times[1] == 0.5


@pytest.mark.parametrize(
    "times",
    [
        [],
        [0.0],
        [[0.0, 1.0]],
        [0.1, 1.0],
        [0.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, np.nan],
        [0.0, np.inf],
    ],
)
def test_invalid_time_grids(times):
    with pytest.raises(ValueError):
        calculate(times=times)


@pytest.mark.parametrize(
    "ee",
    [
        [0.0],
        [[0.0, 100.0]],
        [0.0, -1.0, 1.0, 1.0, 1.0],
        [0.0, np.nan, 1.0, 1.0, 1.0],
        [0.0, np.inf, 1.0, 1.0, 1.0],
    ],
)
def test_invalid_positive_exposures(ee):
    with pytest.raises(ValueError):
        calculate(ee=ee)


@pytest.mark.parametrize(
    "ene",
    [
        [0.0],
        [[0.0, 50.0]],
        [0.0, -1.0, 1.0, 1.0, 1.0],
        [0.0, np.nan, 1.0, 1.0, 1.0],
        [0.0, np.inf, 1.0, 1.0, 1.0],
    ],
)
def test_invalid_negative_exposures(ene):
    with pytest.raises(ValueError):
        calculate(ene=ene)


@pytest.mark.parametrize(
    "df",
    [
        [1.0],
        [[1.0, 1.0]],
        [1.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0, 1.0],
        [1.0, np.nan, 1.0, 1.0, 1.0],
        [1.0, np.inf, 1.0, 1.0, 1.0],
    ],
)
def test_invalid_discount_factors(df):
    with pytest.raises(ValueError):
        calculate(df=df)


@pytest.mark.parametrize(
    "recovery",
    [
        -0.01,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_counterparty_recovery(recovery):
    with pytest.raises(ValueError):
        calculate(cp_recovery=recovery)


@pytest.mark.parametrize(
    "recovery",
    [
        -0.01,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_own_recovery(recovery):
    with pytest.raises(ValueError):
        calculate(own_recovery=recovery)


@pytest.mark.parametrize(
    "field",
    [
        "counterparty_credit_curve",
        "own_credit_curve",
    ],
)
def test_invalid_curve_type(field):
    kwargs = {
        "times": TIMES,
        "expected_exposure": EE,
        "expected_negative_exposure": ENE,
        "discount_factors": DF,
        "counterparty_credit_curve": ConstantHazardCurve(0.02),
        "own_credit_curve": ConstantHazardCurve(0.03),
    }

    kwargs[field] = object()

    with pytest.raises(TypeError):
        calculate_first_to_default_xva(**kwargs)


def test_result_probabilities_are_nonnegative():
    result = calculate()

    assert np.all(result.counterparty_first_default_probabilities >= 0.0)
    assert np.all(result.own_first_default_probabilities >= 0.0)


def test_result_fields_are_independent_of_inputs():
    ee = EE.copy()
    ene = ENE.copy()
    df = DF.copy()

    result = calculate(ee=ee, ene=ene, df=df)

    ee[:] = 0.0
    ene[:] = 0.0
    df[:] = 2.0

    assert result.cva > 0.0
    assert result.dva > 0.0
