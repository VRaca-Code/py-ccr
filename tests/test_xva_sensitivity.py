"""Regression tests for bilateral XVA credit-spread sensitivity."""

import numpy as np
import pytest

from pyccr.credit.cds_calibration import calibrate_flat_hazard_from_cds
from pyccr.xva.bilateral import calculate_bilateral_xva
from pyccr.xva.first_to_default import calculate_first_to_default_xva
from pyccr.xva.sensitivity import (
    BilateralXVASensitivityResult,
    calculate_bilateral_xva_sensitivity,
)

TIMES = np.array([0.0, 1.0, 2.0, 3.0])
EE = np.array([0.0, 100.0, 100.0, 100.0])
ENE = np.array([0.0, 50.0, 50.0, 50.0])
DF = np.ones_like(TIMES)


def calculate(
    *,
    times=TIMES,
    ee=EE,
    ene=ENE,
    df=DF,
    cp_spreads=(0.0, 120.0, 200.0),
    own_spreads=(0.0, 180.0),
    cp_recovery=0.4,
    own_recovery=0.4,
):
    return calculate_bilateral_xva_sensitivity(
        times=times,
        expected_exposure=ee,
        expected_negative_exposure=ene,
        discount_factors=df,
        counterparty_spreads_bps=cp_spreads,
        own_spreads_bps=own_spreads,
        counterparty_recovery_rate=cp_recovery,
        own_recovery_rate=own_recovery,
    )


MATRIX_FIELDS = (
    "naive_cva",
    "naive_dva",
    "naive_net_xva",
    "first_to_default_cva",
    "first_to_default_dva",
    "first_to_default_net_xva",
    "cva_correction",
    "dva_correction",
    "net_xva_correction",
)


def test_result_type_shapes_and_spread_axes():
    result = calculate()

    assert isinstance(result, BilateralXVASensitivityResult)

    np.testing.assert_array_equal(
        result.counterparty_spreads_bps,
        [0.0, 120.0, 200.0],
    )
    np.testing.assert_array_equal(
        result.own_spreads_bps,
        [0.0, 180.0],
    )

    for field in MATRIX_FIELDS:
        matrix = getattr(result, field)
        assert matrix.shape == (2, 3), field
        assert np.all(np.isfinite(matrix)), field


def test_analytical_first_to_default_benchmark():
    result = calculate()

    total_probability = 1.0 - np.exp(-0.05 * 3.0)

    expected_cva = 0.6 * 100.0 * 0.02 / 0.05 * total_probability
    expected_dva = 0.6 * 50.0 * 0.03 / 0.05 * total_probability

    assert result.first_to_default_cva[1, 1] == pytest.approx(
        expected_cva,
        rel=1e-13,
    )
    assert result.first_to_default_dva[1, 1] == pytest.approx(
        expected_dva,
        rel=1e-13,
    )


def test_every_scenario_matches_existing_calculators():
    cp_spreads = [25.0, 120.0, 300.0]
    own_spreads = [50.0, 180.0]

    result = calculate(
        cp_spreads=cp_spreads,
        own_spreads=own_spreads,
    )

    for row, own_spread in enumerate(own_spreads):
        own_curve = calibrate_flat_hazard_from_cds(own_spread)

        for col, cp_spread in enumerate(cp_spreads):
            cp_curve = calibrate_flat_hazard_from_cds(cp_spread)

            naive = calculate_bilateral_xva(
                TIMES,
                EE,
                ENE,
                DF,
                cp_curve,
                own_curve,
            )

            ftd = calculate_first_to_default_xva(
                TIMES,
                EE,
                ENE,
                DF,
                cp_curve,
                own_curve,
            )

            assert result.naive_cva[row, col] == pytest.approx(naive.cva)
            assert result.naive_dva[row, col] == pytest.approx(naive.dva)
            assert result.naive_net_xva[row, col] == pytest.approx(naive.net_xva)

            assert result.first_to_default_cva[row, col] == pytest.approx(ftd.cva)

            assert result.first_to_default_dva[row, col] == pytest.approx(ftd.dva)

            assert result.first_to_default_net_xva[row, col] == pytest.approx(
                ftd.net_xva
            )


def test_zero_spreads_remove_default_adjustments():
    result = calculate(
        cp_spreads=[0.0],
        own_spreads=[0.0],
    )

    for field in MATRIX_FIELDS:
        np.testing.assert_array_equal(
            getattr(result, field),
            np.zeros((1, 1)),
        )


def test_zero_own_hazard_matches_unilateral_cva():
    result = calculate(
        cp_spreads=[120.0],
        own_spreads=[0.0],
    )

    np.testing.assert_allclose(
        result.first_to_default_cva,
        result.naive_cva,
    )
    np.testing.assert_array_equal(
        result.first_to_default_dva,
        [[0.0]],
    )
    np.testing.assert_allclose(
        result.net_xva_correction,
        [[0.0]],
        atol=1e-12,
    )


def test_zero_counterparty_hazard_matches_unilateral_dva():
    result = calculate(
        cp_spreads=[0.0],
        own_spreads=[180.0],
    )

    np.testing.assert_allclose(
        result.first_to_default_dva,
        result.naive_dva,
    )
    np.testing.assert_array_equal(
        result.first_to_default_cva,
        [[0.0]],
    )
    np.testing.assert_allclose(
        result.net_xva_correction,
        [[0.0]],
        atol=1e-12,
    )


def test_asymmetric_recoveries_match_direct_calculators():
    result = calculate(
        cp_spreads=[150.0],
        own_spreads=[200.0],
        cp_recovery=0.25,
        own_recovery=0.55,
    )

    cp_curve = calibrate_flat_hazard_from_cds(150.0, 0.25)
    own_curve = calibrate_flat_hazard_from_cds(200.0, 0.55)

    naive = calculate_bilateral_xva(
        TIMES,
        EE,
        ENE,
        DF,
        cp_curve,
        own_curve,
        counterparty_recovery_rate=0.25,
        own_recovery_rate=0.55,
    )

    ftd = calculate_first_to_default_xva(
        TIMES,
        EE,
        ENE,
        DF,
        cp_curve,
        own_curve,
        counterparty_recovery_rate=0.25,
        own_recovery_rate=0.55,
    )

    assert result.naive_cva[0, 0] == pytest.approx(naive.cva)
    assert result.naive_dva[0, 0] == pytest.approx(naive.dva)
    assert result.first_to_default_cva[0, 0] == pytest.approx(ftd.cva)
    assert result.first_to_default_dva[0, 0] == pytest.approx(ftd.dva)


def test_signed_net_and_correction_identities():
    result = calculate(
        cp_spreads=[25.0, 120.0, 500.0],
        own_spreads=[25.0, 180.0, 500.0],
    )

    np.testing.assert_allclose(
        result.naive_net_xva,
        result.naive_dva - result.naive_cva,
    )
    np.testing.assert_allclose(
        result.first_to_default_net_xva,
        result.first_to_default_dva - result.first_to_default_cva,
    )
    np.testing.assert_allclose(
        result.cva_correction,
        result.first_to_default_cva - result.naive_cva,
    )
    np.testing.assert_allclose(
        result.dva_correction,
        result.first_to_default_dva - result.naive_dva,
    )
    np.testing.assert_allclose(
        result.net_xva_correction,
        result.first_to_default_net_xva - result.naive_net_xva,
    )
    np.testing.assert_allclose(
        result.net_xva_correction,
        result.dva_correction - result.cva_correction,
    )


def test_first_to_default_reduces_both_gross_adjustments():
    result = calculate(
        cp_spreads=[25.0, 120.0, 500.0],
        own_spreads=[25.0, 180.0, 500.0],
    )

    assert np.all(result.first_to_default_cva <= result.naive_cva + 1e-12)
    assert np.all(result.first_to_default_dva <= result.naive_dva + 1e-12)


def test_grid_orientation_and_monotonicity():
    result = calculate(
        cp_spreads=[25.0, 120.0, 500.0],
        own_spreads=[25.0, 180.0, 500.0],
    )

    # With own spread fixed, increasing counterparty credit risk
    # makes the net adjustment more negative.
    assert np.all(np.diff(result.first_to_default_net_xva, axis=1) < 0.0)

    # With counterparty spread fixed, increasing own credit risk
    # makes the net adjustment more positive for this benchmark.
    assert np.all(np.diff(result.first_to_default_net_xva, axis=0) > 0.0)


@pytest.mark.parametrize(
    "bad_spreads",
    [
        [],
        [[25.0, 50.0]],
        -1.0,
        [-1.0],
        [np.nan],
        [np.inf],
        [-np.inf],
    ],
)
@pytest.mark.parametrize(
    "axis",
    ["counterparty", "own"],
)
def test_invalid_spread_grids_rejected(bad_spreads, axis):
    kwargs = (
        {
            "cp_spreads": bad_spreads,
        }
        if axis == "counterparty"
        else {
            "own_spreads": bad_spreads,
        }
    )

    with pytest.raises(ValueError):
        calculate(**kwargs)


@pytest.mark.parametrize(
    "cp_recovery,own_recovery",
    [
        (-0.1, 0.4),
        (1.0, 0.4),
        (np.nan, 0.4),
        (0.4, -0.1),
        (0.4, 1.0),
        (0.4, np.inf),
    ],
)
def test_invalid_recoveries_rejected(cp_recovery, own_recovery):
    with pytest.raises(ValueError):
        calculate(
            cp_recovery=cp_recovery,
            own_recovery=own_recovery,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"times": [0.0, 2.0, 1.0, 3.0]},
        {"ee": [0.0, -1.0, 100.0, 100.0]},
        {"ene": [0.0, 50.0, np.nan, 50.0]},
        {"df": [1.0, 0.99, 0.0, 0.95]},
        {"ee": [0.0, 100.0]},
    ],
)
def test_invalid_exposure_inputs_rejected(override):
    with pytest.raises(ValueError):
        calculate(**override)


def test_input_arrays_are_not_mutated():
    times = TIMES.copy()
    ee = EE.copy()
    ene = ENE.copy()
    df = DF.copy()
    cp_spreads = np.array([25.0, 120.0])
    own_spreads = np.array([50.0, 180.0])

    originals = [
        array.copy()
        for array in (
            times,
            ee,
            ene,
            df,
            cp_spreads,
            own_spreads,
        )
    ]

    calculate(
        times=times,
        ee=ee,
        ene=ene,
        df=df,
        cp_spreads=cp_spreads,
        own_spreads=own_spreads,
    )

    for array, original in zip(
        (times, ee, ene, df, cp_spreads, own_spreads),
        originals,
    ):
        np.testing.assert_array_equal(array, original)


def test_result_spread_axes_are_independent_copies():
    cp_spreads = np.array([25.0, 120.0])
    own_spreads = np.array([50.0, 180.0])

    result = calculate(
        cp_spreads=cp_spreads,
        own_spreads=own_spreads,
    )

    cp_spreads[:] = 999.0
    own_spreads[:] = 999.0

    np.testing.assert_array_equal(
        result.counterparty_spreads_bps,
        [25.0, 120.0],
    )
    np.testing.assert_array_equal(
        result.own_spreads_bps,
        [50.0, 180.0],
    )


def test_single_scenario_grid():
    result = calculate(
        cp_spreads=[120.0],
        own_spreads=[180.0],
    )

    for field in MATRIX_FIELDS:
        assert getattr(result, field).shape == (1, 1)


def test_nonuniform_grid_and_discounting():
    times = np.array([0.0, 0.25, 1.5, 3.0])
    df = np.exp(-0.03 * times)

    result = calculate(
        times=times,
        df=df,
        cp_spreads=[120.0],
        own_spreads=[180.0],
    )

    cp_curve = calibrate_flat_hazard_from_cds(120.0)
    own_curve = calibrate_flat_hazard_from_cds(180.0)

    direct = calculate_first_to_default_xva(
        times,
        EE,
        ENE,
        df,
        cp_curve,
        own_curve,
    )

    assert result.first_to_default_cva[0, 0] == pytest.approx(direct.cva)
    assert result.first_to_default_dva[0, 0] == pytest.approx(direct.dva)
