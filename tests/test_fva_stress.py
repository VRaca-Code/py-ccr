"""Tests for deterministic funding-spread stress comparisons."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyccr.xva.fva import calculate_fva
from pyccr.xva.fva_stress import (
    FVAStressComparison,
    compare_fva_spread_stress,
)


def compare(**overrides):
    params = {
        "times": [0.0, 0.5, 1.0],
        "expected_exposure": [0.0, 100.0, 100.0],
        "expected_negative_exposure": [0.0, 40.0, 40.0],
        "discount_factors": [1.0, 1.0, 1.0],
        "borrowing_spread": 0.02,
        "lending_spread": 0.01,
    }
    params.update(overrides)
    return compare_fva_spread_stress(**params)


def test_result_type_and_shapes():
    result = compare()
    assert isinstance(result, FVAStressComparison)
    assert result.delta_contributions.shape == (2,)
    assert result.borrowing_shock_bps.shape == (2,)
    assert result.lending_shock_bps.shape == (2,)


def test_zero_shocks_produce_zero_deltas():
    result = compare()
    assert result.delta_fca == 0.0
    assert result.delta_fba == 0.0
    assert result.delta_fva == 0.0
    np.testing.assert_array_equal(result.delta_contributions, [0.0, 0.0])


def test_borrowing_shock_analytical_benchmark():
    result = compare(borrowing_shock_bps=100.0)
    assert result.base.fca == pytest.approx(2.0)
    assert result.stressed.fca == pytest.approx(3.0)
    assert result.delta_fca == pytest.approx(1.0)
    assert result.delta_fba == 0.0
    assert result.delta_fva == pytest.approx(-1.0)


def test_lending_shock_analytical_benchmark():
    result = compare(lending_shock_bps=100.0)
    assert result.delta_fca == 0.0
    assert result.delta_fba == pytest.approx(0.4)
    assert result.delta_fva == pytest.approx(0.4)


def test_combined_shock_benchmark():
    result = compare(
        borrowing_shock_bps=100.0,
        lending_shock_bps=50.0,
    )
    assert result.delta_fca == pytest.approx(1.0)
    assert result.delta_fba == pytest.approx(0.2)
    assert result.delta_fva == pytest.approx(-0.8)


def test_negative_shocks_can_reduce_spreads():
    result = compare(
        borrowing_shock_bps=-100.0,
        lending_shock_bps=-50.0,
    )
    assert result.stressed.fca == pytest.approx(1.0)
    assert result.stressed.fba == pytest.approx(0.2)
    assert result.delta_fva == pytest.approx(0.8)


def test_shocks_can_reduce_spreads_exactly_to_zero():
    result = compare(
        borrowing_shock_bps=-200.0,
        lending_shock_bps=-100.0,
    )
    assert result.stressed.fca == pytest.approx(0.0)
    assert result.stressed.fba == pytest.approx(0.0)
    assert result.stressed.fva == pytest.approx(0.0)


def test_interval_specific_shocks():
    result = compare(
        borrowing_shock_bps=[100.0, 200.0],
        lending_shock_bps=[0.0, 100.0],
    )
    np.testing.assert_allclose(
        result.stressed.borrowing_spreads,
        [0.03, 0.04],
    )
    np.testing.assert_allclose(
        result.stressed.lending_spreads,
        [0.01, 0.02],
    )
    assert result.delta_fca == pytest.approx(1.5)
    assert result.delta_fba == pytest.approx(0.2)
    assert result.delta_fva == pytest.approx(-1.3)


def test_scalar_and_vector_shocks_equivalent():
    scalar = compare(
        borrowing_shock_bps=100.0,
        lending_shock_bps=50.0,
    )
    vector = compare(
        borrowing_shock_bps=[100.0, 100.0],
        lending_shock_bps=[50.0, 50.0],
    )
    np.testing.assert_allclose(
        scalar.delta_contributions,
        vector.delta_contributions,
    )


def test_manual_repricing_parity():
    result = compare(
        borrowing_shock_bps=100.0,
        lending_shock_bps=50.0,
    )
    manual = calculate_fva(
        [0.0, 0.5, 1.0],
        [0.0, 100.0, 100.0],
        [0.0, 40.0, 40.0],
        [1.0, 1.0, 1.0],
        0.03,
        0.015,
    )
    assert result.stressed.fca == pytest.approx(manual.fca)
    assert result.stressed.fba == pytest.approx(manual.fba)
    assert result.stressed.fva == pytest.approx(manual.fva)


def test_delta_sign_identity():
    result = compare(
        borrowing_shock_bps=100.0,
        lending_shock_bps=50.0,
    )
    assert result.delta_fva == pytest.approx(result.delta_fba - result.delta_fca)


def test_interval_delta_reconciliation():
    result = compare(
        borrowing_shock_bps=[100.0, 200.0],
        lending_shock_bps=[0.0, 100.0],
    )
    np.testing.assert_allclose(
        result.delta_contributions,
        result.stressed.net_contributions - result.base.net_contributions,
    )
    assert result.delta_fva == pytest.approx(np.sum(result.delta_contributions))


def test_discount_factors_are_unchanged():
    result = compare(
        discount_factors=[1.0, 0.95, 0.9],
        borrowing_shock_bps=100.0,
    )
    np.testing.assert_allclose(
        result.base.discounted_expected_exposures,
        result.stressed.discounted_expected_exposures,
    )
    np.testing.assert_allclose(
        result.base.discounted_expected_negative_exposures,
        result.stressed.discounted_expected_negative_exposures,
    )


def test_zero_exposures_eliminate_spread_sensitivity():
    result = compare(
        expected_exposure=[0.0, 0.0, 0.0],
        expected_negative_exposure=[0.0, 0.0, 0.0],
        borrowing_shock_bps=1000.0,
        lending_shock_bps=1000.0,
    )
    assert result.delta_fca == 0.0
    assert result.delta_fba == 0.0
    assert result.delta_fva == 0.0


def test_nonuniform_time_grid():
    result = compare(
        times=[0.0, 0.25, 1.0],
        borrowing_shock_bps=100.0,
    )
    assert result.delta_fca == pytest.approx(1.0)


def test_funding_shocks_do_not_modify_base_spreads():
    result = compare(
        borrowing_shock_bps=100.0,
        lending_shock_bps=100.0,
    )
    np.testing.assert_allclose(
        result.base.borrowing_spreads,
        [0.02, 0.02],
    )
    np.testing.assert_allclose(
        result.base.lending_spreads,
        [0.01, 0.01],
    )


@pytest.mark.parametrize(
    "shock",
    [
        np.nan,
        np.inf,
        -np.inf,
        [100.0],
        [100.0, np.nan],
        [[100.0, 100.0]],
        "invalid",
    ],
)
def test_invalid_borrowing_shocks(shock):
    with pytest.raises(ValueError, match="borrowing_shock_bps"):
        compare(borrowing_shock_bps=shock)


@pytest.mark.parametrize(
    "shock",
    [
        np.nan,
        np.inf,
        -np.inf,
        [100.0],
        [100.0, np.inf],
        [[100.0, 100.0]],
        "invalid",
    ],
)
def test_invalid_lending_shocks(shock):
    with pytest.raises(ValueError, match="lending_shock_bps"):
        compare(lending_shock_bps=shock)


def test_negative_stressed_borrowing_spread_rejected():
    with pytest.raises(ValueError, match="stressed borrowing"):
        compare(borrowing_shock_bps=-201.0)


def test_negative_stressed_lending_spread_rejected():
    with pytest.raises(ValueError, match="stressed lending"):
        compare(lending_shock_bps=-101.0)


def test_invalid_base_spread_rejected():
    with pytest.raises(ValueError, match="borrowing_spread"):
        compare(borrowing_spread=-0.01)


def test_invalid_base_time_grid_rejected():
    with pytest.raises(ValueError, match="times"):
        compare(times=[0.5, 1.0, 1.5])


def test_input_arrays_are_not_modified():
    times = np.array([0.0, 0.5, 1.0])
    ee = np.array([0.0, 100.0, 100.0])
    ene = np.array([0.0, 40.0, 40.0])
    df = np.ones(3)
    borrow = np.array([0.02, 0.02])
    lend = np.array([0.01, 0.01])
    bshock = np.array([100.0, 200.0])
    lshock = np.array([50.0, 100.0])

    arrays = (times, ee, ene, df, borrow, lend, bshock, lshock)
    copies = [array.copy() for array in arrays]

    compare_fva_spread_stress(
        times,
        ee,
        ene,
        df,
        borrowing_spread=borrow,
        lending_spread=lend,
        borrowing_shock_bps=bshock,
        lending_shock_bps=lshock,
    )

    for actual, original in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_result_shock_arrays_do_not_alias_inputs():
    shocks = np.array([100.0, 200.0])
    result = compare(borrowing_shock_bps=shocks)
    shocks[0] = 999.0
    assert result.borrowing_shock_bps[0] == 100.0


def test_result_dataclass_is_frozen():
    result = compare()
    with pytest.raises(FrozenInstanceError):
        result.delta_fva = 999.0
