"""Tests for deterministic combined credit/funding XVA stress."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.xva.combined_stress import compare_combined_xva_stress


def compare(**overrides):
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
    return compare_combined_xva_stress(**params)


def test_zero_shocks_produce_zero_deltas():
    result = compare()
    for delta in (
        result.delta_cva,
        result.delta_dva,
        result.delta_fca,
        result.delta_fba,
        result.delta_fva,
        result.delta_net_xva,
    ):
        assert delta == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        result.delta_net_contributions,
        np.zeros(2),
        atol=1e-12,
    )


def test_counterparty_hazard_shock_changes_only_cva():
    result = compare(counterparty_hazard_shock_bps=100.0)
    assert result.delta_cva > 0.0
    assert result.delta_dva == 0.0
    assert result.delta_fca == 0.0
    assert result.delta_fba == 0.0
    assert result.delta_net_xva == pytest.approx(-result.delta_cva)


def test_own_hazard_shock_changes_only_dva():
    result = compare(own_hazard_shock_bps=100.0)
    assert result.delta_dva > 0.0
    assert result.delta_cva == 0.0
    assert result.delta_fca == 0.0
    assert result.delta_fba == 0.0
    assert result.delta_net_xva == pytest.approx(result.delta_dva)


def test_borrowing_shock_changes_only_fca():
    result = compare(borrowing_shock_bps=100.0)
    assert result.delta_fca == pytest.approx(1.0)
    assert result.delta_cva == 0.0
    assert result.delta_dva == 0.0
    assert result.delta_fba == 0.0
    assert result.delta_net_xva == pytest.approx(-1.0)


def test_lending_shock_changes_only_fba():
    result = compare(lending_shock_bps=100.0)
    assert result.delta_fba == pytest.approx(0.4)
    assert result.delta_cva == 0.0
    assert result.delta_dva == 0.0
    assert result.delta_fca == 0.0
    assert result.delta_net_xva == pytest.approx(0.4)


def test_combined_shocks_reconcile():
    result = compare(
        counterparty_hazard_shock_bps=100.0,
        own_hazard_shock_bps=50.0,
        borrowing_shock_bps=100.0,
        lending_shock_bps=50.0,
    )
    assert result.delta_net_xva == pytest.approx(
        result.delta_dva - result.delta_cva + result.delta_fba - result.delta_fca
    )
    assert result.delta_fva == pytest.approx(result.delta_fba - result.delta_fca)
    assert result.delta_net_xva == pytest.approx(np.sum(result.delta_net_contributions))


def test_interval_specific_funding_shocks():
    result = compare(
        borrowing_shock_bps=[100.0, 200.0],
        lending_shock_bps=[0.0, 100.0],
    )
    assert result.delta_fca == pytest.approx(1.5)
    assert result.delta_fba == pytest.approx(0.2)
    assert result.delta_fva == pytest.approx(-1.3)


def test_negative_hazard_shocks_can_reach_zero():
    result = compare(
        counterparty_hazard_shock_bps=-200.0,
        own_hazard_shock_bps=-300.0,
    )
    assert result.stressed.cva == pytest.approx(0.0)
    assert result.stressed.dva == pytest.approx(0.0)


def test_negative_funding_shocks_can_reach_zero():
    result = compare(
        borrowing_shock_bps=-200.0,
        lending_shock_bps=-100.0,
    )
    assert result.stressed.fca == pytest.approx(0.0)
    assert result.stressed.fba == pytest.approx(0.0)


def test_credit_shock_analytical_benchmark():
    result = compare(counterparty_hazard_shock_bps=100.0)
    expected = 60.0 * ((1.0 - np.exp(-0.03)) - (1.0 - np.exp(-0.02)))
    assert result.delta_cva == pytest.approx(expected)


def test_zero_exposures_eliminate_all_sensitivity():
    result = compare(
        expected_exposure=[0.0, 0.0, 0.0],
        expected_negative_exposure=[0.0, 0.0, 0.0],
        counterparty_hazard_shock_bps=100.0,
        own_hazard_shock_bps=100.0,
        borrowing_shock_bps=100.0,
        lending_shock_bps=100.0,
    )
    assert result.delta_net_xva == 0.0


def test_base_and_stressed_share_grid():
    result = compare(counterparty_hazard_shock_bps=100.0)
    np.testing.assert_array_equal(
        result.base.times,
        result.stressed.times,
    )


@pytest.mark.parametrize(
    "shock",
    [np.nan, np.inf, -np.inf, "invalid", [1.0]],
)
def test_invalid_counterparty_hazard_shocks(shock):
    with pytest.raises(ValueError, match="counterparty_hazard_shock_bps"):
        compare(counterparty_hazard_shock_bps=shock)


def test_boolean_counterparty_hazard_shock_rejected():
    with pytest.raises(TypeError, match="counterparty_hazard_shock_bps"):
        compare(counterparty_hazard_shock_bps=True)


@pytest.mark.parametrize(
    "shock",
    [np.nan, np.inf, -np.inf, "invalid", [1.0]],
)
def test_invalid_own_hazard_shocks(shock):
    with pytest.raises(ValueError, match="own_hazard_shock_bps"):
        compare(own_hazard_shock_bps=shock)


def test_boolean_own_hazard_shock_rejected():
    with pytest.raises(TypeError, match="own_hazard_shock_bps"):
        compare(own_hazard_shock_bps=True)


def test_negative_stressed_counterparty_hazard_rejected():
    with pytest.raises(ValueError, match="counterparty_hazard_shock_bps"):
        compare(counterparty_hazard_shock_bps=-201.0)


def test_negative_stressed_own_hazard_rejected():
    with pytest.raises(ValueError, match="own_hazard_shock_bps"):
        compare(own_hazard_shock_bps=-301.0)


@pytest.mark.parametrize(
    "shock",
    [np.nan, np.inf, "invalid", [1.0], [1.0, np.nan]],
)
def test_invalid_borrowing_shocks(shock):
    with pytest.raises(ValueError, match="borrowing"):
        compare(borrowing_shock_bps=shock)


@pytest.mark.parametrize(
    "shock",
    [np.nan, np.inf, "invalid", [1.0], [1.0, np.nan]],
)
def test_invalid_lending_shocks(shock):
    with pytest.raises(ValueError, match="lending"):
        compare(lending_shock_bps=shock)


def test_negative_stressed_borrowing_spread_rejected():
    with pytest.raises(ValueError, match="borrowing"):
        compare(borrowing_shock_bps=-201.0)


def test_negative_stressed_lending_spread_rejected():
    with pytest.raises(ValueError, match="lending"):
        compare(lending_shock_bps=-101.0)


def test_input_arrays_not_modified():
    times = np.array([0.0, 0.5, 1.0])
    ee = np.array([0.0, 100.0, 100.0])
    ene = np.array([0.0, 40.0, 40.0])
    df = np.ones(3)
    borrow = np.array([0.02, 0.02])
    lend = np.array([0.01, 0.01])
    bshock = np.array([100.0, 200.0])
    lshock = np.array([50.0, 100.0])
    arrays = (times, ee, ene, df, borrow, lend, bshock, lshock)
    originals = [array.copy() for array in arrays]

    compare_combined_xva_stress(
        times,
        ee,
        ene,
        df,
        ConstantHazardCurve(0.02),
        ConstantHazardCurve(0.03),
        borrowing_spread=borrow,
        lending_spread=lend,
        borrowing_shock_bps=bshock,
        lending_shock_bps=lshock,
    )

    for actual, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
