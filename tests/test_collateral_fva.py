"""Regression tests for VM-only collateral-aware FVA."""

from dataclasses import replace

import numpy as np
import pytest

from pyccr.collateral.csa import CSAConfig
from pyccr.collateral.margin import simulate_variation_margin
from pyccr.xva.collateral_fva import calculate_collateral_fva
from pyccr.xva.fva import FVAResult, calculate_fva


@pytest.fixture
def example():
    times = np.array([0.0, 0.25, 0.75, 1.0])
    values = np.array(
        [
            [0.0, 100.0, 120.0, 80.0],
            [0.0, -40.0, -60.0, -20.0],
            [0.0, 70.0, -30.0, 50.0],
        ]
    )
    df = np.exp(-0.03 * times)
    return times, values, df


def margin(values, **kwargs):
    return simulate_variation_margin(
        values,
        CSAConfig(**kwargs),
    )


def evaluate(example, margin_result, borrow=0.02, lend=0.005):
    times, values, df = example
    return calculate_collateral_fva(
        times,
        values,
        margin_result,
        df,
        borrowing_spread=borrow,
        lending_spread=lend,
    )


def test_returns_existing_fva_result(example):
    _, values, _ = example
    result = evaluate(
        example,
        margin(values, counterparty_threshold=1e6, own_threshold=1e6),
    )
    assert isinstance(result, FVAResult)
    assert result.fva == pytest.approx(result.fba - result.fca)


def test_uncollateralized_parity_with_existing_engine(example):
    times, values, df = example
    vm = margin(
        values,
        counterparty_threshold=1e6,
        own_threshold=1e6,
    )
    actual = evaluate(example, vm)

    ee = np.maximum(values, 0.0).mean(axis=0)
    ene = np.maximum(-values, 0.0).mean(axis=0)
    expected = calculate_fva(times, ee, ene, df, 0.02, 0.005)

    assert actual.fca == pytest.approx(expected.fca)
    assert actual.fba == pytest.approx(expected.fba)
    assert actual.fva == pytest.approx(expected.fva)
    np.testing.assert_allclose(
        actual.net_contributions,
        expected.net_contributions,
    )


def test_full_immediate_vm_zero_funding(example):
    _, values, _ = example
    result = evaluate(example, margin(values))

    assert result.fca == pytest.approx(0.0)
    assert result.fba == pytest.approx(0.0)
    assert result.fva == pytest.approx(0.0)


def test_zero_spreads_zero_fva(example):
    _, values, _ = example
    result = evaluate(
        example,
        margin(values, settlement_lag_steps=1),
        borrow=0.0,
        lend=0.0,
    )
    assert result.fca == 0.0
    assert result.fba == 0.0
    assert result.fva == 0.0


def test_settlement_lag_creates_funding_exposure(example):
    _, values, _ = example

    immediate = evaluate(example, margin(values))
    delayed = evaluate(
        example,
        margin(values, settlement_lag_steps=1),
    )

    assert immediate.fca == pytest.approx(0.0)
    assert delayed.fca > 0.0
    assert delayed.fva < 0.0


def test_pending_vm_not_treated_as_settled(example):
    times, values, df = example
    delayed = margin(values, settlement_lag_steps=1)

    assert np.any(delayed.pending_transfers != 0.0)

    actual = evaluate(example, delayed)

    residual = values - delayed.vm_balance
    expected = calculate_fva(
        times,
        np.maximum(residual, 0.0).mean(axis=0),
        np.maximum(-residual, 0.0).mean(axis=0),
        df,
        0.02,
        0.005,
    )

    assert actual.fva == pytest.approx(expected.fva)


def test_thresholds_create_residual_exposure(example):
    _, values, _ = example
    result = evaluate(
        example,
        margin(
            values,
            counterparty_threshold=25.0,
            own_threshold=10.0,
        ),
    )
    assert result.fca > 0.0
    assert result.fba > 0.0


def test_borrowing_only_negative_fva(example):
    _, values, _ = example
    result = evaluate(
        example,
        margin(values, settlement_lag_steps=1),
        borrow=0.03,
        lend=0.0,
    )
    assert result.fca > 0.0
    assert result.fba == 0.0
    assert result.fva < 0.0


def test_lending_only_positive_fva(example):
    _, values, _ = example
    result = evaluate(
        example,
        margin(values, settlement_lag_steps=1),
        borrow=0.0,
        lend=0.01,
    )
    assert result.fca == 0.0
    assert result.fba > 0.0
    assert result.fva > 0.0


def test_interval_spreads_and_nonuniform_times(example):
    times, values, df = example
    delayed = margin(values, settlement_lag_steps=1)
    borrow = np.array([0.01, 0.02, 0.03])
    lend = np.array([0.002, 0.004, 0.006])

    actual = evaluate(example, delayed, borrow=borrow, lend=lend)

    residual = values - delayed.vm_balance
    ee = np.maximum(residual, 0.0).mean(axis=0)

    expected_fca = np.sum(df[1:] * ee[1:] * borrow * np.diff(times))

    assert actual.fca == pytest.approx(expected_fca)


def test_overcollateralization_changes_funding_sign(example):
    times, values, df = example
    base = margin(values)
    over = replace(
        base,
        vm_balance=values + 10.0,
    )

    result = calculate_collateral_fva(
        times,
        values,
        over,
        df,
        borrowing_spread=0.02,
        lending_spread=0.005,
    )

    assert result.fca == pytest.approx(0.0)
    assert result.fba > 0.0


@pytest.mark.parametrize(
    "bad_values",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 3)),
        np.empty((2, 0)),
        np.array([[0.0, np.nan]]),
        np.array([[0.0, np.inf]]),
    ],
)
def test_invalid_portfolio_values(bad_values):
    times = np.array([0.0, 1.0])
    valid = np.zeros((1, 2))
    vm = margin(valid)

    with pytest.raises(ValueError):
        calculate_collateral_fva(
            times,
            bad_values,
            vm,
            np.ones(2),
            borrowing_spread=0.02,
        )


def test_invalid_margin_result_type(example):
    times, values, df = example
    with pytest.raises(TypeError, match="MarginResult"):
        calculate_collateral_fva(
            times,
            values,
            None,
            df,
            borrowing_spread=0.02,
        )


def test_mismatched_vm_shape_rejected(example):
    times, values, df = example
    valid = margin(values)
    invalid = replace(
        valid,
        vm_balance=np.zeros((values.shape[0] + 1, values.shape[1])),
    )

    with pytest.raises(ValueError, match="shape"):
        calculate_collateral_fva(
            times,
            values,
            invalid,
            df,
            borrowing_spread=0.02,
        )


def test_nonfinite_vm_rejected(example):
    times, values, df = example
    valid = margin(values)
    bad_vm = valid.vm_balance.copy()
    bad_vm[0, 1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        calculate_collateral_fva(
            times,
            values,
            replace(valid, vm_balance=bad_vm),
            df,
            borrowing_spread=0.02,
        )


def test_mismatched_time_axis_rejected(example):
    _, values, df = example
    with pytest.raises(ValueError, match="times"):
        calculate_collateral_fva(
            np.array([0.0, 1.0]),
            values,
            margin(values),
            df,
            borrowing_spread=0.02,
        )


@pytest.mark.parametrize(
    "bad_times",
    [
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.5, 0.25, 1.0],
        [0.1, 0.25, 0.75, 1.0],
        [0.0, 0.25, np.nan, 1.0],
    ],
)
def test_invalid_time_grid_delegates_to_fva(example, bad_times):
    _, values, df = example

    with pytest.raises(ValueError):
        calculate_collateral_fva(
            bad_times,
            values,
            margin(values),
            df,
            borrowing_spread=0.02,
        )


@pytest.mark.parametrize(
    "bad_spread",
    [-0.01, np.nan, np.inf, [0.01, 0.02]],
)
def test_invalid_spread_delegates_to_fva(example, bad_spread):
    _, values, df = example

    with pytest.raises(ValueError):
        calculate_collateral_fva(
            example[0],
            values,
            margin(values),
            df,
            borrowing_spread=bad_spread,
        )


def test_invalid_discount_factors_delegates_to_fva(example):
    times, values, _ = example

    with pytest.raises(ValueError):
        calculate_collateral_fva(
            times,
            values,
            margin(values),
            np.array([1.0, 0.0, 0.9, 0.8]),
            borrowing_spread=0.02,
        )


def test_inputs_not_mutated(example):
    times, values, df = example
    vm = margin(values, settlement_lag_steps=1)

    inputs = [
        times,
        values,
        df,
        vm.vm_balance,
        vm.pending_transfers,
    ]
    originals = [array.copy() for array in inputs]

    evaluate(example, vm)

    for actual, expected in zip(inputs, originals):
        np.testing.assert_array_equal(actual, expected)


def test_result_arrays_do_not_alias_inputs(example):
    times, values, _ = example
    vm = margin(values, settlement_lag_steps=1)

    result = evaluate(example, vm)
    original = result.times.copy()

    times[1] = 999.0
    values[0, 1] = 999.0
    vm.vm_balance[0, 1] = 999.0

    np.testing.assert_array_equal(result.times, original)


def test_single_path_supported():
    times = np.array([0.0, 1.0])
    values = np.array([[0.0, 100.0]])
    df = np.array([1.0, 0.95])

    result = calculate_collateral_fva(
        times,
        values,
        margin(values, settlement_lag_steps=1),
        df,
        borrowing_spread=0.02,
    )

    assert result.fca == pytest.approx(1.9)
    assert result.fba == 0.0
    assert result.fva == pytest.approx(-1.9)
