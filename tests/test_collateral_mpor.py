"""Tests for MPOR closeout exposure accounting."""

import numpy as np
import pytest

from pyccr.collateral import (
    CSAConfig,
    MPORResult,
    calculate_mpor_exposures,
    simulate_variation_margin,
)


def test_bilateral_closeout_example():
    values = np.array(
        [
            [100.0, 110.0, 130.0, 145.0],
            [-100.0, -110.0, -135.0, -150.0],
        ]
    )

    csa = CSAConfig(
        counterparty_threshold=5.0,
        own_threshold=5.0,
    )
    vm = simulate_variation_margin(values, csa)

    result = calculate_mpor_exposures(values, vm.vm_balance, mpor_steps=2)

    assert isinstance(result, MPORResult)
    np.testing.assert_array_equal(result.default_indices, [0, 1])
    np.testing.assert_array_equal(result.closeout_indices, [2, 3])
    np.testing.assert_allclose(result.frozen_vm_balance[:, 0], [95.0, -95.0])
    np.testing.assert_allclose(result.positive_exposure[:, 0], [35.0, 0.0])
    np.testing.assert_allclose(result.negative_exposure[:, 0], [0.0, 40.0])


def test_zero_mpor_matches_vm_exposure():
    values = np.array(
        [
            [10.0, -5.0, 30.0],
            [-7.0, 8.0, -2.0],
        ]
    )

    vm = simulate_variation_margin(values, CSAConfig(settlement_lag_steps=1))

    result = calculate_mpor_exposures(values, vm.vm_balance, mpor_steps=0)

    np.testing.assert_allclose(result.positive_exposure, vm.positive_exposure)
    np.testing.assert_allclose(result.negative_exposure, vm.negative_exposure)


def test_pending_transfers_excluded():
    values = np.array([[100.0, 130.0, 150.0, 170.0]])

    vm = simulate_variation_margin(values, CSAConfig(settlement_lag_steps=2))

    assert vm.pending_transfers[0, 0] == 100.0
    assert vm.vm_balance[0, 0] == 0.0

    result = calculate_mpor_exposures(values, vm.vm_balance, mpor_steps=1)

    assert result.positive_exposure[0, 0] == 130.0


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_accounting_identities(steps):
    values = np.array(
        [
            [10.0, 20.0, -5.0, 30.0],
            [-10.0, -20.0, 5.0, -30.0],
        ]
    )
    vm = np.array(
        [
            [2.0, 5.0, 1.0, 10.0],
            [-2.0, -5.0, -1.0, -10.0],
        ]
    )

    result = calculate_mpor_exposures(values, vm, mpor_steps=steps)

    residual = result.closeout_values - result.frozen_vm_balance

    np.testing.assert_allclose(
        result.positive_exposure - result.negative_exposure,
        residual,
    )
    np.testing.assert_allclose(
        result.positive_exposure * result.negative_exposure,
        0.0,
    )

    assert np.all(result.positive_exposure >= 0.0)
    assert np.all(result.negative_exposure >= 0.0)


def test_multiple_paths():
    values = np.array(
        [
            [10.0, 20.0, 30.0],
            [-10.0, -20.0, -30.0],
        ]
    )
    vm = np.array(
        [
            [8.0, 18.0, 28.0],
            [-8.0, -18.0, -28.0],
        ]
    )

    result = calculate_mpor_exposures(values, vm, mpor_steps=1)

    np.testing.assert_allclose(
        result.positive_exposure,
        [[12.0, 12.0], [0.0, 0.0]],
    )
    np.testing.assert_allclose(
        result.negative_exposure,
        [[0.0, 0.0], [12.0, 12.0]],
    )


def test_maximum_valid_horizon():
    values = np.array([[1.0, 2.0, 3.0, 4.0]])
    vm = np.zeros_like(values)

    result = calculate_mpor_exposures(values, vm, mpor_steps=3)

    assert result.positive_exposure.shape == (1, 1)
    assert result.default_indices.tolist() == [0]
    assert result.closeout_indices.tolist() == [3]
    assert result.positive_exposure[0, 0] == 4.0


def test_single_time_zero_mpor():
    values = np.array([[5.0]])
    vm = np.array([[2.0]])

    result = calculate_mpor_exposures(values, vm, mpor_steps=0)

    assert result.positive_exposure[0, 0] == 3.0


@pytest.mark.parametrize("steps", [-1, -2])
def test_negative_mpor_rejected(steps):
    values = np.ones((2, 3))

    with pytest.raises(ValueError, match="nonnegative"):
        calculate_mpor_exposures(values, values, mpor_steps=steps)


@pytest.mark.parametrize(
    "steps",
    [True, False, np.bool_(True), 1.5, "1", None],
)
def test_invalid_mpor_type_rejected(steps):
    values = np.ones((2, 3))

    with pytest.raises(TypeError, match="integer"):
        calculate_mpor_exposures(values, values, mpor_steps=steps)


@pytest.mark.parametrize("steps", [3, 4, 100])
def test_closeout_beyond_horizon_rejected(steps):
    values = np.ones((2, 3))

    with pytest.raises(ValueError, match="horizon"):
        calculate_mpor_exposures(values, values, mpor_steps=steps)


def test_shape_mismatch_rejected():
    values = np.ones((2, 3))
    vm = np.ones((2, 2))

    with pytest.raises(ValueError, match="shape"):
        calculate_mpor_exposures(values, vm, mpor_steps=1)


@pytest.mark.parametrize(
    "values",
    [
        np.ones(3),
        np.ones((0, 3)),
        np.ones((2, 0)),
    ],
)
def test_invalid_portfolio_shape_rejected(values):
    with pytest.raises(ValueError):
        calculate_mpor_exposures(values, values, mpor_steps=0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_portfolio_rejected(invalid):
    values = np.array([[1.0, invalid]])
    vm = np.zeros_like(values)

    with pytest.raises(ValueError, match="finite"):
        calculate_mpor_exposures(values, vm, mpor_steps=0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_vm_rejected(invalid):
    values = np.array([[1.0, 2.0]])
    vm = np.array([[0.0, invalid]])

    with pytest.raises(ValueError, match="finite"):
        calculate_mpor_exposures(values, vm, mpor_steps=0)


def test_input_mutation_does_not_change_output():
    values = np.array([[10.0, 20.0, 30.0]])
    vm = np.array([[5.0, 10.0, 15.0]])

    result = calculate_mpor_exposures(values, vm, mpor_steps=1)

    values[:] = 999.0
    vm[:] = 999.0

    np.testing.assert_allclose(result.closeout_values, [[20.0, 30.0]])
    np.testing.assert_allclose(result.frozen_vm_balance, [[5.0, 10.0]])


def test_deterministic_reproducibility():
    values = np.array(
        [
            [10.0, 20.0, 30.0],
            [-5.0, -10.0, -15.0],
        ]
    )
    vm = np.zeros_like(values)

    first = calculate_mpor_exposures(values, vm, mpor_steps=1)
    second = calculate_mpor_exposures(values, vm, mpor_steps=1)

    np.testing.assert_array_equal(
        first.positive_exposure,
        second.positive_exposure,
    )
    np.testing.assert_array_equal(
        first.negative_exposure,
        second.negative_exposure,
    )
