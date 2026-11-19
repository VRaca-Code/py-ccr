"""Numerical and accounting tests for bilateral variation margin."""

import numpy as np
import pytest

from pyccr.collateral import (
    CSAConfig,
    simulate_variation_margin,
    variation_margin_target,
)


def test_zero_threshold_immediate_margin():
    values = np.array([[10.0, 15.0, -4.0]])

    result = simulate_variation_margin(values, CSAConfig())

    np.testing.assert_allclose(result.vm_balance, values)
    np.testing.assert_allclose(result.positive_exposure, 0.0)
    np.testing.assert_allclose(result.negative_exposure, 0.0)


def test_bilateral_thresholds():
    values = np.array([[12.0, -9.0, 3.0, -2.0]])

    csa = CSAConfig(
        counterparty_threshold=5.0,
        own_threshold=4.0,
    )

    expected = np.array([[7.0, -5.0, 0.0, 0.0]])

    np.testing.assert_allclose(
        variation_margin_target(values, csa),
        expected,
    )


def test_minimum_transfer_amount():
    values = np.array([[0.0, 2.0, 4.0, 8.0]])

    csa = CSAConfig(minimum_transfer_amount=5.0)

    result = simulate_variation_margin(values, csa)

    np.testing.assert_allclose(
        result.vm_balance,
        [[0.0, 0.0, 0.0, 8.0]],
    )


def test_call_frequency():
    values = np.array([[0.0, 10.0, 20.0, 30.0, 40.0]])

    csa = CSAConfig(call_frequency_steps=2)

    result = simulate_variation_margin(values, csa)

    np.testing.assert_allclose(
        result.vm_balance,
        [[0.0, 0.0, 20.0, 20.0, 40.0]],
    )


def test_settlement_lag():
    values = np.array([[10.0, 20.0, 30.0, 40.0]])

    csa = CSAConfig(settlement_lag_steps=1)

    result = simulate_variation_margin(values, csa)

    np.testing.assert_allclose(
        result.vm_balance,
        [[0.0, 10.0, 20.0, 30.0]],
    )

    np.testing.assert_allclose(
        result.margin_calls,
        [[10.0, 10.0, 10.0, 10.0]],
    )

    np.testing.assert_allclose(
        result.pending_transfers,
        [[10.0, 10.0, 10.0, 10.0]],
    )


def test_pending_calls_prevent_duplicate_requests():
    values = np.array([[10.0, 10.0, 10.0, 10.0]])

    csa = CSAConfig(settlement_lag_steps=2)

    result = simulate_variation_margin(values, csa)

    np.testing.assert_allclose(
        result.margin_calls,
        [[10.0, 0.0, 0.0, 0.0]],
    )

    np.testing.assert_allclose(
        result.vm_balance,
        [[0.0, 0.0, 10.0, 10.0]],
    )


def test_multiple_paths_are_independent():
    values = np.array(
        [
            [10.0, 20.0, 30.0],
            [-5.0, -10.0, -15.0],
        ]
    )

    result = simulate_variation_margin(
        values,
        CSAConfig(settlement_lag_steps=1),
    )

    np.testing.assert_allclose(
        result.vm_balance,
        [
            [0.0, 10.0, 20.0],
            [0.0, -5.0, -10.0],
        ],
    )


def test_zero_collateral_equivalence():
    values = np.array(
        [
            [10.0, -5.0, 7.0],
            [-4.0, 6.0, -3.0],
        ]
    )

    csa = CSAConfig(
        counterparty_threshold=1e9,
        own_threshold=1e9,
    )

    result = simulate_variation_margin(values, csa)

    np.testing.assert_allclose(result.vm_balance, 0.0)

    np.testing.assert_allclose(
        result.positive_exposure,
        np.maximum(values, 0.0),
    )

    np.testing.assert_allclose(
        result.negative_exposure,
        np.maximum(-values, 0.0),
    )


def test_overcollateralization():
    values = np.array([[5.0, -5.0]])

    csa = CSAConfig(
        initial_vm_balance=10.0,
        call_frequency_steps=3,
    )

    result = simulate_variation_margin(values, csa)

    # At t=0, the scheduled call returns excess collateral.
    # At t=1, no new call occurs.
    np.testing.assert_allclose(
        result.vm_balance,
        [[5.0, 5.0]],
    )

    np.testing.assert_allclose(
        result.negative_exposure,
        [[0.0, 10.0]],
    )


def test_unsettled_transfer_is_not_effective_collateral():
    values = np.array([[10.0, 10.0]])

    result = simulate_variation_margin(
        values,
        CSAConfig(settlement_lag_steps=3),
    )

    np.testing.assert_allclose(result.vm_balance, 0.0)
    np.testing.assert_allclose(result.pending_transfers, 10.0)
    np.testing.assert_allclose(result.positive_exposure, 10.0)


def test_initial_vm_balance():
    values = np.array([[10.0, 10.0]])

    result = simulate_variation_margin(
        values,
        CSAConfig(
            initial_vm_balance=4.0,
            call_frequency_steps=2,
            settlement_lag_steps=1,
        ),
    )

    np.testing.assert_allclose(
        result.vm_balance,
        [[4.0, 10.0]],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"counterparty_threshold": -1.0},
        {"own_threshold": -1.0},
        {"minimum_transfer_amount": -1.0},
        {"call_frequency_steps": 0},
        {"settlement_lag_steps": -1},
        {"initial_vm_balance": float("nan")},
    ],
)
def test_invalid_csa_parameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        CSAConfig(**kwargs)


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 3)),
        np.empty((2, 0)),
        np.array([[float("nan")]]),
        np.array([[float("inf")]]),
    ],
)
def test_invalid_portfolio_values(values):
    with pytest.raises(ValueError):
        simulate_variation_margin(values, CSAConfig())


def test_deterministic_reproducibility():
    values = np.array(
        [
            [1.0, 4.0, -3.0, 2.0],
            [-2.0, -5.0, 8.0, 1.0],
        ]
    )

    csa = CSAConfig(
        counterparty_threshold=1.0,
        own_threshold=2.0,
        minimum_transfer_amount=0.5,
        settlement_lag_steps=1,
    )

    first = simulate_variation_margin(values, csa)
    second = simulate_variation_margin(values, csa)

    for field in (
        "vm_balance",
        "margin_calls",
        "settled_transfers",
        "pending_transfers",
        "positive_exposure",
        "negative_exposure",
    ):
        np.testing.assert_array_equal(
            getattr(first, field),
            getattr(second, field),
        )
