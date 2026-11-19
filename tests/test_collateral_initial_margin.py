"""Tests for bilateral segregated IM-adjusted MPOR exposures."""

import numpy as np
import pytest

from pyccr.collateral.initial_margin import (
    calculate_initial_margin_exposures,
)
from pyccr.collateral.mpor import calculate_mpor_exposures


def make_mpor(values, vm, steps):
    return calculate_mpor_exposures(
        np.asarray(values, dtype=np.float64),
        np.asarray(vm, dtype=np.float64),
        mpor_steps=steps,
    )


def test_received_im_reduces_positive_exposure():
    result = make_mpor([[100, 115, 130]], [[95, 95, 95]], 2)

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=20.0,
    )

    np.testing.assert_array_equal(
        adjusted.positive_exposure,
        [[15.0]],
    )
    np.testing.assert_array_equal(
        adjusted.negative_exposure,
        [[0.0]],
    )


def test_posted_im_reduces_negative_exposure():
    result = make_mpor(
        [[-100, -120, -135]],
        [[-95, -95, -95]],
        2,
    )

    adjusted = calculate_initial_margin_exposures(
        result,
        posted_im=25.0,
    )

    np.testing.assert_array_equal(
        adjusted.negative_exposure,
        [[15.0]],
    )
    np.testing.assert_array_equal(
        adjusted.positive_exposure,
        [[0.0]],
    )


def test_zero_im_matches_mpor():
    result = make_mpor(
        [[100, 115, 130], [-100, -120, -135]],
        [[95, 100, 110], [-95, -100, -110]],
        1,
    )

    adjusted = calculate_initial_margin_exposures(result)

    np.testing.assert_array_equal(
        adjusted.positive_exposure,
        result.positive_exposure,
    )
    np.testing.assert_array_equal(
        adjusted.negative_exposure,
        result.negative_exposure,
    )


def test_pathwise_im_freezes_at_default():
    result = make_mpor(
        [[100, 115, 130], [-100, -120, -135]],
        [[95, 100, 110], [-95, -100, -110]],
        2,
    )

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=[
            [20, 35, 50],
            [5, 10, 15],
        ],
        posted_im=[
            [10, 15, 25],
            [25, 35, 45],
        ],
    )

    np.testing.assert_array_equal(
        adjusted.frozen_received_im,
        [[20], [5]],
    )

    np.testing.assert_array_equal(
        adjusted.frozen_posted_im,
        [[10], [25]],
    )

    np.testing.assert_array_equal(
        adjusted.positive_exposure,
        [[15], [0]],
    )

    np.testing.assert_array_equal(
        adjusted.negative_exposure,
        [[0], [15]],
    )


def test_im_directionality():
    result = make_mpor(
        [[100, -100]],
        [[95, -95]],
        0,
    )

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=20.0,
        posted_im=25.0,
    )

    np.testing.assert_array_equal(
        adjusted.positive_exposure,
        [[0, 0]],
    )
    np.testing.assert_array_equal(
        adjusted.negative_exposure,
        [[0, 0]],
    )


def test_overcollateralization_floors_exposure():
    result = make_mpor(
        [[110, -110]],
        [[100, -100]],
        0,
    )

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=50.0,
        posted_im=50.0,
    )

    assert np.all(adjusted.positive_exposure == 0.0)
    assert np.all(adjusted.negative_exposure == 0.0)


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_mpor_horizons(steps):
    values = [[100, 115, 130]]
    vm = [[95, 100, 110]]

    result = make_mpor(values, vm, steps)

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=10.0,
    )

    assert adjusted.positive_exposure.shape == (
        1,
        3 - steps,
    )


def test_single_time_grid():
    result = make_mpor([[100]], [[95]], 0)

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=3.0,
    )

    np.testing.assert_array_equal(
        adjusted.positive_exposure,
        [[2.0]],
    )


@pytest.mark.parametrize(
    "bad",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_received_im_scalar(bad):
    result = make_mpor([[100]], [[95]], 0)

    with pytest.raises(ValueError):
        calculate_initial_margin_exposures(
            result,
            received_im=bad,
        )


@pytest.mark.parametrize(
    "bad",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_posted_im_scalar(bad):
    result = make_mpor([[100]], [[95]], 0)

    with pytest.raises(ValueError):
        calculate_initial_margin_exposures(
            result,
            posted_im=bad,
        )


def test_invalid_received_im_shape():
    result = make_mpor([[100, 110]], [[95, 95]], 1)

    with pytest.raises(ValueError):
        calculate_initial_margin_exposures(
            result,
            received_im=[[10.0]],
        )


def test_invalid_posted_im_shape():
    result = make_mpor([[100, 110]], [[95, 95]], 1)

    with pytest.raises(ValueError):
        calculate_initial_margin_exposures(
            result,
            posted_im=[[10.0]],
        )


def test_invalid_mpor_type():
    with pytest.raises(TypeError):
        calculate_initial_margin_exposures(
            object(),
        )


def test_determinism():
    result = make_mpor(
        [[100, 115, 130]],
        [[95, 100, 110]],
        1,
    )

    first = calculate_initial_margin_exposures(
        result,
        received_im=10.0,
        posted_im=5.0,
    )

    second = calculate_initial_margin_exposures(
        result,
        received_im=10.0,
        posted_im=5.0,
    )

    np.testing.assert_array_equal(
        first.positive_exposure,
        second.positive_exposure,
    )

    np.testing.assert_array_equal(
        first.negative_exposure,
        second.negative_exposure,
    )


def test_inputs_not_mutated():
    values = np.array([[100.0, 115.0, 130.0]])
    vm = np.array([[95.0, 100.0, 110.0]])
    received = np.array([[10.0, 20.0, 30.0]])
    posted = np.array([[5.0, 10.0, 15.0]])

    originals = [array.copy() for array in (values, vm, received, posted)]

    result = make_mpor(values, vm, 1)

    calculate_initial_margin_exposures(
        result,
        received_im=received,
        posted_im=posted,
    )

    for array, original in zip(
        (values, vm, received, posted),
        originals,
    ):
        np.testing.assert_array_equal(array, original)


def test_im_monotonicity():
    rng = np.random.default_rng(528011)

    values = rng.normal(size=(30, 8)) * 100
    vm = rng.normal(size=(30, 8)) * 80

    received = rng.uniform(0, 50, size=(30, 8))
    posted = rng.uniform(0, 50, size=(30, 8))

    result = make_mpor(values, vm, 2)

    without = calculate_initial_margin_exposures(result)

    with_im = calculate_initial_margin_exposures(
        result,
        received_im=received,
        posted_im=posted,
    )

    assert np.all(with_im.positive_exposure <= without.positive_exposure + 1e-12)

    assert np.all(with_im.negative_exposure <= without.negative_exposure + 1e-12)


def test_closeout_accounting():
    result = make_mpor(
        [[100, 115, 130], [-100, -120, -135]],
        [[95, 100, 110], [-95, -100, -110]],
        2,
    )

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=20.0,
        posted_im=25.0,
    )

    residual = adjusted.closeout_values - adjusted.frozen_vm_balance

    np.testing.assert_allclose(
        adjusted.positive_exposure,
        np.maximum(
            residual - adjusted.frozen_received_im,
            0.0,
        ),
    )

    np.testing.assert_allclose(
        adjusted.negative_exposure,
        np.maximum(
            -residual - adjusted.frozen_posted_im,
            0.0,
        ),
    )


def test_output_arrays_are_independent():
    result = make_mpor(
        [[100, 115, 130]],
        [[95, 100, 110]],
        1,
    )

    adjusted = calculate_initial_margin_exposures(
        result,
        received_im=10.0,
    )

    original_mpor_values = result.closeout_values.copy()
    original_mpor_vm = result.frozen_vm_balance.copy()

    adjusted.closeout_values[0, 0] = -999.0
    adjusted.frozen_vm_balance[0, 0] = -999.0

    np.testing.assert_array_equal(
        result.closeout_values,
        original_mpor_values,
    )

    np.testing.assert_array_equal(
        result.frozen_vm_balance,
        original_mpor_vm,
    )
