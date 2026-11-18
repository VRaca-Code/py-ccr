"""Analytical, accounting, and validation tests for deterministic FVA."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.xva.fva import FVAResult, calculate_fva


def calculate(**overrides):
    params = {
        "times": [0.0, 0.5, 1.0],
        "expected_exposure": [0.0, 100.0, 100.0],
        "expected_negative_exposure": [0.0, 40.0, 40.0],
        "discount_factors": [1.0, 1.0, 1.0],
        "borrowing_spread": 0.02,
        "lending_spread": 0.01,
    }
    params.update(overrides)
    return calculate_fva(**params)


def test_result_type_and_shapes():
    result = calculate()
    assert isinstance(result, FVAResult)
    assert result.times.shape == (3,)
    assert result.fca_contributions.shape == (2,)
    assert result.fba_contributions.shape == (2,)
    assert result.net_contributions.shape == (2,)
    assert result.borrowing_spreads.shape == (2,)


def test_analytical_constant_exposure_benchmark():
    result = calculate()
    assert result.fca == pytest.approx(2.0)
    assert result.fba == pytest.approx(0.4)
    assert result.fva == pytest.approx(-1.6)


def test_interval_contributions_reconcile():
    result = calculate()
    np.testing.assert_allclose(result.fca_contributions, [1.0, 1.0])
    np.testing.assert_allclose(result.fba_contributions, [0.2, 0.2])
    np.testing.assert_allclose(result.net_contributions, [-0.8, -0.8])
    assert result.fca == pytest.approx(sum(result.fca_contributions))
    assert result.fba == pytest.approx(sum(result.fba_contributions))
    assert result.fva == pytest.approx(sum(result.net_contributions))


def test_zero_spreads_produce_zero_fva():
    result = calculate(borrowing_spread=0.0, lending_spread=0.0)
    assert result.fca == 0.0
    assert result.fba == 0.0
    assert result.fva == 0.0


def test_zero_exposures_produce_zero_fva():
    result = calculate(
        expected_exposure=[0.0, 0.0, 0.0],
        expected_negative_exposure=[0.0, 0.0, 0.0],
    )
    assert result.fva == 0.0


def test_default_lending_spread_is_zero():
    result = calculate_fva(
        [0.0, 1.0],
        [0.0, 100.0],
        [0.0, 40.0],
        [1.0, 1.0],
        0.02,
    )
    assert result.fca == pytest.approx(2.0)
    assert result.fba == 0.0
    assert result.fva == pytest.approx(-2.0)


def test_funding_benefit_can_exceed_cost():
    result = calculate(borrowing_spread=0.01, lending_spread=0.05)
    assert result.fva > 0.0
    assert result.fva == pytest.approx(result.fba - result.fca)


def test_discount_factors_apply_at_endpoints():
    result = calculate(discount_factors=[1.0, 0.9, 0.8])
    assert result.fca == pytest.approx(1.7)
    assert result.fba == pytest.approx(0.34)


def test_time_zero_exposure_does_not_contribute():
    result = calculate(
        expected_exposure=[1_000_000.0, 100.0, 100.0],
        expected_negative_exposure=[1_000_000.0, 40.0, 40.0],
    )
    assert result.fca == pytest.approx(2.0)
    assert result.fba == pytest.approx(0.4)


def test_nonuniform_time_grid():
    result = calculate(
        times=[0.0, 0.25, 1.0],
        borrowing_spread=[0.01, 0.03],
        lending_spread=[0.0, 0.02],
    )
    assert result.fca == pytest.approx(2.5)
    assert result.fba == pytest.approx(0.6)


def test_scalar_and_interval_spreads_equivalent():
    first = calculate()
    second = calculate(
        borrowing_spread=[0.02, 0.02],
        lending_spread=[0.01, 0.01],
    )
    np.testing.assert_array_equal(
        first.net_contributions,
        second.net_contributions,
    )


def test_spread_scaling():
    base = calculate()
    doubled = calculate(
        borrowing_spread=0.04,
        lending_spread=0.02,
    )
    assert doubled.fva == pytest.approx(2.0 * base.fva)


def test_exposure_scaling():
    base = calculate()
    doubled = calculate(
        expected_exposure=[0.0, 200.0, 200.0],
        expected_negative_exposure=[0.0, 80.0, 80.0],
    )
    assert doubled.fva == pytest.approx(2.0 * base.fva)


def test_two_point_grid():
    result = calculate(
        times=[0.0, 1.0],
        expected_exposure=[0.0, 100.0],
        expected_negative_exposure=[0.0, 40.0],
        discount_factors=[1.0, 1.0],
    )
    assert result.fca == pytest.approx(2.0)
    assert result.fba == pytest.approx(0.4)


def test_integration_with_exposure_profiles():
    times = np.array([0.0, 0.5, 1.0])
    values = np.array(
        [
            [0.0, 100.0, -20.0],
            [0.0, -40.0, 60.0],
        ]
    )
    profiles = calculate_exposure_profiles(times, values)
    result = calculate_fva(
        profiles.times,
        profiles.ee,
        profiles.ene,
        np.ones(3),
        0.02,
        0.01,
    )
    np.testing.assert_allclose(profiles.ee, [0.0, 50.0, 30.0])
    np.testing.assert_allclose(profiles.ene, [0.0, 20.0, 10.0])
    assert result.fca == pytest.approx(0.8)
    assert result.fba == pytest.approx(0.15)
    assert result.fva == pytest.approx(-0.65)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [0.0],
        [0.1, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.5],
        [0.0, np.nan, 1.0],
        [[0.0, 0.5], [0.5, 1.0]],
    ],
)
def test_invalid_times(times):
    with pytest.raises(ValueError, match="times"):
        calculate(times=times)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 100.0],
        [0.0, -1.0, 1.0],
        [0.0, np.nan, 1.0],
        [[0.0, 1.0, 1.0]],
    ],
)
def test_invalid_expected_exposure(values):
    with pytest.raises(ValueError, match="expected_exposure"):
        calculate(expected_exposure=values)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 40.0],
        [0.0, -1.0, 1.0],
        [0.0, np.inf, 1.0],
        [[0.0, 1.0, 1.0]],
    ],
)
def test_invalid_expected_negative_exposure(values):
    with pytest.raises(ValueError, match="expected_negative_exposure"):
        calculate(expected_negative_exposure=values)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, -0.1, 1.0],
        [1.0, np.inf, 1.0],
        [[1.0, 1.0, 1.0]],
    ],
)
def test_invalid_discount_factors(values):
    with pytest.raises(ValueError, match="discount_factors"):
        calculate(discount_factors=values)


@pytest.mark.parametrize(
    "spread",
    [
        -0.01,
        np.nan,
        np.inf,
        [0.01],
        [0.01, -0.01],
        [0.01, np.nan],
        [[0.01, 0.01]],
        "invalid",
    ],
)
def test_invalid_borrowing_spread(spread):
    with pytest.raises(ValueError, match="borrowing_spread"):
        calculate(borrowing_spread=spread)


@pytest.mark.parametrize(
    "spread",
    [
        -0.01,
        np.nan,
        np.inf,
        [0.01],
        [0.01, -0.01],
        [0.01, np.inf],
        [[0.01, 0.01]],
        "invalid",
    ],
)
def test_invalid_lending_spread(spread):
    with pytest.raises(ValueError, match="lending_spread"):
        calculate(lending_spread=spread)


def test_input_arrays_not_mutated():
    times = np.array([0.0, 0.5, 1.0])
    ee = np.array([0.0, 100.0, 100.0])
    ene = np.array([0.0, 40.0, 40.0])
    df = np.ones(3)
    borrow = np.array([0.02, 0.02])
    lend = np.array([0.01, 0.01])

    originals = [value.copy() for value in (times, ee, ene, df, borrow, lend)]

    calculate_fva(times, ee, ene, df, borrow, lend)

    for current, original in zip(
        (times, ee, ene, df, borrow, lend),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_result_arrays_do_not_alias_inputs():
    times = np.array([0.0, 0.5, 1.0])
    borrow = np.array([0.02, 0.02])
    result = calculate(times=times, borrowing_spread=borrow)

    times[1] = 0.75
    borrow[0] = 0.5

    assert result.times[1] == 0.5
    assert result.borrowing_spreads[0] == 0.02


def test_result_dataclass_is_frozen():
    result = calculate()
    with pytest.raises(FrozenInstanceError):
        result.fca = 100.0


def test_positive_cost_and_benefit():
    result = calculate()
    assert result.fca >= 0.0
    assert result.fba >= 0.0
    assert np.all(result.fca_contributions >= 0.0)
    assert np.all(result.fba_contributions >= 0.0)


def test_fva_sign_identity():
    result = calculate()
    assert result.fva == pytest.approx(result.fba - result.fca)
    np.testing.assert_allclose(
        result.net_contributions,
        result.fba_contributions - result.fca_contributions,
    )
