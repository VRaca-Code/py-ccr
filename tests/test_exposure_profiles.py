"""Regression tests for uncollateralized exposure profiles."""

import numpy as np
import pytest

from pyccr.exposure.profiles import (
    ExposureProfiles,
    calculate_exposure_profiles,
)


def test_known_ee_ene_pfe_and_epe() -> None:
    times = [0.0, 1.0, 2.0]
    values = [
        [10.0, -10.0, 30.0],
        [-20.0, 20.0, -40.0],
        [30.0, 0.0, 10.0],
        [-40.0, -30.0, 0.0],
    ]

    result = calculate_exposure_profiles(times, values, confidence_level=0.5)

    assert isinstance(result, ExposureProfiles)

    np.testing.assert_array_equal(result.ee, [10.0, 5.0, 10.0])
    np.testing.assert_array_equal(result.ene, [15.0, 10.0, 10.0])
    np.testing.assert_array_equal(result.pfe, [5.0, 0.0, 5.0])

    assert result.epe == pytest.approx(7.5)
    assert result.confidence_level == 0.5


def test_positive_and_negative_exposure_matrices() -> None:
    result = calculate_exposure_profiles(
        [0.0, 1.0],
        [[10.0, -5.0], [-20.0, 15.0]],
    )

    np.testing.assert_array_equal(
        result.positive_exposures,
        [[10.0, 0.0], [0.0, 15.0]],
    )
    np.testing.assert_array_equal(
        result.negative_exposures,
        [[0.0, 5.0], [20.0, 0.0]],
    )


def test_all_negative_values_have_zero_positive_exposure() -> None:
    result = calculate_exposure_profiles(
        [0.0, 1.0, 2.0],
        [[-10.0, -20.0, -30.0], [-5.0, -15.0, -25.0]],
    )

    np.testing.assert_array_equal(result.ee, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.pfe, [0.0, 0.0, 0.0])
    assert result.epe == 0.0

    np.testing.assert_array_equal(result.ene, [7.5, 17.5, 27.5])


def test_all_zero_values() -> None:
    result = calculate_exposure_profiles(
        [0.0, 0.5, 1.0],
        np.zeros((3, 3)),
    )

    assert result.positive_exposures.shape == (3, 3)
    assert result.negative_exposures.shape == (3, 3)

    np.testing.assert_array_equal(result.ee, np.zeros(3))
    np.testing.assert_array_equal(result.ene, np.zeros(3))
    np.testing.assert_array_equal(result.pfe, np.zeros(3))
    assert result.epe == 0.0


def test_irregular_grid_uses_time_weighted_epe() -> None:
    result = calculate_exposure_profiles(
        [0.0, 1.0, 3.0],
        [[0.0, 10.0, 20.0]],
    )

    # Trapezoidal integral:
    # [0, 1]: 5
    # [1, 3]: 30
    # EPE = 35 / 3.
    assert result.epe == pytest.approx(35.0 / 3.0)


def test_single_time_point_epe_equals_initial_ee() -> None:
    result = calculate_exposure_profiles(
        [0.0],
        [[10.0], [-20.0], [30.0]],
        confidence_level=0.5,
    )

    assert result.ee[0] == pytest.approx(40.0 / 3.0)
    assert result.epe == pytest.approx(result.ee[0])
    assert result.pfe[0] == pytest.approx(10.0)


def test_pfe_uses_linear_quantile_interpolation() -> None:
    result = calculate_exposure_profiles(
        [0.0, 1.0],
        [
            [0.0, 10.0],
            [0.0, 20.0],
            [0.0, 30.0],
            [0.0, 40.0],
        ],
        confidence_level=0.95,
    )

    np.testing.assert_allclose(result.pfe, [0.0, 38.5])


def test_different_confidence_levels_change_pfe_only() -> None:
    values = [[0.0], [10.0], [20.0], [30.0]]

    low = calculate_exposure_profiles([0.0], values, confidence_level=0.25)
    high = calculate_exposure_profiles([0.0], values, confidence_level=0.75)

    assert low.pfe[0] == pytest.approx(7.5)
    assert high.pfe[0] == pytest.approx(22.5)

    np.testing.assert_array_equal(low.ee, high.ee)
    np.testing.assert_array_equal(low.ene, high.ene)
    assert low.epe == high.epe


def test_single_path_profiles() -> None:
    result = calculate_exposure_profiles(
        [0.0, 1.0, 2.0],
        [[10.0, -5.0, 20.0]],
    )

    np.testing.assert_array_equal(result.ee, [10.0, 0.0, 20.0])
    np.testing.assert_array_equal(result.ene, [0.0, 5.0, 0.0])
    np.testing.assert_array_equal(result.pfe, [10.0, 0.0, 20.0])
    assert result.epe == pytest.approx(7.5)


def test_inputs_are_not_mutated() -> None:
    times = np.array([0.0, 1.0, 2.0])
    values = np.array(
        [
            [10.0, -20.0, 30.0],
            [-5.0, 15.0, -25.0],
        ]
    )

    original_times = times.copy()
    original_values = values.copy()

    result = calculate_exposure_profiles(times, values)

    np.testing.assert_array_equal(times, original_times)
    np.testing.assert_array_equal(values, original_values)

    assert not np.shares_memory(result.times, times)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [[0.0, 1.0]],
        [0.5, 1.0],
        [0.0, 0.0],
        [0.0, 1.0, 0.5],
        [0.0, float("nan")],
        [0.0, float("inf")],
    ],
)
def test_invalid_times(times: list) -> None:
    with pytest.raises(ValueError):
        calculate_exposure_profiles(
            times,
            np.ones((2, len(times))),
        )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1.0, 2.0],
        [[1.0]],
        [[1.0, 2.0, 3.0]],
        np.empty((0, 2)),
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
        [[float("-inf"), 1.0]],
    ],
)
def test_invalid_portfolio_values(values: object) -> None:
    with pytest.raises(ValueError):
        calculate_exposure_profiles(
            [0.0, 1.0],
            values,
        )


@pytest.mark.parametrize(
    "level",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "invalid",
        [0.95],
    ],
)
def test_invalid_confidence_levels(level: object) -> None:
    with pytest.raises(ValueError):
        calculate_exposure_profiles(
            [0.0, 1.0],
            [[10.0, 20.0]],
            confidence_level=level,
        )


def test_portfolio_values_can_be_used_directly() -> None:
    from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
    from pyccr.pricing.fx_forward_portfolio import (
        value_fx_forward_portfolio,
    )

    times = np.array([0.0, 0.5, 1.0])
    spots = np.array(
        [
            [1.10, 1.15, 1.20],
            [1.10, 1.05, 1.00],
        ]
    )

    portfolio = value_fx_forward_portfolio(
        [FXForwardExposureTrade(100.0, 1.10, 2.0)],
        times,
        spots,
        domestic_rate=0.0,
        foreign_rate=0.0,
    )

    result = calculate_exposure_profiles(times, portfolio.portfolio_values)

    np.testing.assert_allclose(result.ee, [0.0, 2.5, 5.0])
    np.testing.assert_allclose(result.ene, [0.0, 2.5, 5.0])
    np.testing.assert_allclose(
        result.pfe,
        [0.0, 4.75, 9.5],
    )
    assert result.epe == pytest.approx(2.5)
