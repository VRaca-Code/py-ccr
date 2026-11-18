"""Tests for deterministic FX EPE time-integration convergence."""

import numpy as np
import pytest

from pyccr.convergence.epe_grid import (
    fx_epe_grid_convergence,
    fx_expected_positive_exposure,
)


def parameters(**overrides):
    values = {
        "horizon": 1.0,
        "step_counts": (4, 16, 64, 256),
        "initial_spot": 1.1,
        "domestic_rate": 0.03,
        "foreign_rate": 0.01,
        "volatility": 0.2,
        "strike": 1.1,
    }
    values.update(overrides)
    return values


def test_expected_exposure_at_zero():
    values = fx_expected_positive_exposure(
        [0.0],
        initial_spot=1.1,
        domestic_rate=0.03,
        foreign_rate=0.01,
        volatility=0.2,
        strike=1.0,
    )
    np.testing.assert_allclose(values, [0.1])


def test_expected_exposure_matches_zero_volatility():
    times = np.array([0.0, 0.25, 0.5, 1.0])
    values = fx_expected_positive_exposure(
        times,
        initial_spot=1.1,
        domestic_rate=0.03,
        foreign_rate=0.01,
        volatility=0.0,
        strike=1.0,
    )
    expected = np.maximum(1.1 * np.exp(0.02 * times) - 1.0, 0.0)
    np.testing.assert_allclose(values, expected)


def test_exposure_scale_is_linear():
    common = {
        "initial_spot": 1.1,
        "domestic_rate": 0.03,
        "foreign_rate": 0.01,
        "volatility": 0.2,
        "strike": 1.1,
    }
    baseline = fx_expected_positive_exposure([0.0, 0.5, 1.0], **common)
    scaled = fx_expected_positive_exposure(
        [0.0, 0.5, 1.0],
        exposure_scale=7.0,
        **common,
    )
    np.testing.assert_allclose(scaled, 7.0 * baseline)


def test_expected_exposure_nonnegative_and_finite():
    values = fx_expected_positive_exposure(
        np.linspace(0, 2, 21),
        initial_spot=1.1,
        domestic_rate=0.03,
        foreign_rate=0.01,
        volatility=0.2,
        strike=1.1,
    )
    assert values.shape == (21,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)


def test_epe_grid_shapes():
    result = fx_epe_grid_convergence(**parameters())
    assert result.step_counts.tolist() == [4, 16, 64, 256]
    assert result.epe_estimates.shape == (4,)
    assert result.absolute_errors.shape == (4,)
    assert result.reference_epe > 0
    assert result.reference_error_estimate >= 0
    np.testing.assert_allclose(
        result.signed_errors,
        result.epe_estimates - result.reference_epe,
    )


def test_epe_grid_refinement_reduces_error():
    result = fx_epe_grid_convergence(**parameters())
    assert result.absolute_errors[-1] < result.absolute_errors[0]
    assert result.absolute_errors[-1] < 1e-5


def test_constant_exposure_has_zero_integration_error():
    result = fx_epe_grid_convergence(
        **parameters(
            domestic_rate=0.0,
            foreign_rate=0.0,
            volatility=0.0,
            strike=1.0,
        )
    )
    np.testing.assert_allclose(
        result.epe_estimates,
        0.1,
        atol=1e-12,
        rtol=0,
    )
    assert result.reference_epe == pytest.approx(0.1)


def test_reproducibility():
    first = fx_epe_grid_convergence(**parameters())
    second = fx_epe_grid_convergence(**parameters())
    np.testing.assert_array_equal(first.epe_estimates, second.epe_estimates)


@pytest.mark.parametrize(
    "counts",
    [
        (),
        (0, 2),
        (4, 4),
        (8, 4),
        (True, 4),
        (1.5, 4),
        [4, 8],
    ],
)
def test_invalid_step_counts(counts):
    with pytest.raises(ValueError, match="step_counts"):
        fx_epe_grid_convergence(**parameters(step_counts=counts))


@pytest.mark.parametrize(
    "name,value",
    [
        ("initial_spot", 0.0),
        ("volatility", -0.1),
        ("strike", 0.0),
        ("exposure_scale", 0.0),
        ("domestic_rate", float("nan")),
        ("foreign_rate", float("inf")),
    ],
)
def test_invalid_numeric_values(name, value):
    with pytest.raises(ValueError):
        fx_epe_grid_convergence(**parameters(**{name: value}))


def test_boolean_parameter_is_type_error():
    with pytest.raises(TypeError, match="exposure_scale"):
        fx_epe_grid_convergence(**parameters(exposure_scale=True))


def test_negative_time_rejected():
    with pytest.raises(ValueError, match="times"):
        fx_expected_positive_exposure(
            [-0.1, 0.2],
            initial_spot=1.1,
            domestic_rate=0.03,
            foreign_rate=0.01,
            volatility=0.2,
            strike=1.1,
        )


def test_invalid_horizon():
    with pytest.raises(ValueError):
        fx_epe_grid_convergence(**parameters(horizon=0.0))
