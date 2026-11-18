"""Tests for exact-transition terminal grid controls."""

import numpy as np
import pytest

from pyccr.convergence.time_step import (
    fx_terminal_grid_convergence,
    hull_white_terminal_grid_convergence,
)
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


def fx_parameters(**overrides):
    params = {
        "horizon": 1.0,
        "step_counts": (1, 4, 16),
        "seeds": (101, 202, 303),
        "n_paths": 512,
        "initial_spot": 1.1,
        "domestic_rate": 0.04,
        "foreign_rate": 0.02,
        "volatility": 0.15,
    }
    params.update(overrides)
    return params


def make_model(volatility=0.01):
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0),
        zero_rates=(0.02, 0.022, 0.024, 0.026, 0.03),
    )
    return HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.08,
        volatility=volatility,
    )


def hw_parameters(**overrides):
    params = {
        "model": make_model(),
        "horizon": 1.0,
        "step_counts": (1, 4, 16),
        "seeds": (101, 202, 303),
        "n_paths": 512,
    }
    params.update(overrides)
    return params


def test_fx_shapes_and_analytic_mean():
    result = fx_terminal_grid_convergence(**fx_parameters())
    assert result.model_name == "FX exact lognormal"
    assert result.estimates.shape == (3,)
    assert result.standard_errors.shape == (3,)
    assert result.n_independent_units == 512
    assert result.analytic_terminal_mean == pytest.approx(1.1 * np.exp(0.02))
    np.testing.assert_allclose(
        result.errors,
        result.estimates - result.analytic_terminal_mean,
    )


def test_fx_matches_direct_simulator():
    result = fx_terminal_grid_convergence(**fx_parameters())
    for index, (steps, seed) in enumerate(
        zip(result.step_counts, result.seeds, strict=True)
    ):
        config = SimulationConfig(
            time_grid=TimeGrid.uniform(1.0, int(steps)),
            n_paths=512,
            seed=int(seed),
        )
        direct = simulate_fx_spot_paths(
            config,
            initial_spot=1.1,
            domestic_rate=0.04,
            foreign_rate=0.02,
            volatility=0.15,
        )
        assert result.estimates[index] == pytest.approx(np.mean(direct[:, -1]))


def test_fx_seeded_reproducibility():
    first = fx_terminal_grid_convergence(**fx_parameters())
    second = fx_terminal_grid_convergence(**fx_parameters())
    np.testing.assert_array_equal(first.estimates, second.estimates)


def test_fx_antithetic_sampling():
    result = fx_terminal_grid_convergence(**fx_parameters(antithetic=True))
    assert result.n_independent_units == 256
    assert np.all(result.standard_errors >= 0)


def test_hull_white_terminal_mean():
    result = hull_white_terminal_grid_convergence(**hw_parameters())
    assert result.model_name == "Hull-White exact OU"
    assert result.estimates.shape == (3,)
    assert np.all(np.isfinite(result.estimates))
    assert np.all(result.standard_errors >= 0)
    np.testing.assert_allclose(
        result.errors,
        result.estimates - result.analytic_terminal_mean,
    )


def test_hull_white_antithetic_mean_is_analytic():
    result = hull_white_terminal_grid_convergence(**hw_parameters(antithetic=True))
    np.testing.assert_allclose(
        result.estimates,
        result.analytic_terminal_mean,
        atol=1e-12,
        rtol=0,
    )
    assert result.n_independent_units == 256


def test_hull_white_zero_volatility():
    result = hull_white_terminal_grid_convergence(
        **hw_parameters(model=make_model(volatility=0.0))
    )
    np.testing.assert_allclose(
        result.estimates,
        result.analytic_terminal_mean,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "step_counts,seeds",
    [
        ((), ()),
        ((0, 2), (1, 2)),
        ((2, 2), (1, 2)),
        ((4, 2), (1, 2)),
        ((1, True), (1, 2)),
        ((1, 2), (1,)),
        ((1, 2), (1, 1)),
        ((1, 2), (1, -2)),
        ((1, 2), (1, True)),
        ((1, 2), [1, 2]),
    ],
)
def test_invalid_grid_or_seed_inputs(step_counts, seeds):
    with pytest.raises(ValueError):
        fx_terminal_grid_convergence(
            **fx_parameters(step_counts=step_counts, seeds=seeds)
        )


@pytest.mark.parametrize(
    "n_paths,antithetic",
    [
        (1, False),
        (True, False),
        (3, True),
        (2, True),
        (1.5, False),
    ],
)
def test_invalid_path_counts(n_paths, antithetic):
    with pytest.raises(ValueError, match="n_paths"):
        fx_terminal_grid_convergence(
            **fx_parameters(n_paths=n_paths, antithetic=antithetic)
        )


def test_nonboolean_antithetic():
    with pytest.raises(TypeError, match="antithetic"):
        fx_terminal_grid_convergence(**fx_parameters(antithetic="yes"))


def test_invalid_hull_white_model():
    with pytest.raises(TypeError, match="model"):
        hull_white_terminal_grid_convergence(**hw_parameters(model=None))


def test_hull_white_curve_horizon():
    with pytest.raises(ValueError, match="horizon"):
        hull_white_terminal_grid_convergence(**hw_parameters(horizon=6.0))


def test_invalid_confidence_level():
    with pytest.raises(ValueError, match="confidence_level"):
        fx_terminal_grid_convergence(**fx_parameters(confidence_level=1.0))
