"""Tests for paired correlation stress and deterministic-rate XVA repricing."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.joint_fx_hw_paths import simulate_joint_fx_hull_white_paths
from pyccr.simulation.time_grid import TimeGrid
from pyccr.stress.correlated import (
    CorrelatedFXStressResult,
    compare_correlated_fx_stress,
)
from pyccr.stress.repricing import compare_fx_forward_stress


def make_inputs(*, seed=42, antithetic=False):
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=200,
        seed=seed,
        antithetic=antithetic,
    )
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0),
        zero_rates=(0.03,) * 4,
    )
    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=0.10,
        volatility=0.01,
    )
    trades = (
        FXForwardExposureTrade(
            foreign_notional=100.0,
            strike=1.10,
            maturity=1.0,
        ),
    )
    return config, model, trades


def compare(*, seed=42, antithetic=False, **overrides):
    config, model, trades = make_inputs(
        seed=seed,
        antithetic=antithetic,
    )
    parameters = {
        "initial_spot": 1.10,
        "domestic_rate": 0.03,
        "foreign_rate": 0.02,
        "fx_volatility": 0.15,
        "correlation": 0.25,
        "correlation_shock": 0.35,
        "counterparty_curve": ConstantHazardCurve(0.02),
        "own_curve": ConstantHazardCurve(0.01),
    }
    parameters.update(overrides)
    return compare_correlated_fx_stress(
        config,
        model,
        trades,
        **parameters,
    )


def test_result_container_and_shapes():
    result = compare()
    assert isinstance(result, CorrelatedFXStressResult)
    assert result.base_paths.fx_spots.shape == (200, 9)
    assert result.stressed_paths.short_rates.shape == (200, 9)
    assert result.comparison.delta_ee.shape == (9,)


def test_correlation_values():
    result = compare()
    assert result.base_correlation == pytest.approx(0.25)
    assert result.stressed_correlation == pytest.approx(0.60)


def test_short_rate_paths_unchanged_by_correlation_only():
    result = compare()
    np.testing.assert_array_equal(
        result.base_paths.short_rates,
        result.stressed_paths.short_rates,
    )


def test_fx_paths_change_under_correlation_shock():
    result = compare()
    assert not np.array_equal(
        result.base_paths.fx_spots,
        result.stressed_paths.fx_spots,
    )


def test_initial_fx_spot_is_identical():
    result = compare()
    np.testing.assert_array_equal(
        result.base_paths.fx_spots[:, 0],
        result.stressed_paths.fx_spots[:, 0],
    )


def test_zero_correlation_shock_has_zero_deltas():
    result = compare(correlation_shock=0.0)
    np.testing.assert_array_equal(
        result.base_paths.fx_spots,
        result.stressed_paths.fx_spots,
    )
    np.testing.assert_array_equal(
        result.comparison.delta_ee,
        np.zeros(9),
    )
    assert result.comparison.delta_cva == 0.0
    assert result.comparison.delta_dva == 0.0
    assert result.comparison.delta_net_xva == 0.0


def test_seeded_reproducibility():
    first = compare()
    second = compare()
    np.testing.assert_array_equal(
        first.stressed_paths.fx_spots,
        second.stressed_paths.fx_spots,
    )
    assert first.comparison.delta_cva == second.comparison.delta_cva


@pytest.mark.parametrize("antithetic", [False, True])
def test_manual_joint_simulation_parity(antithetic):
    config, model, _ = make_inputs(antithetic=antithetic)
    result = compare(antithetic=antithetic)
    expected = simulate_joint_fx_hull_white_paths(
        config,
        model,
        initial_spot=1.10,
        domestic_rate=0.03,
        foreign_rate=0.02,
        fx_volatility=0.15,
        correlation=0.60,
    )
    np.testing.assert_array_equal(
        result.stressed_paths.fx_spots,
        expected.fx_spots,
    )
    np.testing.assert_array_equal(
        result.stressed_paths.short_rates,
        expected.short_rates,
    )


def test_manual_repricing_parity():
    config, model, trades = make_inputs()
    result = compare()
    expected = compare_fx_forward_stress(
        trades,
        result.base_paths.times,
        result.base_paths.fx_spots,
        result.stressed_paths.fx_spots,
        base_domestic_rate=0.03,
        base_foreign_rate=0.02,
        stressed_domestic_rate=0.03,
        stressed_foreign_rate=0.02,
        base_counterparty_curve=ConstantHazardCurve(0.02),
        base_own_curve=ConstantHazardCurve(0.01),
        stressed_counterparty_curve=ConstantHazardCurve(0.02),
        stressed_own_curve=ConstantHazardCurve(0.01),
    )
    np.testing.assert_array_equal(
        result.comparison.delta_ee,
        expected.delta_ee,
    )
    assert result.comparison.delta_cva == expected.delta_cva
    assert config.seed == 42
    assert model.volatility == 0.01


def test_deterministic_discount_factors():
    result = compare()
    expected = np.exp(-0.03 * result.base_paths.times)
    np.testing.assert_allclose(
        result.comparison.base.discount_factors,
        expected,
    )
    np.testing.assert_allclose(
        result.comparison.stressed.discount_factors,
        expected,
    )


def test_net_xva_delta_identity():
    result = compare()
    assert result.comparison.delta_net_xva == pytest.approx(
        result.comparison.delta_dva - result.comparison.delta_cva
    )


def test_matured_exposure_is_zero():
    result = compare()
    assert result.comparison.base.exposures.ee[-1] == 0.0
    assert result.comparison.stressed.exposures.ee[-1] == 0.0


@pytest.mark.parametrize(
    ("correlation", "shock"),
    [
        (0.9, 0.2),
        (-0.9, -0.2),
        (1.0, 0.0),
        (0.0, np.nan),
    ],
)
def test_invalid_correlations(correlation, shock):
    expected_message = "absolute_shock" if np.isnan(shock) else "correlation"
    with pytest.raises(ValueError, match=expected_message):
        compare(correlation=correlation, correlation_shock=shock)


def test_unseeded_config_rejected():
    with pytest.raises(ValueError, match="seed"):
        compare(seed=None)


def test_invalid_config_rejected():
    _, model, trades = make_inputs()
    with pytest.raises(TypeError, match="config"):
        compare_correlated_fx_stress(
            object(),
            model,
            trades,
            initial_spot=1.10,
            domestic_rate=0.03,
            foreign_rate=0.02,
            fx_volatility=0.15,
            correlation=0.25,
            correlation_shock=0.10,
            counterparty_curve=ConstantHazardCurve(0.02),
            own_curve=ConstantHazardCurve(0.01),
        )


def test_invalid_model_rejected():
    config, _, trades = make_inputs()
    with pytest.raises(TypeError, match="model"):
        compare_correlated_fx_stress(
            config,
            object(),
            trades,
            initial_spot=1.10,
            domestic_rate=0.03,
            foreign_rate=0.02,
            fx_volatility=0.15,
            correlation=0.25,
            correlation_shock=0.10,
            counterparty_curve=ConstantHazardCurve(0.02),
            own_curve=ConstantHazardCurve(0.01),
        )
