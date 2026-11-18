"""Tests for simulated FX-forward stress and common random numbers."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid
from pyccr.stress.simulation import (
    FXStressShocks,
    simulate_fx_forward_stress,
)


def inputs(*, antithetic=False, seed=42):
    config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 8),
        n_paths=200,
        seed=seed,
        antithetic=antithetic,
    )
    return {
        "config": config,
        "trades": [
            FXForwardExposureTrade(100.0, 1.10, 1.5),
            FXForwardExposureTrade(-25.0, 1.15, 2.0),
        ],
        "initial_spot": 1.10,
        "domestic_rate": 0.03,
        "foreign_rate": 0.01,
        "volatility": 0.15,
        "counterparty_curve": ConstantHazardCurve(0.02),
        "own_curve": ConstantHazardCurve(0.015),
        "shocks": FXStressShocks(),
    }


def test_no_shock_has_identical_paths_and_zero_deltas():
    result = simulate_fx_forward_stress(**inputs())
    np.testing.assert_array_equal(result.base_fx_spots, result.stressed_fx_spots)
    comparison = result.comparison
    for name in ("delta_ee", "delta_ene", "delta_pfe"):
        np.testing.assert_array_equal(getattr(comparison, name), 0.0)
    for name in (
        "delta_epe",
        "delta_cva",
        "delta_dva",
        "delta_net_xva",
    ):
        assert getattr(comparison, name) == pytest.approx(0.0)


def test_reproducibility():
    params = inputs()
    params["shocks"] = FXStressShocks(
        fx_relative=0.1,
        volatility_absolute=0.05,
        counterparty_hazard_bps=100.0,
    )
    first = simulate_fx_forward_stress(**params)
    second = simulate_fx_forward_stress(**params)

    np.testing.assert_array_equal(first.base_fx_spots, second.base_fx_spots)
    np.testing.assert_array_equal(first.stressed_fx_spots, second.stressed_fx_spots)
    assert first.comparison.delta_cva == second.comparison.delta_cva


def test_base_paths_match_existing_simulator():
    params = inputs()
    result = simulate_fx_forward_stress(**params)
    expected = simulate_fx_spot_paths(
        params["config"],
        initial_spot=params["initial_spot"],
        domestic_rate=params["domestic_rate"],
        foreign_rate=params["foreign_rate"],
        volatility=params["volatility"],
    )
    np.testing.assert_array_equal(result.base_fx_spots, expected)


def test_fx_spot_shock_is_pathwise_proportional():
    params = inputs()
    params["shocks"] = FXStressShocks(fx_relative=0.2)
    result = simulate_fx_forward_stress(**params)
    np.testing.assert_allclose(
        result.stressed_fx_spots,
        result.base_fx_spots * 1.2,
        rtol=1e-14,
        atol=1e-14,
    )
    assert result.stressed_initial_spot == pytest.approx(1.32)


def test_interest_rate_shocks_change_drifts_and_discounts():
    params = inputs()
    params["shocks"] = FXStressShocks(
        domestic_rate_bps=100.0,
        foreign_rate_bps=-50.0,
    )
    result = simulate_fx_forward_stress(**params)
    assert result.stressed_domestic_rate == pytest.approx(0.04)
    assert result.stressed_foreign_rate == pytest.approx(0.005)

    times = np.asarray(params["config"].time_grid.times)
    np.testing.assert_allclose(
        result.comparison.stressed.discount_factors,
        np.exp(-0.04 * times),
    )

    # Shared draws imply an exact pathwise drift difference.
    expected_ratio = np.exp((0.01 + 0.005) * times)
    np.testing.assert_allclose(
        result.stressed_fx_spots / result.base_fx_spots,
        np.broadcast_to(
            expected_ratio[None, :],
            result.base_fx_spots.shape,
        ),
        rtol=1e-13,
        atol=1e-13,
    )


def test_volatility_shock_reuses_same_normal_draws():
    params = inputs()
    params["shocks"] = FXStressShocks(volatility_absolute=0.05)
    result = simulate_fx_forward_stress(**params)
    assert result.stressed_volatility == pytest.approx(0.20)

    dt = np.asarray(params["config"].time_grid.dt)
    rd = params["domestic_rate"]
    rf = params["foreign_rate"]

    base_increments = np.diff(np.log(result.base_fx_spots), axis=1)
    stressed_increments = np.diff(np.log(result.stressed_fx_spots), axis=1)

    base_z = (base_increments - (rd - rf - 0.5 * 0.15**2) * dt[None, :]) / (
        0.15 * np.sqrt(dt)[None, :]
    )

    stressed_z = (stressed_increments - (rd - rf - 0.5 * 0.20**2) * dt[None, :]) / (
        0.20 * np.sqrt(dt)[None, :]
    )

    np.testing.assert_allclose(base_z, stressed_z, atol=1e-12, rtol=0.0)


def test_credit_only_shock_keeps_paths_and_exposures():
    params = inputs()
    params["shocks"] = FXStressShocks(counterparty_hazard_bps=200.0)
    result = simulate_fx_forward_stress(**params)

    np.testing.assert_array_equal(result.base_fx_spots, result.stressed_fx_spots)
    np.testing.assert_array_equal(result.comparison.delta_ee, 0.0)
    np.testing.assert_array_equal(result.comparison.delta_ene, 0.0)
    assert result.stressed_counterparty_curve.hazard_rate == (pytest.approx(0.04))
    assert result.comparison.delta_cva >= 0.0


def test_own_credit_shock_changes_only_own_curve():
    params = inputs()
    params["shocks"] = FXStressShocks(own_hazard_bps=100.0)
    result = simulate_fx_forward_stress(**params)

    assert result.stressed_own_curve.hazard_rate == pytest.approx(0.025)
    assert result.stressed_counterparty_curve.hazard_rate == pytest.approx(0.02)
    np.testing.assert_array_equal(result.base_fx_spots, result.stressed_fx_spots)
    assert result.comparison.delta_cva == pytest.approx(0.0)


def test_combined_shocks_and_shapes():
    params = inputs()
    params["shocks"] = FXStressShocks(
        fx_relative=-0.1,
        domestic_rate_bps=75.0,
        foreign_rate_bps=-25.0,
        volatility_absolute=0.03,
        counterparty_hazard_bps=150.0,
        own_hazard_bps=50.0,
    )
    result = simulate_fx_forward_stress(**params)

    assert result.base_fx_spots.shape == (200, 9)
    assert result.stressed_fx_spots.shape == (200, 9)
    assert np.all(result.stressed_fx_spots > 0.0)
    assert np.all(np.isfinite(result.stressed_fx_spots))
    assert result.comparison.times.shape == (9,)
    assert np.isfinite(result.comparison.delta_net_xva)


@pytest.mark.parametrize("antithetic", [False, True])
def test_antithetic_and_ordinary_sampling(antithetic):
    params = inputs(antithetic=antithetic)
    params["shocks"] = FXStressShocks(
        fx_relative=0.05,
        volatility_absolute=0.02,
    )
    result = simulate_fx_forward_stress(**params)

    assert result.base_fx_spots.shape == (200, 9)
    assert result.stressed_fx_spots.shape == (200, 9)

    if antithetic:
        dt = np.asarray(params["config"].time_grid.dt)
        drift = (
            params["domestic_rate"]
            - params["foreign_rate"]
            - 0.5 * params["volatility"] ** 2
        ) * dt

        innovations = np.diff(np.log(result.base_fx_spots), axis=1) - drift[None, :]
        np.testing.assert_allclose(
            innovations[:100],
            -innovations[100:],
            atol=1e-13,
            rtol=0.0,
        )


def test_seed_required_for_common_random_numbers():
    params = inputs(seed=None)
    with pytest.raises(ValueError, match="fixed simulation seed"):
        simulate_fx_forward_stress(**params)


@pytest.mark.parametrize(
    "shocks",
    [
        FXStressShocks(fx_relative=-1.0),
        FXStressShocks(volatility_absolute=-0.2),
        FXStressShocks(counterparty_hazard_bps=-300.0),
        FXStressShocks(own_hazard_bps=-200.0),
        FXStressShocks(domestic_rate_bps=float("nan")),
        FXStressShocks(foreign_rate_bps=float("inf")),
    ],
)
def test_invalid_shocks_rejected(shocks):
    params = inputs()
    params["shocks"] = shocks
    with pytest.raises(ValueError):
        simulate_fx_forward_stress(**params)


def test_invalid_configuration_and_shock_types():
    params = inputs()
    params["config"] = None
    with pytest.raises(TypeError, match="SimulationConfig"):
        simulate_fx_forward_stress(**params)

    params = inputs()
    params["shocks"] = None
    with pytest.raises(TypeError, match="FXStressShocks"):
        simulate_fx_forward_stress(**params)


def test_shock_configuration_is_immutable():
    shocks = FXStressShocks(fx_relative=0.1)
    with pytest.raises(FrozenInstanceError):
        shocks.fx_relative = 0.2


def test_base_inputs_are_not_mutated():
    params = inputs()
    original_trades = tuple(params["trades"])
    original_grid = params["config"].time_grid.times
    original_hazards = (
        params["counterparty_curve"].hazard_rate,
        params["own_curve"].hazard_rate,
    )

    params["shocks"] = FXStressShocks(
        fx_relative=0.1,
        domestic_rate_bps=100.0,
        counterparty_hazard_bps=50.0,
    )
    simulate_fx_forward_stress(**params)

    assert tuple(params["trades"]) == original_trades
    assert params["config"].time_grid.times == original_grid
    assert (
        params["counterparty_curve"].hazard_rate,
        params["own_curve"].hazard_rate,
    ) == original_hazards
