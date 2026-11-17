"""Regression tests for the illustrative dashboard scenario runner."""

from dataclasses import replace

import numpy as np
import pytest

from pyccr.dashboard.scenario import (
    DashboardScenarioConfig,
    run_dashboard_scenario,
)


@pytest.fixture(scope="module")
def scenario():
    return run_dashboard_scenario(
        DashboardScenarioConfig(n_paths=100, n_steps=48, seed=42)
    )


def test_default_config_is_reproducible():
    config = DashboardScenarioConfig(n_paths=20, n_steps=48, seed=42)
    first = run_dashboard_scenario(config)
    second = run_dashboard_scenario(config)

    np.testing.assert_array_equal(first.short_rates, second.short_rates)
    np.testing.assert_array_equal(first.portfolio_values, second.portfolio_values)
    np.testing.assert_array_equal(first.exposures.ee, second.exposures.ee)
    assert first.first_to_default_xva.net_xva == second.first_to_default_xva.net_xva


def test_scenario_shapes(scenario):
    assert scenario.times.shape == (49,)
    assert scenario.short_rates.shape == (100, 49)
    assert scenario.trade_values.shape == (2, 100, 49)
    assert scenario.portfolio_values.shape == (100, 49)
    assert scenario.exposures.ee.shape == (49,)
    assert scenario.exposures.ene.shape == (49,)
    assert scenario.exposures.pfe.shape == (49,)


def test_portfolio_is_signed_sum_of_trades(scenario):
    np.testing.assert_allclose(
        scenario.portfolio_values,
        np.sum(scenario.trade_values, axis=0),
        rtol=0,
        atol=1e-9,
    )


def test_exposure_definitions(scenario):
    values = scenario.portfolio_values
    np.testing.assert_allclose(
        scenario.exposures.ee,
        np.maximum(values, 0.0).mean(axis=0),
    )
    np.testing.assert_allclose(
        scenario.exposures.ene,
        np.maximum(-values, 0.0).mean(axis=0),
    )


def test_epe_is_nonnegative(scenario):
    assert np.isfinite(scenario.exposures.epe)
    assert scenario.exposures.epe >= 0.0


def test_xva_components_are_nonnegative(scenario):
    for result in (scenario.standalone_xva, scenario.first_to_default_xva):
        assert result.cva >= 0.0
        assert result.dva >= 0.0
        assert np.isfinite(result.net_xva)


def test_net_xva_identity(scenario):
    for result in (scenario.standalone_xva, scenario.first_to_default_xva):
        assert result.net_xva == pytest.approx(result.dva - result.cva)


def test_first_to_default_does_not_exceed_standalone(scenario):
    assert scenario.first_to_default_xva.cva <= (scenario.standalone_xva.cva + 1e-8)
    assert scenario.first_to_default_xva.dva <= (scenario.standalone_xva.dva + 1e-8)


def test_sensitivity_shape_and_finite_values(scenario):
    sensitivity = scenario.sensitivity

    assert sensitivity.first_to_default_net_xva.shape == (8, 9)

    for name in (
        "naive_cva",
        "naive_dva",
        "naive_net_xva",
        "first_to_default_cva",
        "first_to_default_dva",
        "first_to_default_net_xva",
        "cva_correction",
        "dva_correction",
        "net_xva_correction",
    ):
        assert np.all(np.isfinite(getattr(sensitivity, name)))


def test_sensitivity_matches_baseline(scenario):
    sensitivity = scenario.sensitivity

    cp = np.where(sensitivity.counterparty_spreads_bps == 120.0)[0]
    own = np.where(sensitivity.own_spreads_bps == 180.0)[0]

    assert cp.size == 1
    assert own.size == 1

    row, column = int(own[0]), int(cp[0])

    assert sensitivity.naive_cva[row, column] == pytest.approx(
        scenario.standalone_xva.cva
    )
    assert sensitivity.naive_dva[row, column] == pytest.approx(
        scenario.standalone_xva.dva
    )
    assert sensitivity.first_to_default_cva[row, column] == pytest.approx(
        scenario.first_to_default_xva.cva
    )
    assert sensitivity.first_to_default_dva[row, column] == pytest.approx(
        scenario.first_to_default_xva.dva
    )
    assert sensitivity.first_to_default_net_xva[row, column] == pytest.approx(
        scenario.first_to_default_xva.net_xva
    )


def test_par_rates_are_finite_and_positive(scenario):
    assert np.isfinite(scenario.swap_fixed_rate)
    assert np.isfinite(scenario.swaption_fixed_rate)
    assert scenario.swap_fixed_rate > 0.0
    assert scenario.swaption_fixed_rate > 0.0


def test_zero_volatility_produces_identical_paths():
    config = DashboardScenarioConfig(
        n_paths=20,
        n_steps=48,
        volatility=0.0,
    )
    result = run_dashboard_scenario(config)

    np.testing.assert_allclose(
        result.short_rates,
        np.broadcast_to(result.short_rates[0], result.short_rates.shape),
    )


def test_notional_scaling():
    base = DashboardScenarioConfig(n_paths=20, n_steps=48)
    first = run_dashboard_scenario(base)
    second = run_dashboard_scenario(replace(base, notional=2 * base.notional))

    np.testing.assert_allclose(
        second.portfolio_values,
        2.0 * first.portfolio_values,
        rtol=1e-12,
        atol=1e-7,
    )
    assert second.first_to_default_xva.cva == pytest.approx(
        2.0 * first.first_to_default_xva.cva
    )
    assert second.first_to_default_xva.dva == pytest.approx(
        2.0 * first.first_to_default_xva.dva
    )


@pytest.mark.parametrize("n_steps", [0, 1, 5, 7, 47, -6])
def test_invalid_step_counts(n_steps):
    with pytest.raises(ValueError, match="multiple of six"):
        run_dashboard_scenario(DashboardScenarioConfig(n_paths=20, n_steps=n_steps))


@pytest.mark.parametrize("n_steps", [True, 48.0, "48", None])
def test_invalid_step_types(n_steps):
    with pytest.raises(TypeError, match="multiple of six"):
        run_dashboard_scenario(DashboardScenarioConfig(n_paths=20, n_steps=n_steps))


@pytest.mark.parametrize("notional", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_notionals(notional):
    with pytest.raises(ValueError, match="notional"):
        run_dashboard_scenario(
            DashboardScenarioConfig(n_paths=20, n_steps=48, notional=notional)
        )


def test_invalid_config_type():
    with pytest.raises(TypeError, match="DashboardScenarioConfig"):
        run_dashboard_scenario(None)


def test_antithetic_path_requirement():
    with pytest.raises(ValueError, match="even"):
        run_dashboard_scenario(DashboardScenarioConfig(n_paths=21, n_steps=48))


def test_invalid_recovery():
    with pytest.raises(ValueError, match="recovery"):
        run_dashboard_scenario(
            DashboardScenarioConfig(
                n_paths=20,
                n_steps=48,
                recovery_rate=1.0,
            )
        )


def test_invalid_confidence_level():
    with pytest.raises(ValueError, match="confidence_level"):
        run_dashboard_scenario(
            DashboardScenarioConfig(
                n_paths=20,
                n_steps=48,
                confidence_level=1.0,
            )
        )
