"""Regression tests for scenario-level FX-forward stress repricing."""

import numpy as np
import pytest

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
from pyccr.pricing.fx_forward_portfolio import value_fx_forward_portfolio
from pyccr.stress.repricing import compare_fx_forward_stress

TIMES = np.array([0.0, 0.5, 1.0])
BASE_SPOTS = np.array(
    [
        [1.10, 1.15, 1.20],
        [1.10, 1.05, 1.00],
        [1.10, 1.12, 1.14],
        [1.10, 1.08, 1.06],
    ]
)
TRADES = [
    FXForwardExposureTrade(100.0, 1.12, 2.0),
    FXForwardExposureTrade(-30.0, 1.08, 1.5),
]


def compare(**overrides):
    params = {
        "trades": TRADES,
        "times": TIMES,
        "base_fx_spots": BASE_SPOTS,
        "stressed_fx_spots": BASE_SPOTS * 1.1,
        "base_domestic_rate": 0.03,
        "base_foreign_rate": 0.01,
        "stressed_domestic_rate": 0.04,
        "stressed_foreign_rate": 0.015,
        "base_counterparty_curve": ConstantHazardCurve(0.02),
        "base_own_curve": ConstantHazardCurve(0.015),
        "stressed_counterparty_curve": ConstantHazardCurve(0.035),
        "stressed_own_curve": ConstantHazardCurve(0.02),
    }
    params.update(overrides)
    return compare_fx_forward_stress(**params)


def test_base_and_stressed_portfolios_use_existing_pricer():
    result = compare()

    for scenario, spots, rd, rf in (
        (result.base, BASE_SPOTS, 0.03, 0.01),
        (result.stressed, BASE_SPOTS * 1.1, 0.04, 0.015),
    ):
        expected = value_fx_forward_portfolio(
            TRADES,
            TIMES,
            spots,
            domestic_rate=rd,
            foreign_rate=rf,
        )
        np.testing.assert_allclose(
            scenario.portfolio.portfolio_values,
            expected.portfolio_values,
        )
        np.testing.assert_allclose(
            scenario.portfolio.trade_values,
            expected.trade_values,
        )


def test_stress_deltas_are_stressed_minus_base():
    result = compare()

    for field in ("ee", "ene", "pfe"):
        np.testing.assert_allclose(
            getattr(result, f"delta_{field}"),
            getattr(result.stressed.exposures, field)
            - getattr(result.base.exposures, field),
        )

    assert result.delta_epe == pytest.approx(
        result.stressed.exposures.epe - result.base.exposures.epe
    )
    for field in ("cva", "dva", "net_xva"):
        assert getattr(result, f"delta_{field}") == pytest.approx(
            getattr(result.stressed.xva, field) - getattr(result.base.xva, field)
        )


def test_discount_factors_use_scenario_domestic_rates():
    result = compare()
    np.testing.assert_allclose(result.base.discount_factors, np.exp(-0.03 * TIMES))
    np.testing.assert_allclose(result.stressed.discount_factors, np.exp(-0.04 * TIMES))


def test_no_shock_produces_zero_deltas():
    result = compare(
        stressed_fx_spots=BASE_SPOTS.copy(),
        stressed_domestic_rate=0.03,
        stressed_foreign_rate=0.01,
        stressed_counterparty_curve=ConstantHazardCurve(0.02),
        stressed_own_curve=ConstantHazardCurve(0.015),
    )
    for field in ("delta_ee", "delta_ene", "delta_pfe"):
        np.testing.assert_array_equal(getattr(result, field), np.zeros(TIMES.size))
    for field in (
        "delta_epe",
        "delta_cva",
        "delta_dva",
        "delta_net_xva",
    ):
        assert getattr(result, field) == pytest.approx(0.0)


def test_credit_only_shock_changes_xva_not_exposures():
    result = compare(
        stressed_fx_spots=BASE_SPOTS.copy(),
        stressed_domestic_rate=0.03,
        stressed_foreign_rate=0.01,
        stressed_counterparty_curve=ConstantHazardCurve(0.08),
        stressed_own_curve=ConstantHazardCurve(0.015),
    )
    np.testing.assert_array_equal(result.delta_ee, np.zeros(TIMES.size))
    np.testing.assert_array_equal(result.delta_ene, np.zeros(TIMES.size))
    assert result.delta_epe == pytest.approx(0.0)
    assert result.delta_cva > 0.0


def test_matured_trades_have_zero_exposure_and_xva():
    result = compare(trades=[FXForwardExposureTrade(100.0, 1.1, 0.0)])
    for scenario in (result.base, result.stressed):
        np.testing.assert_array_equal(
            scenario.portfolio.portfolio_values,
            np.zeros_like(BASE_SPOTS),
        )
        assert scenario.exposures.epe == 0.0
        assert scenario.xva.cva == 0.0
        assert scenario.xva.dva == 0.0


def test_offsetting_trades_cancel_under_both_scenarios():
    result = compare(
        trades=[
            FXForwardExposureTrade(100.0, 1.1, 2.0),
            FXForwardExposureTrade(-100.0, 1.1, 2.0),
        ]
    )
    for scenario in (result.base, result.stressed):
        np.testing.assert_allclose(scenario.portfolio.portfolio_values, 0.0, atol=1e-12)
        assert scenario.xva.cva == pytest.approx(0.0, abs=1e-12)
        assert scenario.xva.dva == pytest.approx(0.0, abs=1e-12)


def test_signed_net_xva_identity():
    result = compare()
    for scenario in (result.base, result.stressed):
        assert scenario.xva.net_xva == pytest.approx(
            scenario.xva.dva - scenario.xva.cva
        )
    assert result.delta_net_xva == pytest.approx(result.delta_dva - result.delta_cva)


@pytest.mark.parametrize(
    "stressed_spots",
    [
        np.ones((3, 3)),
        np.ones((4, 2)),
        np.ones((4, 3, 1)),
    ],
)
def test_rejects_mismatched_path_shapes(stressed_spots):
    with pytest.raises(ValueError, match="identical shapes"):
        compare(stressed_fx_spots=stressed_spots)


@pytest.mark.parametrize(
    "bad_spots",
    [
        np.full((4, 3), -1.0),
        np.full((4, 3), np.nan),
    ],
)
def test_rejects_invalid_stressed_paths(bad_spots):
    with pytest.raises(ValueError):
        compare(stressed_fx_spots=bad_spots)


def test_custom_pfe_confidence_level():
    result = compare(pfe_confidence_level=0.99)
    assert result.base.exposures.confidence_level == pytest.approx(0.99)
    assert result.stressed.exposures.confidence_level == pytest.approx(0.99)


def test_input_arrays_are_not_modified():
    times = TIMES.copy()
    spots = BASE_SPOTS.copy()
    stressed = spots * 1.1
    original_stressed = stressed.copy()

    compare(
        times=times,
        base_fx_spots=spots,
        stressed_fx_spots=stressed,
    )

    np.testing.assert_array_equal(times, TIMES)
    np.testing.assert_array_equal(spots, BASE_SPOTS)
    np.testing.assert_array_equal(stressed, original_stressed)
