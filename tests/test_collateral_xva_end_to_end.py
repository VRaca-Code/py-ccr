"""Integration tests for the collateral-aware XVA example."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyccr.collateral.csa import CSAConfig
from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.collateral_profiles import (
    calculate_collateral_exposure_profiles,
)
from pyccr.xva.collateral_xva import calculate_collateral_xva

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "collateral_xva_end_to_end.py"
)


def load_example_module():
    """Load the example without requiring examples to be a package."""
    spec = importlib.util.spec_from_file_location(
        "collateral_xva_end_to_end",
        EXAMPLE_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    import sys

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def example_module():
    return load_example_module()


@pytest.fixture
def result(example_module):
    return example_module.run_example()


def test_example_executes(result):
    assert result.simulation_times.shape == (5,)
    assert result.portfolio_values.shape == (3, 5)
    assert result.discount_factors.shape == (4,)


def test_default_and_closeout_grids(result):
    np.testing.assert_array_equal(
        result.exposures.profiles.times,
        [0.0, 0.25, 0.5, 0.75],
    )

    np.testing.assert_array_equal(
        result.exposures.closeout_times,
        [0.25, 0.5, 0.75, 1.0],
    )

    assert result.exposures.profiles.times[-1] < (result.simulation_times[-1])


def test_discount_factors_align_to_default_times(result):
    np.testing.assert_allclose(
        result.discount_factors,
        np.exp(-0.04 * result.exposures.profiles.times),
        rtol=0.0,
        atol=1e-14,
    )

    np.testing.assert_array_equal(
        result.xva.times,
        result.exposures.profiles.times,
    )


def test_margin_and_mpor_results_are_exposed(result):
    assert result.exposures.margin_result.vm_balance.shape == (3, 5)
    assert result.exposures.mpor_result.closeout_values.shape == (3, 4)
    assert result.exposures.initial_margin_result.positive_exposure.shape == (3, 4)


def test_exposure_profiles_match_pathwise_averages(result):
    positive = result.exposures.initial_margin_result.positive_exposure
    negative = result.exposures.initial_margin_result.negative_exposure

    np.testing.assert_allclose(
        result.exposures.profiles.ee,
        np.mean(positive, axis=0),
    )

    np.testing.assert_allclose(
        result.exposures.profiles.ene,
        np.mean(negative, axis=0),
    )


def test_xva_outputs_are_finite_and_nonnegative_where_applicable(result):
    xva = result.xva

    for value in (
        xva.cva_result.cva,
        xva.dva_result.dva,
        xva.bilateral_result.cva,
        xva.bilateral_result.dva,
        xva.first_to_default_result.cva,
        xva.first_to_default_result.dva,
    ):
        assert np.isfinite(value)
        assert value >= 0.0

    assert np.isfinite(xva.bilateral_result.net_xva)
    assert np.isfinite(xva.first_to_default_result.net_xva)


def test_bilateral_net_xva_identity(result):
    bilateral = result.xva.bilateral_result

    assert bilateral.net_xva == pytest.approx(
        bilateral.dva - bilateral.cva,
        abs=1e-14,
    )


def test_example_matches_direct_engine_calls(result):
    direct_exposures = calculate_collateral_exposure_profiles(
        result.simulation_times,
        result.portfolio_values,
        CSAConfig(
            counterparty_threshold=2.0,
            own_threshold=3.0,
            minimum_transfer_amount=1.0,
            call_frequency_steps=1,
            settlement_lag_steps=0,
        ),
        mpor_steps=1,
        received_im=1.5,
        posted_im=2.0,
        confidence_level=0.95,
    )

    np.testing.assert_array_equal(
        direct_exposures.profiles.ee,
        result.exposures.profiles.ee,
    )

    np.testing.assert_array_equal(
        direct_exposures.profiles.ene,
        result.exposures.profiles.ene,
    )

    direct_xva = calculate_collateral_xva(
        direct_exposures,
        result.discount_factors,
        ConstantHazardCurve(0.03),
        ConstantHazardCurve(0.02),
    )

    assert direct_xva.cva_result.cva == result.xva.cva_result.cva
    assert direct_xva.dva_result.dva == result.xva.dva_result.dva

    assert direct_xva.bilateral_result.net_xva == result.xva.bilateral_result.net_xva

    assert (
        direct_xva.first_to_default_result.net_xva
        == result.xva.first_to_default_result.net_xva
    )


def test_example_reproducible(example_module):
    first = example_module.run_example()
    second = example_module.run_example()

    np.testing.assert_array_equal(
        first.exposures.profiles.ee,
        second.exposures.profiles.ee,
    )

    np.testing.assert_array_equal(
        first.exposures.profiles.ene,
        second.exposures.profiles.ene,
    )

    assert first.xva.cva_result.cva == second.xva.cva_result.cva
    assert first.xva.dva_result.dva == second.xva.dva_result.dva


def test_example_inputs_are_independent_between_runs(example_module):
    first = example_module.run_example()
    second = example_module.run_example()

    first.portfolio_values[0, 1] = 999.0
    first.simulation_times[1] = 999.0
    first.discount_factors[1] = 999.0

    assert second.portfolio_values[0, 1] == 10.0
    assert second.simulation_times[1] == 0.25
    assert second.discount_factors[1] != 999.0


def test_example_prints_key_outputs(example_module, capsys):
    example_module.main()

    output = capsys.readouterr().out

    assert "Collateral-Aware XVA" in output
    assert "CVA:" in output
    assert "DVA:" in output
    assert "Bilateral net XVA:" in output
    assert "First-to-default net XVA:" in output
    assert "MPOR" in output


def test_example_contains_no_random_sampling(result):
    assert result.portfolio_values.shape == (3, 5)

    np.testing.assert_array_equal(
        result.portfolio_values[0],
        [0.0, 10.0, 14.0, 7.0, 3.0],
    )
