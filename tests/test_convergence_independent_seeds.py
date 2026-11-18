"""Independent-seed FX convergence tests."""

import numpy as np
import pytest

from pyccr.convergence.independent_seeds import (
    independent_seed_fx_study,
)
from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.simulation.time_grid import TimeGrid


def parameters(**overrides):
    result = {
        "time_grid": TimeGrid.uniform(1.0, 8),
        "seeds": (11, 22, 33, 44),
        "n_paths": 256,
        "initial_spot": 1.10,
        "domestic_rate": 0.04,
        "foreign_rate": 0.02,
        "volatility": 0.15,
    }
    result.update(overrides)
    return result


def test_reproducible_independent_runs():
    first = independent_seed_fx_study(**parameters())
    second = independent_seed_fx_study(**parameters())

    np.testing.assert_array_equal(
        first.terminal_spot_means,
        second.terminal_spot_means,
    )
    np.testing.assert_array_equal(first.epe, second.epe)
    np.testing.assert_array_equal(first.terminal_pfe, second.terminal_pfe)
    assert first.terminal_ee_across_runs == (second.terminal_ee_across_runs)
    assert first.n_independent_units_per_run == 256


def test_each_spot_mean_matches_direct_simulator():
    result = independent_seed_fx_study(**parameters())

    for index, seed in enumerate(result.seeds):
        config = SimulationConfig(
            time_grid=TimeGrid.uniform(1.0, 8),
            n_paths=256,
            seed=int(seed),
        )
        paths = simulate_fx_spot_paths(
            config,
            initial_spot=1.10,
            domestic_rate=0.04,
            foreign_rate=0.02,
            volatility=0.15,
        )
        assert result.terminal_spot_means[index] == (
            pytest.approx(np.mean(paths[:, -1]))
        )


def test_analytic_fx_mean_and_between_run_error():
    result = independent_seed_fx_study(**parameters())

    expected = 1.10 * np.exp(0.02)
    assert result.terminal_spot_analytic_mean == (pytest.approx(expected))
    expected_across = monte_carlo_mean_interval(result.terminal_spot_means)
    assert result.terminal_spot_across_runs == expected_across
    assert result.terminal_spot_across_runs.n_paths == 4
    assert result.terminal_spot_across_runs.n_independent_units == 4
    assert result.terminal_spot_across_runs.standard_error > 0


def test_exposure_statistics_and_uncertainties():
    result = independent_seed_fx_study(**parameters())

    assert result.terminal_ee.shape == (4,)
    assert result.epe.shape == (4,)
    assert result.terminal_pfe.shape == (4,)
    assert np.all(result.terminal_ee >= 0)
    assert np.all(result.epe >= 0)
    assert np.all(result.terminal_pfe >= 0)
    assert np.all(result.terminal_ee_within_standard_errors >= 0)
    assert np.all(result.epe_within_standard_errors >= 0)

    assert result.terminal_ee_across_runs.estimate == (
        pytest.approx(np.mean(result.terminal_ee))
    )
    assert result.epe_across_runs.estimate == (pytest.approx(np.mean(result.epe)))


def test_different_seeds_produce_different_estimates():
    result = independent_seed_fx_study(**parameters())
    assert np.unique(result.terminal_spot_means).size > 1
    assert np.unique(result.terminal_pfe).size > 1


def test_antithetic_replications():
    result = independent_seed_fx_study(**parameters(antithetic=True))
    assert result.n_paths_per_run == 256
    assert result.n_independent_units_per_run == 128
    assert result.antithetic
    assert result.terminal_spot_means.shape == (4,)
    assert result.epe_across_runs.n_independent_units == 4


@pytest.mark.parametrize(
    "bad_seeds",
    [
        (),
        (1, 2),
        (1, 1, 2),
        (-1, 2, 3),
        (True, 2, 3),
        (1.0, 2, 3),
        [1, 2, 3],
    ],
)
def test_invalid_seeds(bad_seeds):
    with pytest.raises(ValueError, match="seeds"):
        independent_seed_fx_study(**parameters(seeds=bad_seeds))


@pytest.mark.parametrize("n_paths", [0, 1, True, 1.5])
def test_invalid_path_counts(n_paths):
    with pytest.raises(ValueError, match="n_paths"):
        independent_seed_fx_study(**parameters(n_paths=n_paths))


@pytest.mark.parametrize("n_paths", [2, 3, 5])
def test_invalid_antithetic_counts(n_paths):
    with pytest.raises(ValueError, match="n_paths"):
        independent_seed_fx_study(**parameters(n_paths=n_paths, antithetic=True))


def test_nonboolean_antithetic_rejected():
    with pytest.raises(TypeError, match="antithetic"):
        independent_seed_fx_study(**parameters(antithetic="yes"))


@pytest.mark.parametrize(
    "name,value",
    [
        ("exposure_scale", 0.0),
        ("exposure_scale", -1.0),
        ("exposure_scale", float("nan")),
        ("exposure_strike", float("inf")),
        ("exposure_strike", True),
    ],
)
def test_invalid_exposure_parameters(name, value):
    expected_exception = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(expected_exception, match=name):
        independent_seed_fx_study(**parameters(**{name: value}))


def test_non_time_grid_rejected():
    with pytest.raises(TypeError, match="time_grid"):
        independent_seed_fx_study(**parameters(time_grid=[0.0, 1.0]))


def test_invalid_confidence_level_propagates():
    with pytest.raises(ValueError, match="confidence_level"):
        independent_seed_fx_study(**parameters(confidence_level=1.0))


def test_zero_volatility_is_deterministic():
    result = independent_seed_fx_study(**parameters(volatility=0.0))
    assert result.terminal_spot_across_runs.standard_error == 0
    assert result.terminal_ee_across_runs.standard_error == 0
    assert result.epe_across_runs.standard_error == 0
    assert result.terminal_pfe_across_runs.standard_error == 0
