"""Fixed-seed statistical calibration of FX mean and EPE intervals.

These tests check empirical coverage against analytical expectations
under a controlled lognormal FX model. They are not proofs of
distribution-free coverage or general model validity.

Fixed seeds make the experiments reproducible. Conservative
binomial acceptance regions limit false failures.
"""

import numpy as np
import pytest
from scipy.stats import binom

from pyccr.convergence.epe_grid import (
    fx_expected_positive_exposure,
)
from pyccr.convergence.exposure import exposure_mean_intervals
from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import (
    simulate_fx_spot_paths,
)
from pyccr.simulation.time_grid import TimeGrid

HORIZON = 1.0
INITIAL_SPOT = 1.1
DOMESTIC_RATE = 0.03
FOREIGN_RATE = 0.01
VOLATILITY = 0.2
STRIKE = 1.1
CONFIDENCE_LEVEL = 0.95
N_EXPERIMENTS = 160
GRID = TimeGrid.uniform(horizon=HORIZON, n_steps=16)
TIMES = np.asarray(GRID.times, dtype=float)

TERMINAL_TARGET = INITIAL_SPOT * np.exp((DOMESTIC_RATE - FOREIGN_RATE) * HORIZON)

ANALYTICAL_EE = fx_expected_positive_exposure(
    TIMES,
    initial_spot=INITIAL_SPOT,
    domestic_rate=DOMESTIC_RATE,
    foreign_rate=FOREIGN_RATE,
    volatility=VOLATILITY,
    strike=STRIKE,
)

DISCRETE_EPE_TARGET = float(np.trapezoid(ANALYTICAL_EE, TIMES) / HORIZON)


def simulate(seed, n_paths, antithetic):
    config = SimulationConfig(
        time_grid=GRID,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    return simulate_fx_spot_paths(
        config,
        initial_spot=INITIAL_SPOT,
        domestic_rate=DOMESTIC_RATE,
        foreign_rate=FOREIGN_RATE,
        volatility=VOLATILITY,
    )


def interval_experiment(seed, n_paths, antithetic):
    paths = simulate(seed, n_paths, antithetic)

    terminal = monte_carlo_mean_interval(
        paths[:, -1],
        confidence_level=CONFIDENCE_LEVEL,
        antithetic=antithetic,
    )
    epe = exposure_mean_intervals(
        TIMES,
        paths - STRIKE,
        confidence_level=CONFIDENCE_LEVEL,
        antithetic=antithetic,
    ).epe

    expected_units = n_paths // 2 if antithetic else n_paths
    assert terminal.n_independent_units == expected_units
    assert epe.n_independent_units == expected_units

    return terminal, epe


def assert_empirical_coverage(hits, n_experiments):
    # 99.9% central binomial acceptance region under nominal
    # 95% coverage. This is a deliberately conservative
    # statistical regression criterion.
    lower = int(
        binom.ppf(
            0.0005,
            n_experiments,
            CONFIDENCE_LEVEL,
        )
    )
    upper = int(
        binom.ppf(
            0.9995,
            n_experiments,
            CONFIDENCE_LEVEL,
        )
    )

    assert lower <= hits <= upper, (
        f"Coverage {hits}/{n_experiments} outside "
        f"prespecified acceptance counts [{lower}, {upper}]"
    )


@pytest.mark.parametrize(
    ("antithetic", "n_paths"),
    [
        (False, 128),
        (True, 128),
        (False, 256),
        (True, 256),
    ],
)
def test_terminal_and_epe_interval_coverage(
    antithetic,
    n_paths,
):
    terminal_hits = 0
    epe_hits = 0

    group = (1 if antithetic else 0) + (2 if n_paths == 256 else 0)

    for experiment in range(N_EXPERIMENTS):
        seed = 45400000 + group * 100000 + experiment
        terminal, epe = interval_experiment(
            seed,
            n_paths,
            antithetic,
        )

        terminal_hits += int(
            terminal.lower_bound <= TERMINAL_TARGET <= terminal.upper_bound
        )
        epe_hits += int(epe.lower_bound <= DISCRETE_EPE_TARGET <= epe.upper_bound)

    assert_empirical_coverage(
        terminal_hits,
        N_EXPERIMENTS,
    )
    assert_empirical_coverage(
        epe_hits,
        N_EXPERIMENTS,
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_mean_standard_error_decreases_with_path_count(
    antithetic,
):
    # Average across independent experiments instead of
    # demanding monotonicity for a single noisy sample.
    mean_se = {}

    for n_paths in (128, 512):
        terminal_se = []
        epe_se = []

        for experiment in range(80):
            seed = 45480000 + (100000 if antithetic else 0) + n_paths * 100 + experiment
            terminal, epe = interval_experiment(
                seed,
                n_paths,
                antithetic,
            )
            terminal_se.append(terminal.standard_error)
            epe_se.append(epe.standard_error)

        mean_se[n_paths] = (
            float(np.mean(terminal_se)),
            float(np.mean(epe_se)),
        )

    for quantity in range(2):
        small = mean_se[128][quantity]
        large = mean_se[512][quantity]

        assert np.isfinite(small) and small > 0
        assert np.isfinite(large) and large > 0

        # Four times as many paths should materially reduce
        # the average SE. Avoid enforcing an exact ratio.
        assert large < 0.8 * small


@pytest.mark.parametrize("antithetic", [False, True])
def test_zero_volatility_analytical_control(antithetic):
    config = SimulationConfig(
        time_grid=GRID,
        n_paths=128,
        seed=45499999,
        antithetic=antithetic,
    )

    paths = simulate_fx_spot_paths(
        config,
        initial_spot=INITIAL_SPOT,
        domestic_rate=DOMESTIC_RATE,
        foreign_rate=FOREIGN_RATE,
        volatility=0.0,
    )

    terminal = monte_carlo_mean_interval(
        paths[:, -1],
        antithetic=antithetic,
    )

    epe = exposure_mean_intervals(
        TIMES,
        paths - STRIKE,
        antithetic=antithetic,
    ).epe

    zero_ee = fx_expected_positive_exposure(
        TIMES,
        initial_spot=INITIAL_SPOT,
        domestic_rate=DOMESTIC_RATE,
        foreign_rate=FOREIGN_RATE,
        volatility=0.0,
        strike=STRIKE,
    )
    zero_target = float(np.trapezoid(zero_ee, TIMES) / HORIZON)

    np.testing.assert_allclose(
        terminal.estimate,
        TERMINAL_TARGET,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        terminal.standard_error,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        epe.estimate,
        zero_target,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        epe.standard_error,
        0.0,
        atol=1e-12,
    )


def test_analytical_targets_are_finite():
    assert np.isfinite(TERMINAL_TARGET)
    assert np.isfinite(DISCRETE_EPE_TARGET)
    assert TERMINAL_TARGET > 0
    assert DISCRETE_EPE_TARGET > 0
    assert ANALYTICAL_EE.shape == TIMES.shape
    assert np.all(ANALYTICAL_EE >= 0)
