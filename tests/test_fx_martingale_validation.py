"""Risk-neutral consistency tests for deterministic-rate FX simulation.

The standalone FX simulator uses constant domestic and foreign
interest rates and exact lognormal transitions. Accordingly,
S(t) * exp(-(r_d-r_f)*t) is a martingale under its specified
pricing measure.

These tests do not establish consistency of a stochastic-rate
FX/Hull-White joint model.

Fixed seeds and conservative statistical thresholds are used
for reproducible regression tests. Statistical checks are not
mathematical proofs of model correctness.
"""

import numpy as np
import pytest

from pyccr.convergence.mean import monte_carlo_mean_interval
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import (
    simulate_fx_spot_paths,
)
from pyccr.simulation.time_grid import TimeGrid

INITIAL_SPOT = 1.1


def simulate(
    *,
    horizon=1.0,
    n_steps=12,
    n_paths=4096,
    seed=455001,
    domestic_rate=0.03,
    foreign_rate=0.01,
    volatility=0.2,
    antithetic=False,
):
    grid = TimeGrid.uniform(
        horizon=horizon,
        n_steps=n_steps,
    )
    config = SimulationConfig(
        time_grid=grid,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    paths = simulate_fx_spot_paths(
        config,
        initial_spot=INITIAL_SPOT,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        volatility=volatility,
    )
    return np.asarray(grid.times, dtype=float), paths


@pytest.mark.parametrize(
    ("domestic_rate", "foreign_rate", "volatility"),
    [
        (0.03, 0.01, 0.2),
        (0.01, 0.04, 0.3),
        (0.0, 0.0, 0.1),
        (-0.01, 0.02, 0.25),
    ],
)
@pytest.mark.parametrize("antithetic", [False, True])
def test_discounted_fx_mean_matches_martingale_target(
    domestic_rate,
    foreign_rate,
    volatility,
    antithetic,
):
    times, paths = simulate(
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        volatility=volatility,
        antithetic=antithetic,
        seed=(455100 + int(volatility * 1000) + (10000 if antithetic else 0)),
    )

    discounted = paths * np.exp(-(domestic_rate - foreign_rate) * times)

    # Multiple observation dates share simulated paths.
    # This is a conservative per-date diagnostic rather
    # than a simultaneous confidence band.
    for column in (3, 6, 9, 12):
        interval = monte_carlo_mean_interval(
            discounted[:, column],
            antithetic=antithetic,
        )

        error = abs(interval.estimate - INITIAL_SPOT)

        # Wide fixed-seed regression tolerance: flag
        # material drift errors without relying on a
        # fragile nominal 95% interval inclusion test.
        tolerance = 6.0 * interval.standard_error + 1e-12
        assert error <= tolerance, (
            f"Discounted FX martingale error at "
            f"t={times[column]}: {error} > {tolerance}"
        )


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize(
    "horizon",
    [0.25, 1.0, 2.0],
)
def test_terminal_mean_matches_analytical_expectation(
    horizon,
    antithetic,
):
    domestic_rate = 0.035
    foreign_rate = 0.012

    _, paths = simulate(
        horizon=horizon,
        n_steps=16,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        antithetic=antithetic,
        seed=455200 + int(horizon * 100) + (10000 if antithetic else 0),
    )

    analytical = INITIAL_SPOT * np.exp((domestic_rate - foreign_rate) * horizon)

    interval = monte_carlo_mean_interval(
        paths[:, -1],
        antithetic=antithetic,
    )

    assert abs(interval.estimate - analytical) <= (
        6.0 * interval.standard_error + 1e-12
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_terminal_variance_matches_lognormal_formula(
    antithetic,
):
    horizon = 1.0
    volatility = 0.2
    domestic_rate = 0.03
    foreign_rate = 0.01

    _, paths = simulate(
        horizon=horizon,
        n_paths=8192,
        volatility=volatility,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        antithetic=antithetic,
        seed=455300 + (10000 if antithetic else 0),
    )

    analytical_mean = INITIAL_SPOT * np.exp((domestic_rate - foreign_rate) * horizon)
    analytical_variance = analytical_mean**2 * np.expm1(volatility**2 * horizon)

    empirical_variance = float(np.var(paths[:, -1], ddof=1))

    # A deliberately broad deterministic tolerance.
    # Variance estimates are especially noisy in
    # positively skewed lognormal distributions.
    assert np.isclose(
        empirical_variance,
        analytical_variance,
        rtol=0.12,
        atol=1e-12,
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_zero_volatility_discounted_paths_are_constant(
    antithetic,
):
    domestic_rate = 0.04
    foreign_rate = 0.015

    times, paths = simulate(
        volatility=0.0,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        antithetic=antithetic,
        n_paths=128,
    )

    expected = INITIAL_SPOT * np.exp((domestic_rate - foreign_rate) * times)

    np.testing.assert_allclose(
        paths,
        np.broadcast_to(expected, paths.shape),
        rtol=1e-12,
        atol=1e-12,
    )

    discounted = paths * np.exp(-(domestic_rate - foreign_rate) * times)

    np.testing.assert_allclose(
        discounted,
        INITIAL_SPOT,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_fx_paths_are_strictly_positive(antithetic):
    _, paths = simulate(
        volatility=0.4,
        antithetic=antithetic,
    )
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0)


def test_same_seed_reproduces_paths():
    _, first = simulate(seed=455400)
    _, second = simulate(seed=455400)

    np.testing.assert_array_equal(first, second)


def test_different_seed_changes_stochastic_paths():
    _, first = simulate(seed=455401)
    _, second = simulate(seed=455402)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("antithetic", [False, True])
def test_martingale_initial_value(antithetic):
    _, paths = simulate(antithetic=antithetic)

    np.testing.assert_array_equal(
        paths[:, 0],
        np.full(paths.shape[0], INITIAL_SPOT),
    )
