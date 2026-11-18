"""Simulated FX-forward stress testing with common random numbers.

The model uses constant domestic/foreign rates and GBM FX paths.
It excludes correlation stress, stochastic rates, collateral,
wrong-way risk, and first-to-default adjustments.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.market.discount_curve import DiscountCurve
from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.fx_spot_paths import simulate_fx_spot_paths
from pyccr.stress.repricing import (
    FXStressComparison,
    compare_fx_forward_stress,
)
from pyccr.stress.scenarios import (
    shock_discount_curve,
    shock_fx_spot,
    shock_hazard_curve,
    shock_volatility,
)


@dataclass(frozen=True)
class FXStressShocks:
    """Deterministic shifts applied to the base scenario."""

    fx_relative: float = 0.0
    domestic_rate_bps: float = 0.0
    foreign_rate_bps: float = 0.0
    volatility_absolute: float = 0.0
    counterparty_hazard_bps: float = 0.0
    own_hazard_bps: float = 0.0


@dataclass(frozen=True)
class SimulatedFXStressResult:
    """Paired FX paths and their scenario-level XVA comparison."""

    base_fx_spots: NDArray[np.float64]
    stressed_fx_spots: NDArray[np.float64]
    comparison: FXStressComparison
    stressed_initial_spot: float
    stressed_domestic_rate: float
    stressed_foreign_rate: float
    stressed_volatility: float
    stressed_counterparty_curve: ConstantHazardCurve
    stressed_own_curve: ConstantHazardCurve


def _shift_constant_rate(rate: float, shift_bps: float) -> float:
    """Apply a parallel zero-rate shock to a constant-rate curve."""
    curve = DiscountCurve(
        tenors=(1.0,),
        zero_rates=(rate,),
    )
    return shock_discount_curve(curve, shift_bps).zero_rates[0]


def simulate_fx_forward_stress(
    config: SimulationConfig,
    trades: Sequence[FXForwardExposureTrade],
    *,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    counterparty_curve: ConstantHazardCurve,
    own_curve: ConstantHazardCurve,
    shocks: FXStressShocks,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
    pfe_confidence_level: float = 0.95,
) -> SimulatedFXStressResult:
    """Simulate base and shocked scenarios with shared normal draws.

    A fixed seed is required. Both simulations use the same immutable
    configuration, so their underlying standard-normal draws match.
    Antithetic paths, if enabled, retain their pair structure.

    This is a risk-neutral scenario comparison, not a historical
    stress forecast. Shared draws reduce simulation noise but do
    not establish statistical independence or eliminate uncertainty.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if config.seed is None:
        raise ValueError(
            "a fixed simulation seed is required for common random numbers"
        )
    if not isinstance(shocks, FXStressShocks):
        raise TypeError("shocks must be an FXStressShocks")

    stressed_spot = shock_fx_spot(initial_spot, shocks.fx_relative)
    stressed_domestic_rate = _shift_constant_rate(
        domestic_rate, shocks.domestic_rate_bps
    )
    stressed_foreign_rate = _shift_constant_rate(foreign_rate, shocks.foreign_rate_bps)
    stressed_volatility = shock_volatility(volatility, shocks.volatility_absolute)
    stressed_counterparty_curve = shock_hazard_curve(
        counterparty_curve, shocks.counterparty_hazard_bps
    )
    stressed_own_curve = shock_hazard_curve(own_curve, shocks.own_hazard_bps)

    base_paths = simulate_fx_spot_paths(
        config,
        initial_spot=initial_spot,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        volatility=volatility,
    )

    stressed_paths = simulate_fx_spot_paths(
        config,
        initial_spot=stressed_spot,
        domestic_rate=stressed_domestic_rate,
        foreign_rate=stressed_foreign_rate,
        volatility=stressed_volatility,
    )

    comparison = compare_fx_forward_stress(
        trades,
        np.asarray(config.time_grid.times, dtype=np.float64),
        base_paths,
        stressed_paths,
        base_domestic_rate=domestic_rate,
        base_foreign_rate=foreign_rate,
        stressed_domestic_rate=stressed_domestic_rate,
        stressed_foreign_rate=stressed_foreign_rate,
        base_counterparty_curve=counterparty_curve,
        base_own_curve=own_curve,
        stressed_counterparty_curve=stressed_counterparty_curve,
        stressed_own_curve=stressed_own_curve,
        counterparty_recovery_rate=counterparty_recovery_rate,
        own_recovery_rate=own_recovery_rate,
        pfe_confidence_level=pfe_confidence_level,
    )

    return SimulatedFXStressResult(
        base_fx_spots=base_paths,
        stressed_fx_spots=stressed_paths,
        comparison=comparison,
        stressed_initial_spot=stressed_spot,
        stressed_domestic_rate=stressed_domestic_rate,
        stressed_foreign_rate=stressed_foreign_rate,
        stressed_volatility=stressed_volatility,
        stressed_counterparty_curve=stressed_counterparty_curve,
        stressed_own_curve=stressed_own_curve,
    )
