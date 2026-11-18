"""Common-random-number correlation stress for joint FX/Hull-White paths.

The joint simulator uses constant FX drift rates. Short-rate paths are
reported but do not enter FX drift, FX-forward pricing, or XVA discounting.
This is not a stochastic-rate XVA or wrong-way-risk model.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.fx_forward_exposure import FXForwardExposureTrade
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.joint_fx_hw_paths import (
    JointFXHullWhitePaths,
    simulate_joint_fx_hull_white_paths,
)
from pyccr.stress.repricing import FXStressComparison, compare_fx_forward_stress
from pyccr.stress.scenarios import shock_correlation


@dataclass(frozen=True)
class CorrelatedFXStressResult:
    """Joint paths and their deterministic-rate FX-forward stress comparison."""

    base_paths: JointFXHullWhitePaths
    stressed_paths: JointFXHullWhitePaths
    comparison: FXStressComparison
    base_correlation: float
    stressed_correlation: float


def compare_correlated_fx_stress(
    config: SimulationConfig,
    model: HullWhiteCurveModel,
    trades: Sequence[FXForwardExposureTrade],
    *,
    initial_spot: float,
    domestic_rate: float,
    foreign_rate: float,
    fx_volatility: float,
    correlation: float,
    correlation_shock: float,
    counterparty_curve: ConstantHazardCurve,
    own_curve: ConstantHazardCurve,
    counterparty_recovery_rate: float = 0.4,
    own_recovery_rate: float = 0.4,
    pfe_confidence_level: float = 0.95,
) -> CorrelatedFXStressResult:
    """Compare FX-forward exposures under a two-factor correlation shock.

    Both simulations use the same config and seed, so the underlying
    independent Gaussian draws are paired. A fixed integer seed is
    required to make the pairing explicit and reproducible.

    The first Gaussian factor drives the Hull-White short rate, while
    the second drives FX. The existing Cholesky construction leaves
    the first factor unchanged under a correlation-only shock.

    Rates, volatility, spot, and hazard curves remain identical.
    Simulated short rates do not feed into FX drift or XVA discounting.
    """

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if config.seed is None:
        raise ValueError("config.seed must be fixed for paired stress")
    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")
    if not isinstance(counterparty_curve, ConstantHazardCurve):
        raise TypeError("counterparty_curve must be a ConstantHazardCurve")
    if not isinstance(own_curve, ConstantHazardCurve):
        raise TypeError("own_curve must be a ConstantHazardCurve")

    stressed_correlation = shock_correlation(correlation, correlation_shock)
    base_correlation = shock_correlation(correlation, 0.0)

    parameters = {
        "initial_spot": initial_spot,
        "domestic_rate": domestic_rate,
        "foreign_rate": foreign_rate,
        "fx_volatility": fx_volatility,
    }

    base_paths = simulate_joint_fx_hull_white_paths(
        config,
        model,
        correlation=base_correlation,
        **parameters,
    )
    stressed_paths = simulate_joint_fx_hull_white_paths(
        config,
        model,
        correlation=stressed_correlation,
        **parameters,
    )

    comparison = compare_fx_forward_stress(
        trades,
        base_paths.times,
        base_paths.fx_spots,
        stressed_paths.fx_spots,
        base_domestic_rate=domestic_rate,
        base_foreign_rate=foreign_rate,
        stressed_domestic_rate=domestic_rate,
        stressed_foreign_rate=foreign_rate,
        base_counterparty_curve=counterparty_curve,
        base_own_curve=own_curve,
        stressed_counterparty_curve=counterparty_curve,
        stressed_own_curve=own_curve,
        counterparty_recovery_rate=counterparty_recovery_rate,
        own_recovery_rate=own_recovery_rate,
        pfe_confidence_level=pfe_confidence_level,
    )

    return CorrelatedFXStressResult(
        base_paths=base_paths,
        stressed_paths=stressed_paths,
        comparison=comparison,
        base_correlation=base_correlation,
        stressed_correlation=stressed_correlation,
    )
