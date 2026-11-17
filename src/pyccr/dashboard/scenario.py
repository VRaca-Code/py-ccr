"""Reproducible Hull-White IRS/swaption scenario for dashboard consumers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.credit.cds_calibration import calibrate_flat_hazard_from_cds
from pyccr.exposure.profiles import ExposureProfiles, calculate_exposure_profiles
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_ir_portfolio import value_hull_white_ir_portfolio
from pyccr.pricing.hull_white_swaption_exposure import HullWhiteSwaptionTrade
from pyccr.pricing.interest_rate_swap import InterestRateSwap
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.hull_white_paths import simulate_hull_white_short_rates
from pyccr.simulation.time_grid import TimeGrid
from pyccr.xva.bilateral import BilateralXVAResult, calculate_bilateral_xva
from pyccr.xva.first_to_default import (
    FirstToDefaultXVAResult,
    calculate_first_to_default_xva,
)
from pyccr.xva.sensitivity import (
    BilateralXVASensitivityResult,
    calculate_bilateral_xva_sensitivity,
)


@dataclass(frozen=True)
class DashboardScenarioConfig:
    """Inputs for the illustrative, single-currency dashboard scenario."""

    n_paths: int = 2000
    n_steps: int = 48
    seed: int = 42
    mean_reversion: float = 0.08
    volatility: float = 0.01
    notional: float = 10_000_000.0
    counterparty_spread_bps: float = 120.0
    own_spread_bps: float = 180.0
    recovery_rate: float = 0.4
    confidence_level: float = 0.95


@dataclass(frozen=True)
class DashboardScenarioResult:
    """Model outputs consumed by dashboard charts and summary metrics."""

    times: NDArray[np.float64]
    short_rates: NDArray[np.float64]
    trade_values: NDArray[np.float64]
    portfolio_values: NDArray[np.float64]
    exposures: ExposureProfiles
    standalone_xva: BilateralXVAResult
    first_to_default_xva: FirstToDefaultXVAResult
    sensitivity: BilateralXVASensitivityResult
    swap_fixed_rate: float
    swaption_fixed_rate: float


def run_dashboard_scenario(
    config: DashboardScenarioConfig = DashboardScenarioConfig(),
) -> DashboardScenarioResult:
    """Run a reproducible illustrative EUR interest-rate portfolio.

    All trades use the same Hull-White paths and initial discount curve.
    Signed trade values are aggregated for illustration only; this does
    not establish a legally enforceable netting set.

    CDS spreads use the flat credit-triangle hazard approximation.
    XVA assumes independent default times and deterministic discounting,
    without collateral or wrong-way risk.
    """
    if not isinstance(config, DashboardScenarioConfig):
        raise TypeError("config must be a DashboardScenarioConfig")

    if isinstance(config.n_steps, bool) or not isinstance(config.n_steps, int):
        raise TypeError("n_steps must be a positive multiple of six")
    if config.n_steps < 6 or config.n_steps % 6 != 0:
        raise ValueError("n_steps must be a positive multiple of six")

    if not np.isfinite(config.notional) or config.notional <= 0:
        raise ValueError("notional must be positive and finite")

    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
        zero_rates=(0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
    )
    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=config.mean_reversion,
        volatility=config.volatility,
    )

    grid = TimeGrid.uniform(horizon=3.0, n_steps=config.n_steps)
    times = np.asarray(grid.times, dtype=np.float64)

    simulation = SimulationConfig(
        time_grid=grid,
        n_paths=config.n_paths,
        seed=config.seed,
        antithetic=True,
    )
    short_rates = simulate_hull_white_short_rates(simulation, model)

    swap_schedule = np.arange(0.0, 3.0 + 0.25, 0.5)
    swap_payments = swap_schedule[1:]
    swap_accruals = np.diff(swap_schedule)
    swap_discount = np.asarray(curve.discount_factor(swap_payments))
    swap_annuity = float(np.dot(swap_accruals, swap_discount))
    swap_par = float((1.0 - swap_discount[-1]) / swap_annuity)

    swap = InterestRateSwap(
        currency="EUR",
        floating_index="EURIBOR",
        notional=config.notional,
        fixed_rate=swap_par,
        payment_times=tuple(float(t) for t in swap_schedule),
        pay_fixed=False,
    )

    expiry = 1.0
    option_payments = (1.5, 2.0, 2.5, 3.0)
    option_schedule = np.asarray((expiry, *option_payments))
    option_accruals = np.diff(option_schedule)
    option_discount = np.asarray(curve.discount_factor(option_payments))
    option_annuity = float(np.dot(option_accruals, option_discount))
    option_par = float(
        (float(curve.discount_factor(expiry)) - option_discount[-1]) / option_annuity
    )

    swaption = HullWhiteSwaptionTrade(
        expiry=expiry,
        payment_times=option_payments,
        fixed_rate=option_par,
        notional=config.notional,
        option_type="payer",
    )

    values = value_hull_white_ir_portfolio(
        trades=[swap, swaption],
        model=model,
        times=times,
        short_rates=short_rates,
    )
    exposures = calculate_exposure_profiles(
        times,
        values.portfolio_values,
        confidence_level=config.confidence_level,
    )

    discount_factors = np.asarray(
        curve.discount_factor(times),
        dtype=np.float64,
    )

    counterparty_curve = calibrate_flat_hazard_from_cds(
        config.counterparty_spread_bps,
        config.recovery_rate,
    )
    own_curve = calibrate_flat_hazard_from_cds(
        config.own_spread_bps,
        config.recovery_rate,
    )

    xva_args = (
        times,
        exposures.ee,
        exposures.ene,
        discount_factors,
        counterparty_curve,
        own_curve,
    )

    standalone = calculate_bilateral_xva(
        *xva_args,
        counterparty_recovery_rate=config.recovery_rate,
        own_recovery_rate=config.recovery_rate,
    )
    first_to_default = calculate_first_to_default_xva(
        *xva_args,
        counterparty_recovery_rate=config.recovery_rate,
        own_recovery_rate=config.recovery_rate,
    )

    counterparty_spreads = np.asarray(
        (25.0, 50.0, 75.0, 100.0, 120.0, 150.0, 200.0, 300.0, 500.0)
    )
    own_spreads = np.asarray((25.0, 50.0, 100.0, 150.0, 180.0, 250.0, 350.0, 500.0))

    sensitivity = calculate_bilateral_xva_sensitivity(
        times=times,
        expected_exposure=exposures.ee,
        expected_negative_exposure=exposures.ene,
        discount_factors=discount_factors,
        counterparty_spreads_bps=counterparty_spreads,
        own_spreads_bps=own_spreads,
        counterparty_recovery_rate=config.recovery_rate,
        own_recovery_rate=config.recovery_rate,
    )

    return DashboardScenarioResult(
        times=times,
        short_rates=short_rates,
        trade_values=values.trade_values,
        portfolio_values=values.portfolio_values,
        exposures=exposures,
        standalone_xva=standalone,
        first_to_default_xva=first_to_default,
        sensitivity=sensitivity,
        swap_fixed_rate=swap_par,
        swaption_fixed_rate=option_par,
    )
