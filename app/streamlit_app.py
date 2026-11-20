"""Interactive Streamlit dashboard for illustrative PyCCR analytics."""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting_charts import (
    binomial_null_chart,
    bootstrap_coverage_chart,
    breach_autocorrelation_chart,
    breach_timeline_chart,
    rolling_coverage_chart,
)
from app.dashboard_theme import CHART_COLORS, apply_dashboard_theme, style_figure
from app.exposure_explorer import render_exposure_explorer
from app.portfolio_contribution import render_portfolio_contributions
from app.term_structure_charts import (
    discount_factor_chart,
    forward_curve_chart,
    forward_slope_chart,
    hull_white_drift_chart,
    parameter_comparison_chart,
)
from app.xva_stress_panel import render_xva_stress_panel
from pyccr.backtesting.bootstrap import bootstrap_coverage_interval
from pyccr.backtesting.christoffersen import christoffersen_coverage_test
from pyccr.backtesting.coverage import test_coverage
from pyccr.backtesting.dependence import breach_dependence_diagnostics
from pyccr.dashboard.scenario import (
    DashboardScenarioConfig,
    run_dashboard_scenario,
)
from pyccr.dashboard.term_structure import calculate_term_structure_diagnostics
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve

st.set_page_config(
    page_title="PyCCR | Counterparty Credit Risk & XVA",
    page_icon="📊",
    layout="wide",
)

apply_dashboard_theme()


@st.cache_data(show_spinner=False)
def calculate_scenario(config: DashboardScenarioConfig):
    """Cache deterministic simulation results for a given configuration."""
    return run_dashboard_scenario(config)


def exposure_chart(result):
    """Plot expected exposures and the selected PFE percentile."""
    fig = go.Figure()

    for name, values, color in (
        ("EE", result.exposures.ee, CHART_COLORS["ee"]),
        ("ENE", result.exposures.ene, CHART_COLORS["ene"]),
        ("PFE", result.exposures.pfe, CHART_COLORS["pfe"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=result.times,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2.8},
                hovertemplate=(
                    "Time: %{x:.2f} years"
                    "<br>Exposure: %{y:,.2f} EUR"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Time (years)",
        yaxis_title="Exposure (EUR)",
        yaxis_tickformat=",.0f",
    )

    return style_figure(
        fig,
        title="Expected exposure and potential future exposure",
        height=430,
    )


def trade_chart(result):
    """Show expected signed mark-to-market for each trade and portfolio."""
    fig = go.Figure()

    series = (
        (
            "Receiver IRS",
            result.trade_values[0].mean(axis=0),
            CHART_COLORS["ee"],
            2.3,
        ),
        (
            "Payer swaption",
            result.trade_values[1].mean(axis=0),
            CHART_COLORS["pfe"],
            2.3,
        ),
        (
            "Combined portfolio",
            result.portfolio_values.mean(axis=0),
            CHART_COLORS["xva"],
            3.5,
        ),
    )

    for name, values, color, width in series:
        fig.add_trace(
            go.Scatter(
                x=result.times,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": width},
                hovertemplate=(
                    "Time: %{x:.2f} years"
                    "<br>Expected MTM: %{y:,.2f} EUR"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
        line_color="#94A3B8",
    )

    fig.update_layout(
        xaxis_title="Time (years)",
        yaxis_title="Expected MTM (EUR)",
        yaxis_tickformat=",.0f",
    )

    return style_figure(
        fig,
        title="Expected signed mark-to-market",
        height=430,
    )


def xva_chart(result):
    """Compare standalone and first-to-default XVA components."""
    standalone = result.standalone_xva
    first = result.first_to_default_xva

    fig = go.Figure()

    for label, values, color in (
        (
            "Standalone",
            (standalone.cva, standalone.dva, standalone.net_xva),
            CHART_COLORS["pfe"],
        ),
        (
            "First-to-default",
            (first.cva, first.dva, first.net_xva),
            CHART_COLORS["xva"],
        ),
    ):
        fig.add_trace(
            go.Bar(
                name=label,
                x=["CVA", "DVA", "Net XVA"],
                y=values,
                marker_color=color,
                hovertemplate=(
                    "%{x}<br>Adjustment: %{y:,.2f} EUR<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis_title="Adjustment (EUR)",
        yaxis_tickformat=",.0f",
        bargap=0.24,
        bargroupgap=0.08,
    )

    return style_figure(
        fig,
        title="Standalone vs first-to-default adjustments",
        height=400,
    )


def sensitivity_chart(result, metric):
    """Visualize bilateral XVA across counterparty and own CDS spreads."""
    sensitivity = result.sensitivity
    matrix = getattr(sensitivity, metric)

    signed = metric.endswith(("net_xva", "correction"))

    if signed:
        bound = max(abs(float(matrix.min())), abs(float(matrix.max())))
        color_settings = {
            "colorscale": "RdBu_r",
            "zmin": -bound,
            "zmax": bound,
            "zmid": 0.0,
        }
    else:
        color_settings = {
            "colorscale": "Viridis",
            "zmin": 0.0,
        }

    fig = go.Figure(
        go.Heatmap(
            x=sensitivity.counterparty_spreads_bps,
            y=sensitivity.own_spreads_bps,
            z=matrix,
            colorbar={"title": "EUR"},
            hovertemplate=(
                "Counterparty CDS: %{x:.0f} bps"
                "<br>Own CDS: %{y:.0f} bps"
                "<br>Adjustment: %{z:,.2f} EUR"
                "<extra></extra>"
            ),
            **color_settings,
        )
    )

    fig.update_layout(
        xaxis_title="Counterparty CDS spread (bps)",
        yaxis_title="Own CDS spread (bps)",
    )

    return style_figure(
        fig,
        title=metric.replace("_", " ").title(),
        height=480,
    )


# Dashboard identity and scenario context.
header_left, header_right = st.columns([4, 1])

with header_left:
    st.title("PyCCR")
    st.caption("Counterparty Credit Risk & XVA Analytics")

with header_right:
    st.caption("MODEL STATUS")
    st.markdown("**Illustrative scenario**")

st.divider()

portfolio_col, model_col, horizon_col = st.columns(3)

with portfolio_col:
    st.caption("PORTFOLIO")
    st.markdown("**EUR receiver IRS + payer swaption**")

with model_col:
    st.caption("MODEL")
    st.markdown("**Hull–White one-factor**")

with horizon_col:
    st.caption("SIMULATION HORIZON")
    st.markdown("**3 years**")

st.info(
    "Illustrative EUR receiver IRS + payer swaption portfolio. "
    "Signed trade aggregation does not establish legally enforceable netting. "
    "This dashboard scenario excludes collateral, wrong-way risk, "
    "and correlated defaults."
)

with st.sidebar:
    st.header("Scenario configuration")

    with st.form("scenario_parameters"):
        n_paths = st.select_slider(
            "Monte Carlo paths",
            options=[200, 500, 1000, 2000, 5000],
            value=2000,
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
        )

        st.subheader("Hull–White model")

        mean_reversion = st.slider(
            "Mean reversion",
            min_value=0.01,
            max_value=0.30,
            value=0.08,
            step=0.01,
        )

        volatility = st.slider(
            "Short-rate volatility",
            min_value=0.001,
            max_value=0.030,
            value=0.010,
            step=0.001,
            format="%.3f",
        )

        st.subheader("Portfolio")

        notional_millions = st.slider(
            "Notional per trade (EUR millions)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        st.subheader("Credit assumptions")

        counterparty_spread = st.slider(
            "Counterparty CDS (bps)",
            min_value=0,
            max_value=500,
            value=120,
            step=5,
        )

        own_spread = st.slider(
            "Own CDS (bps)",
            min_value=0,
            max_value=500,
            value=180,
            step=5,
        )

        recovery = st.slider(
            "Recovery rate",
            min_value=0.0,
            max_value=0.9,
            value=0.4,
            step=0.05,
        )

        confidence = st.selectbox(
            "PFE confidence",
            options=[0.90, 0.95, 0.99],
            index=1,
        )

        st.form_submit_button(
            "Run scenario",
            type="primary",
            width="stretch",
        )

config = DashboardScenarioConfig(
    n_paths=int(n_paths),
    n_steps=48,
    seed=int(seed),
    mean_reversion=float(mean_reversion),
    volatility=float(volatility),
    notional=float(notional_millions) * 1_000_000.0,
    counterparty_spread_bps=float(counterparty_spread),
    own_spread_bps=float(own_spread),
    recovery_rate=float(recovery),
    confidence_level=float(confidence),
)

with st.spinner("Running Hull–White Monte Carlo and XVA analytics..."):
    result = calculate_scenario(config)

st.caption(
    f"{config.n_paths:,} antithetic paths · "
    "3-year horizon · "
    f"Seed {config.seed} · "
    f"PFE {config.confidence_level:.0%}"
)

st.subheader("Risk overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric("EPE", f"€{result.exposures.epe:,.0f}")
m2.metric("FTD CVA", f"€{result.first_to_default_xva.cva:,.0f}")
m3.metric("FTD DVA", f"€{result.first_to_default_xva.dva:,.0f}")
m4.metric(
    "FTD net XVA",
    f"€{result.first_to_default_xva.net_xva:,.0f}",
)

(
    exposure_tab,
    portfolio_tab,
    xva_tab,
    sensitivity_tab,
    curve_tab,
    hull_white_tab,
    backtesting_tab,
) = st.tabs(
    [
        "Exposure profiles",
        "Portfolio MTM",
        "Bilateral XVA",
        "Credit sensitivity",
        "Yield curve & forwards",
        "Hull–White diagnostics",
        "Backtesting & Model Validation",
    ]
)

with exposure_tab:
    st.plotly_chart(exposure_chart(result), width="stretch")
    st.caption(
        "ENE is shown as a positive exposure magnitude. "
        "PFE is a cross-sectional Monte Carlo percentile."
    )

    st.divider()
    render_exposure_explorer(result, config)

with portfolio_tab:
    st.plotly_chart(trade_chart(result), width="stretch")
    st.write(
        f"Receiver IRS fixed rate: **{result.swap_fixed_rate:.4%}**"
        f"  |  Payer swaption strike: **{result.swaption_fixed_rate:.4%}**"
    )

    st.divider()
    render_portfolio_contributions(result)

with xva_tab:
    st.plotly_chart(xva_chart(result), width="stretch")
    st.caption(
        "Net XVA = DVA − CVA. Standalone CVA/DVA use unconditional "
        "default probabilities; FTD uses independent competing hazards."
    )

    st.divider()
    render_xva_stress_panel(config, result)

with sensitivity_tab:
    metric = st.selectbox(
        "Sensitivity metric",
        options=[
            "first_to_default_net_xva",
            "first_to_default_cva",
            "first_to_default_dva",
            "naive_net_xva",
            "cva_correction",
            "dva_correction",
            "net_xva_correction",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
    )

    st.plotly_chart(
        sensitivity_chart(result, metric),
        width="stretch",
    )

    st.caption(
        "Sensitivity grid uses fixed CDS-spread axes. "
        "The sidebar credit spreads determine the headline XVA values."
    )


# The scenario currently uses a flat 3% initial zero curve.
# Keep this monitoring curve consistent with the scenario runner.
monitoring_curve = SmoothDiscountCurve(
    tenors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    zero_rates=(0.03,) * 6,
)

monitoring = calculate_term_structure_diagnostics(
    monitoring_curve,
    mean_reversion=config.mean_reversion,
    volatility=config.volatility,
)

with curve_tab:
    st.subheader("Initial discount and forward term structures")
    st.caption(
        "The current Monte Carlo scenario uses a flat 3% initial zero curve. "
        "The forward curve and slope are derived from that same curve; "
        "they are not independently calibrated market observations."
    )
    st.plotly_chart(discount_factor_chart(monitoring), width="stretch")
    st.plotly_chart(forward_curve_chart(monitoring), width="stretch")
    st.plotly_chart(forward_slope_chart(monitoring), width="stretch")
    st.caption(
        "Forward slope is the maturity derivative of the instantaneous "
        "forward rate, displayed in basis points per year."
    )

with hull_white_tab:
    st.subheader("Hull–White deterministic drift decomposition")
    st.caption(
        "Theta(t) = forward slope + mean reversion × instantaneous forward "
        "+ volatility contribution. These are model-implied diagnostics, "
        "not calibration residuals."
    )
    st.plotly_chart(hull_white_drift_chart(monitoring), width="stretch")

    st.subheader("Compare alternative model parameters")
    comparison_left, comparison_right = st.columns(2)

    with comparison_left:
        alternative_mean_reversion = st.slider(
            "Alternative mean reversion",
            min_value=0.01,
            max_value=0.30,
            value=0.15,
            step=0.01,
            key="alternative_mean_reversion",
        )

    with comparison_right:
        alternative_volatility = st.slider(
            "Alternative short-rate volatility",
            min_value=0.001,
            max_value=0.030,
            value=0.020,
            step=0.001,
            format="%.3f",
            key="alternative_volatility",
        )

    alternative_monitoring = calculate_term_structure_diagnostics(
        monitoring_curve,
        mean_reversion=float(alternative_mean_reversion),
        volatility=float(alternative_volatility),
    )

    st.plotly_chart(
        parameter_comparison_chart(
            monitoring,
            alternative_monitoring,
            baseline_label="Scenario parameters",
            alternative_label="Alternative parameters",
        ),
        width="stretch",
    )
    st.caption(
        "The initial forward curve remains unchanged when Hull–White "
        "mean reversion or volatility changes. The deterministic drift "
        "changes. Alternative parameters are diagnostic only: the Monte "
        "Carlo paths, exposure, and XVA above continue to use the "
        "submitted sidebar scenario."
    )


with backtesting_tab:
    st.subheader("Backtesting & Model Validation")
    st.warning(
        "SYNTHETIC DEMONSTRATION ONLY — these are generated forecasts "
        "and observations, not historical market data or evidence of "
        "Hull–White, FX, or portfolio model accuracy."
    )

    controls_left, controls_middle, controls_right = st.columns(3)

    with controls_left:
        bt_factor = st.selectbox(
            "Backtesting risk factor",
            options=["EUR short rate", "USD short rate", "EUR/USD FX"],
            key="bt_factor",
        )
        bt_tail = st.selectbox(
            "Backtesting tail",
            options=["upper", "lower"],
            key="bt_tail",
        )

    with controls_middle:
        bt_quantile = st.selectbox(
            "Forecast confidence level",
            options=[0.95, 0.975, 0.99, 0.995],
            index=2,
            format_func=lambda value: f"{value:.1%}",
            key="bt_quantile",
        )
        bt_sample_size = st.selectbox(
            "Number of synthetic observations",
            options=[130, 260, 520],
            index=1,
            key="bt_sample_size",
        )

    with controls_right:
        bt_alternative = st.selectbox(
            "Binomial test alternative",
            options=["greater", "less", "two-sided"],
            key="bt_alternative",
        )
        bt_windows = st.multiselect(
            "Rolling windows (observations)",
            options=[13, 26, 52, 104],
            default=[13, 26, 52],
            key="bt_windows",
        )

    # Independent, reproducible synthetic illustration.
    # These are not outputs from the dashboard's risk-neutral
    # Hull-White/XVA scenario or point-in-time historical forecasts.
    bt_rng = np.random.default_rng(20261118)
    bt_index = np.arange(bt_sample_size, dtype=np.float64)
    bt_location = 0.15 * np.sin(bt_index / 23.0)
    bt_scale = 0.8 + 0.2 * (1.0 + np.cos(bt_index / 31.0))
    bt_standardized = bt_rng.standard_normal(bt_sample_size)

    if bt_factor == "EUR short rate":
        bt_center = 0.025 + 0.002 * bt_location
        bt_sigma = 0.004 * bt_scale
    elif bt_factor == "USD short rate":
        bt_center = 0.035 + 0.003 * bt_location
        bt_sigma = 0.006 * bt_scale
    else:
        bt_center = 1.09 + 0.02 * bt_location
        bt_sigma = 0.025 * bt_scale

    bt_realized = bt_center + bt_sigma * bt_standardized
    bt_probability = 1.0 - bt_quantile
    bt_z = norm.ppf(bt_quantile if bt_tail == "upper" else bt_probability)
    bt_forecasts = bt_center + bt_sigma * bt_z

    # Validate UI selections before passing them to the typed engine API.
    if bt_tail not in ("upper", "lower"):
        raise ValueError(f"Unsupported backtesting tail: {bt_tail}")
    if bt_alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"Unsupported binomial alternative: {bt_alternative}")

    coverage_tail: Literal["upper", "lower"] = (
        "upper" if bt_tail == "upper" else "lower"
    )
    coverage_alternative: Literal["greater", "less", "two-sided"]
    if bt_alternative == "greater":
        coverage_alternative = "greater"
    elif bt_alternative == "less":
        coverage_alternative = "less"
    else:
        coverage_alternative = "two-sided"

    bt_result = test_coverage(
        bt_realized,
        bt_forecasts,
        tail=coverage_tail,
        nominal_probability=bt_probability,
        significance_level=0.05,
        alternative=coverage_alternative,
    )

    st.markdown("#### Bootstrap uncertainty")

    bt_bootstrap_left, bt_bootstrap_middle, bt_bootstrap_right = st.columns(3)

    with bt_bootstrap_left:
        bt_block_length = st.selectbox(
            "Circular block length (observations)",
            options=[1, 5, 10, 20],
            index=2,
            key="bt_block_length",
        )

    with bt_bootstrap_middle:
        bt_resamples = st.selectbox(
            "Bootstrap resamples",
            options=[500, 1000, 2000],
            index=0,
            key="bt_resamples",
        )

    with bt_bootstrap_right:
        bt_ci_level = st.selectbox(
            "Bootstrap confidence level",
            options=[0.90, 0.95, 0.99],
            index=1,
            format_func=lambda value: f"{value:.0%}",
            key="bt_ci_level",
        )

    bt_iid_bootstrap = bootstrap_coverage_interval(
        bt_result.breaches,
        method="iid",
        confidence_level=bt_ci_level,
        n_resamples=bt_resamples,
        seed=20261118,
    )
    bt_block_bootstrap = bootstrap_coverage_interval(
        bt_result.breaches,
        method="circular_block",
        confidence_level=bt_ci_level,
        n_resamples=bt_resamples,
        block_length=bt_block_length,
        seed=20261118,
    )

    bt_m1, bt_m2, bt_m3, bt_m4 = st.columns(4)
    bt_m1.metric("Observed breaches", str(bt_result.n_breaches))
    bt_m2.metric("Expected breaches", f"{bt_result.expected_breaches:.2f}")
    bt_m3.metric(
        "Observed breach frequency",
        f"{bt_result.observed_frequency:.2%}",
    )
    bt_m4.metric("Exact binomial p-value", f"{bt_result.p_value:.4g}")

    st.caption(
        f"Nominal tail probability: {bt_probability:.2%} | "
        "Hypothesis-test significance: 5% | "
        f"Alternative: {bt_alternative} | "
        f"Reject IID coverage null: {bt_result.reject_null}"
    )

    if bt_result.expected_breaches < 5:
        st.info(
            "Low expected breach count: this sample provides limited "
            "information about rare-tail coverage. Failure to reject "
            "does not demonstrate model validity."
        )

    st.plotly_chart(
        breach_timeline_chart(bt_realized, bt_forecasts, bt_result),
        width="stretch",
    )

    if bt_windows:
        st.plotly_chart(
            rolling_coverage_chart(bt_result, windows=tuple(bt_windows)),
            width="stretch",
        )
    else:
        st.info("Select at least one rolling window to show the chart.")

    st.plotly_chart(
        binomial_null_chart(bt_result),
        width="stretch",
    )

    bt_ci_left, bt_ci_right = st.columns(2)

    bt_ci_left.metric(
        "IID bootstrap interval",
        f"{bt_iid_bootstrap.lower_bound:.2%} – {bt_iid_bootstrap.upper_bound:.2%}",
    )
    bt_ci_right.metric(
        "Circular block bootstrap interval",
        f"{bt_block_bootstrap.lower_bound:.2%} – {bt_block_bootstrap.upper_bound:.2%}",
    )

    st.plotly_chart(
        bootstrap_coverage_chart(
            bt_iid_bootstrap,
            bt_block_bootstrap,
            nominal_probability=bt_probability,
        ),
        width="stretch",
    )

    st.caption(
        f"Percentile confidence level: {bt_ci_level:.0%}. "
        f"Circular block length: {bt_block_length} observations. "
        "Dotted interval boundaries show descriptive bootstrap "
        "uncertainty around the observed breach frequency. "
        "The black vertical line is the observed frequency; "
        "the green dashed line is the nominal probability. "
        "These are not null-calibrated bootstrap hypothesis tests."
    )

    if bt_result.n_breaches == 0:
        st.warning(
            "No breaches were observed. Resampling an all-zero "
            "breach series produces a zero-width bootstrap interval; "
            "this does not imply the true breach probability is zero."
        )

    st.info(
        "IID bootstrap assumes independent observations. Circular "
        "block bootstrap preserves local dependence only within "
        "resampled blocks and requires suitable block-length and "
        "weak-dependence assumptions. Neither procedure resolves "
        "overlapping forecast horizons or establishes model validity."
    )

    st.markdown("#### Breach clustering and serial dependence")

    bt_max_lag = st.selectbox(
        "Maximum breach autocorrelation lag",
        options=[1, 5, 10, 20],
        index=2,
        key="bt_max_lag",
    )

    bt_dependence = breach_dependence_diagnostics(
        bt_result.breaches,
        max_lag=bt_max_lag,
    )

    bt_dep_left, bt_dep_middle, bt_dep_right = st.columns(3)

    bt_dep_left.metric(
        "Consecutive breach pairs",
        str(bt_dependence.consecutive_breach_pairs),
    )

    bt_lag_one = bt_dependence.autocorrelations[0]
    bt_dep_middle.metric(
        "Lag-1 breach autocorrelation",
        f"{bt_lag_one:.3f}" if np.isfinite(bt_lag_one) else "Undefined",
    )

    bt_inflation = bt_dependence.variance_inflation_estimate
    bt_dep_right.metric(
        "Descriptive variance inflation",
        f"{bt_inflation:.3f}" if bt_inflation is not None else "Undefined",
    )

    st.plotly_chart(
        breach_autocorrelation_chart(bt_dependence),
        width="stretch",
    )

    bt_transitions = bt_dependence.transition_counts

    st.dataframe(
        {
            "Previous observation": [
                "No breach",
                "Breach",
            ],
            "Next: no breach": [
                int(bt_transitions[0, 0]),
                int(bt_transitions[1, 0]),
            ],
            "Next: breach": [
                int(bt_transitions[0, 1]),
                int(bt_transitions[1, 1]),
            ],
        },
        hide_index=True,
        width="stretch",
    )

    if bt_result.n_breaches < 10:
        st.warning(
            "Fewer than ten observed breaches: estimated serial "
            "correlations and transition frequencies are highly "
            "unstable and should not be treated as strong evidence "
            "for or against independence."
        )

    if not np.isfinite(bt_lag_one):
        st.info(
            "Breach autocorrelation is undefined for a constant "
            "breach sequence, not evidence of zero dependence."
        )

    st.caption(
        "Autocorrelations, transition counts, and variance inflation "
        "are descriptive estimates. They do not provide "
        "dependence-adjusted p-values, formal independence tests, "
        "or a validated effective sample size. The exact binomial "
        "coverage test above still assumes independent breaches. "
        "Overlapping forecast horizons require separate treatment."
    )

    st.markdown("#### Christoffersen conditional-coverage tests")

    bt_christoffersen = christoffersen_coverage_test(
        bt_result.breaches,
        nominal_probability=bt_probability,
    )

    bt_lr_uc, bt_lr_ind, bt_lr_cc = st.columns(3)

    bt_lr_uc.metric(
        "LRuc p-value (asymptotic)",
        f"{bt_christoffersen.p_unconditional:.4g}",
    )

    bt_lr_ind.metric(
        "LRind p-value (asymptotic)",
        (
            f"{bt_christoffersen.p_independence:.4g}"
            if bt_christoffersen.p_independence is not None
            else "Unavailable"
        ),
    )

    bt_lr_cc.metric(
        "LRcc p-value (asymptotic)",
        (
            f"{bt_christoffersen.p_conditional:.4g}"
            if bt_christoffersen.p_conditional is not None
            else "Unavailable"
        ),
    )

    st.dataframe(
        {
            "Test": [
                "Unconditional coverage (LRuc)",
                "First-order independence (LRind)",
                "Conditional coverage (LRcc)",
            ],
            "LR statistic": [
                bt_christoffersen.lr_unconditional,
                bt_christoffersen.lr_independence,
                bt_christoffersen.lr_conditional,
            ],
            "Asymptotic p-value": [
                bt_christoffersen.p_unconditional,
                bt_christoffersen.p_independence,
                bt_christoffersen.p_conditional,
            ],
            "Reference degrees of freedom": [1, 1, 2],
        },
        hide_index=True,
        width="stretch",
    )

    if bt_christoffersen.sparse_transitions:
        st.warning(
            "Sparse breach-transition cells detected. "
            "Christoffersen chi-square p-values may be unreliable "
            "in this sample; treat them as exploratory diagnostics."
        )

    if not bt_christoffersen.independence_identifiable:
        st.info(
            "First-order independence and conditional coverage "
            "cannot be estimated because at least one preceding "
            "breach state has no observed transitions."
        )

    st.caption(
        "Christoffersen LRuc uses all observations; LRind uses "
        "consecutive transitions. LRcc combines their likelihood-"
        "ratio statistics. P-values are asymptotic, not exact, "
        "and assume an appropriate first-order Markov framework. "
        "Failure to reject does not establish calibration or "
        "independence. Overlapping forecast horizons and "
        "small breach counts require additional validation."
    )

    st.caption(
        "Rolling windows count observations, not calendar periods. "
        "The exact binomial p-value assumes independent Bernoulli "
        "breaches with the specified nominal probability. "
        "Overlapping forecast horizons, serial dependence, and "
        "cross-factor dependence require separate diagnostics. "
        "This demonstration uses synthetic Gaussian predictive "
        "distributions and does not establish real-world calibration."
    )


st.divider()
st.caption(
    "Research and demonstration only. "
    "Not intended for production pricing, risk limits, "
    "regulatory capital, or investment decisions."
)
