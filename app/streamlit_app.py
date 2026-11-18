"""Interactive Streamlit dashboard for illustrative PyCCR analytics."""

import plotly.graph_objects as go
import streamlit as st

from app.term_structure_charts import (
    discount_factor_chart,
    forward_curve_chart,
    forward_slope_chart,
    hull_white_drift_chart,
    parameter_comparison_chart,
)
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


@st.cache_data(show_spinner=False)
def calculate_scenario(config: DashboardScenarioConfig):
    """Cache deterministic simulation results for a given configuration."""
    return run_dashboard_scenario(config)


def exposure_chart(result):
    """Plot expected exposures and the selected PFE percentile."""
    fig = go.Figure()

    for name, values, color in (
        ("EE", result.exposures.ee, "#2563eb"),
        ("ENE", result.exposures.ene, "#e8794a"),
        ("PFE", result.exposures.pfe, "#8b5cf6"),
    ):
        fig.add_trace(
            go.Scatter(
                x=result.times,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2.5},
            )
        )

    fig.update_layout(
        title="Expected exposure and potential future exposure",
        xaxis_title="Time (years)",
        yaxis_title="Exposure (EUR)",
        hovermode="x unified",
        template="plotly_white",
        height=430,
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def trade_chart(result):
    """Show expected signed mark-to-market for each trade and portfolio."""
    fig = go.Figure()

    series = (
        ("Receiver IRS", result.trade_values[0].mean(axis=0)),
        ("Payer swaption", result.trade_values[1].mean(axis=0)),
        ("Combined portfolio", result.portfolio_values.mean(axis=0)),
    )

    for name, values in series:
        fig.add_trace(
            go.Scatter(
                x=result.times,
                y=values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Expected signed mark-to-market",
        xaxis_title="Time (years)",
        yaxis_title="Expected MTM (EUR)",
        hovermode="x unified",
        template="plotly_white",
        height=410,
    )
    return fig


def xva_chart(result):
    """Compare standalone and first-to-default XVA components."""
    standalone = result.standalone_xva
    first = result.first_to_default_xva

    fig = go.Figure()

    for label, values in (
        ("Standalone", (standalone.cva, standalone.dva, standalone.net_xva)),
        ("First-to-default", (first.cva, first.dva, first.net_xva)),
    ):
        fig.add_trace(
            go.Bar(
                name=label,
                x=["CVA", "DVA", "Net XVA"],
                y=values,
            )
        )

    fig.update_layout(
        title="Standalone vs first-to-default adjustments",
        barmode="group",
        yaxis_title="EUR",
        template="plotly_white",
        height=400,
    )
    return fig


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
            **color_settings,
        )
    )

    fig.update_layout(
        title=metric.replace("_", " ").title(),
        xaxis_title="Counterparty CDS spread (bps)",
        yaxis_title="Own CDS spread (bps)",
        template="plotly_white",
        height=480,
    )
    return fig


st.title("PyCCR")
st.caption("Multi-Asset Counterparty Credit Risk & XVA Engine")

st.info(
    "Illustrative EUR receiver IRS + payer swaption portfolio. "
    "Signed aggregation is not legal netting. "
    "The model excludes collateral, wrong-way risk, and correlated defaults."
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
            use_container_width=True,
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

exposure_tab, portfolio_tab, xva_tab, sensitivity_tab, curve_tab, hull_white_tab = (
    st.tabs(
        [
            "Exposure profiles",
            "Portfolio MTM",
            "Bilateral XVA",
            "Credit sensitivity",
            "Yield curve & forwards",
            "Hull–White diagnostics",
        ]
    )
)

with exposure_tab:
    st.plotly_chart(exposure_chart(result), use_container_width=True)
    st.caption(
        "ENE is shown as a positive exposure magnitude. "
        "PFE is a cross-sectional Monte Carlo percentile."
    )

with portfolio_tab:
    st.plotly_chart(trade_chart(result), use_container_width=True)
    st.write(
        f"Receiver IRS fixed rate: **{result.swap_fixed_rate:.4%}**"
        f"  |  Payer swaption strike: **{result.swaption_fixed_rate:.4%}**"
    )

with xva_tab:
    st.plotly_chart(xva_chart(result), use_container_width=True)
    st.caption(
        "Net XVA = DVA − CVA. Standalone CVA/DVA use unconditional "
        "default probabilities; FTD uses independent competing hazards."
    )

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
        use_container_width=True,
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
    st.plotly_chart(discount_factor_chart(monitoring), use_container_width=True)
    st.plotly_chart(forward_curve_chart(monitoring), use_container_width=True)
    st.plotly_chart(forward_slope_chart(monitoring), use_container_width=True)
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
    st.plotly_chart(hull_white_drift_chart(monitoring), use_container_width=True)

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
        use_container_width=True,
    )
    st.caption(
        "The initial forward curve remains unchanged when Hull–White "
        "mean reversion or volatility changes. The deterministic drift "
        "changes. Alternative parameters are diagnostic only: the Monte "
        "Carlo paths, exposure, and XVA above continue to use the "
        "submitted sidebar scenario."
    )

st.divider()
st.caption(
    "Research and demonstration only. "
    "Not intended for production pricing, risk limits, "
    "regulatory capital, or investment decisions."
)
