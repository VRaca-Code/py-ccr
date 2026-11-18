"""Illustrative additive CVA/DVA/FVA credit and funding stress panel."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pyccr.credit.cds_calibration import calibrate_flat_hazard_from_cds
from pyccr.dashboard.scenario import DashboardScenarioConfig, DashboardScenarioResult
from pyccr.xva.combined_stress import (
    CombinedXVAStressResult,
    compare_combined_xva_stress,
)


def calculate_dashboard_xva_stress(
    config: DashboardScenarioConfig,
    result: DashboardScenarioResult,
    *,
    borrowing_spread_bps: float,
    lending_spread_bps: float,
    counterparty_hazard_shock_bps: float,
    own_hazard_shock_bps: float,
    borrowing_shock_bps: float,
    lending_shock_bps: float,
) -> CombinedXVAStressResult:
    """Reuse scenario exposures, calibrated hazards and initial discounting."""
    times = result.times
    discount_factors = np.exp(-0.03 * times)

    counterparty_curve = calibrate_flat_hazard_from_cds(
        config.counterparty_spread_bps,
        config.recovery_rate,
    )
    own_curve = calibrate_flat_hazard_from_cds(
        config.own_spread_bps,
        config.recovery_rate,
    )

    return compare_combined_xva_stress(
        times,
        result.exposures.ee,
        result.exposures.ene,
        discount_factors,
        counterparty_curve,
        own_curve,
        borrowing_spread=borrowing_spread_bps / 10_000.0,
        lending_spread=lending_spread_bps / 10_000.0,
        counterparty_hazard_shock_bps=counterparty_hazard_shock_bps,
        own_hazard_shock_bps=own_hazard_shock_bps,
        borrowing_shock_bps=borrowing_shock_bps,
        lending_shock_bps=lending_shock_bps,
        counterparty_recovery_rate=config.recovery_rate,
        own_recovery_rate=config.recovery_rate,
    )


def combined_xva_stress_chart(
    comparison: CombinedXVAStressResult,
) -> go.Figure:
    """Show additive adjustment components before and after stress."""
    names = ["CVA", "DVA", "FCA", "FBA", "FVA", "Net XVA"]
    base = comparison.base
    stressed = comparison.stressed

    figure = go.Figure()
    figure.add_bar(
        name="Base",
        x=names,
        y=[
            base.cva,
            base.dva,
            base.fca,
            base.fba,
            base.fva,
            base.net_xva,
        ],
    )
    figure.add_bar(
        name="Stressed",
        x=names,
        y=[
            stressed.cva,
            stressed.dva,
            stressed.fca,
            stressed.fba,
            stressed.fva,
            stressed.net_xva,
        ],
    )
    figure.update_layout(
        title="Additive XVA: base versus stressed",
        barmode="group",
        yaxis_title="EUR",
        legend_title_text="Scenario",
    )
    return figure


def render_xva_stress_panel(
    config: DashboardScenarioConfig,
    result: DashboardScenarioResult,
) -> None:
    """Render funding assumptions and deterministic credit/funding shocks."""
    st.subheader("Funding & Credit Stress")
    st.caption(
        "Illustrative additive standalone CVA/DVA plus FVA. "
        "This is separate from the first-to-default headline XVA. "
        "Exposure profiles and deterministic discount factors stay fixed. "
        "No collateral, wrong-way risk, correlated defaults, stochastic "
        "funding or legally enforceable netting is modelled."
    )

    with st.expander("Funding and credit stress assumptions", expanded=True):
        left, right = st.columns(2)

        with left:
            borrowing = st.number_input(
                "Base borrowing spread (bps)",
                min_value=0.0,
                max_value=1000.0,
                value=100.0,
                step=10.0,
                key="xva_borrowing_spread",
            )
            lending = st.number_input(
                "Base lending spread (bps)",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                step=10.0,
                key="xva_lending_spread",
            )
            borrowing_shock = st.number_input(
                "Borrowing spread shock (bps)",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=10.0,
                key="xva_borrowing_shock",
            )

        with right:
            cp_shock = st.number_input(
                "Counterparty hazard shock (bps)",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=10.0,
                key="xva_counterparty_hazard_shock",
            )
            own_shock = st.number_input(
                "Own hazard shock (bps)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=10.0,
                key="xva_own_hazard_shock",
            )
            lending_shock = st.number_input(
                "Lending spread shock (bps)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=10.0,
                key="xva_lending_shock",
            )

    comparison = calculate_dashboard_xva_stress(
        config,
        result,
        borrowing_spread_bps=float(borrowing),
        lending_spread_bps=float(lending),
        counterparty_hazard_shock_bps=float(cp_shock),
        own_hazard_shock_bps=float(own_shock),
        borrowing_shock_bps=float(borrowing_shock),
        lending_shock_bps=float(lending_shock),
    )

    a, b, c = st.columns(3)
    a.metric(
        "Base additive net XVA",
        f"€{comparison.base.net_xva:,.0f}",
    )
    b.metric(
        "Stressed additive net XVA",
        f"€{comparison.stressed.net_xva:,.0f}",
    )
    c.metric(
        "Change in additive net XVA",
        f"€{comparison.delta_net_xva:,.0f}",
    )

    st.plotly_chart(
        combined_xva_stress_chart(comparison),
        width="stretch",
    )
    st.caption(
        "Net XVA = DVA − CVA + FBA − FCA. "
        "Positive funding and hazard shocks are absolute basis-point changes. "
        "Funding adjustments use deterministic spreads and endpoint "
        "exposure integration, without default-survival coupling."
    )
