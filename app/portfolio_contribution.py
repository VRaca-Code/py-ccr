"""Illustrative pathwise portfolio exposure decomposition."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.dashboard_theme import CHART_COLORS, style_figure


def calculate_portfolio_contributions(result):
    """Calculate standalone and aggregated positive exposure profiles."""
    times = np.asarray(result.times, dtype=float)
    trades = np.asarray(result.trade_values, dtype=float)
    portfolio = np.asarray(result.portfolio_values, dtype=float)

    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty 1D array")
    if trades.ndim != 3 or trades.shape[0] != 2:
        raise ValueError("Expected two trade valuation arrays")
    if portfolio.ndim != 2 or portfolio.shape != trades.shape[1:]:
        raise ValueError("Portfolio and trade path dimensions do not match")
    if portfolio.shape[1] != times.size:
        raise ValueError("Portfolio time dimension does not match times")
    if portfolio.shape[0] == 0:
        raise ValueError("At least one simulation path is required")
    if not (
        np.all(np.isfinite(times))
        and np.all(np.isfinite(trades))
        and np.all(np.isfinite(portfolio))
    ):
        raise ValueError("Scenario arrays must be finite")

    if not np.allclose(
        trades.sum(axis=0),
        portfolio,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError("Trade paths do not reconstruct portfolio paths")

    standalone_ee = np.maximum(trades, 0.0).mean(axis=1)
    portfolio_ee = np.maximum(portfolio, 0.0).mean(axis=0)

    standalone_sum = standalone_ee.sum(axis=0)
    offset = standalone_sum - portfolio_ee

    if np.any(offset < -1e-7):
        raise ValueError("Unexpected negative exposure offset")

    offset = np.maximum(offset, 0.0)

    return {
        "times": times,
        "standalone_ee": standalone_ee,
        "standalone_sum": standalone_sum,
        "portfolio_ee": portfolio_ee,
        "offset": offset,
    }


def portfolio_contribution_chart(contributions) -> go.Figure:
    """Visualize positive exposure and the illustrative offset."""
    times = contributions["times"]
    standalone = contributions["standalone_ee"]

    fig = go.Figure()

    for name, values, color in (
        ("Receiver IRS standalone EE", standalone[0], CHART_COLORS["ee"]),
        ("Payer swaption standalone EE", standalone[1], CHART_COLORS["pfe"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                name=name,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=(
                    "Year %{x:.2f}"
                    "<br>Exposure: %{y:,.2f} EUR"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=contributions["standalone_sum"],
            name="Sum of standalone EE",
            mode="lines",
            line={
                "color": CHART_COLORS["ene"],
                "width": 2.5,
                "dash": "dash",
            },
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=contributions["portfolio_ee"],
            name="Portfolio EE",
            mode="lines",
            line={"color": CHART_COLORS["xva"], "width": 3.5},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=contributions["offset"],
            name="Illustrative offset",
            mode="lines",
            line={"color": CHART_COLORS["cva"], "width": 2.5},
        )
    )

    fig.update_layout(
        xaxis_title="Time (years)",
        yaxis_title="Expected positive exposure (EUR)",
        yaxis_tickformat=",.0f",
    )

    return style_figure(
        fig,
        title="Standalone and portfolio exposure decomposition",
        height=470,
    )


def render_portfolio_contributions(result) -> None:
    """Render trade-level exposure comparison and pathwise offset."""
    st.subheader("Portfolio Exposure Decomposition")
    st.caption(
        "Compare standalone trade exposure with the positive exposure "
        "of the signed aggregated portfolio. The difference is an "
        "illustrative offset, not legally enforceable netting."
    )

    data = calculate_portfolio_contributions(result)

    times = data["times"]
    default_index = int(np.argmin(np.abs(times - 1.5)))

    index = st.select_slider(
        "Decomposition simulation date",
        options=list(range(times.size)),
        value=default_index,
        format_func=lambda i: f"Year {times[i]:.2f}",
        key="portfolio_contribution_time",
    )

    irs_ee = float(data["standalone_ee"][0, index])
    swaption_ee = float(data["standalone_ee"][1, index])
    portfolio_ee = float(data["portfolio_ee"][index])
    offset = float(data["offset"][index])
    standalone_sum = float(data["standalone_sum"][index])

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("Receiver IRS standalone EE", f"EUR {irs_ee:,.0f}")
    col2.metric("Payer swaption standalone EE", f"EUR {swaption_ee:,.0f}")
    col3.metric("Portfolio EE", f"EUR {portfolio_ee:,.0f}")
    col4.metric("Illustrative exposure offset", f"EUR {offset:,.0f}")

    st.caption(
        f"Year {times[index]:.2f}: standalone EE sum "
        f"EUR {standalone_sum:,.2f} = portfolio EE "
        f"EUR {portfolio_ee:,.2f} + illustrative offset "
        f"EUR {offset:,.2f}."
    )

    st.plotly_chart(
        portfolio_contribution_chart(data),
        width="stretch",
    )

    st.caption(
        "Trade-level positive exposures are calculated before "
        "aggregation. Portfolio positive exposure is calculated "
        "after signed trade aggregation, using the same Monte Carlo "
        "paths. No legal netting, collateral, or regulatory capital "
        "benefit is inferred."
    )
