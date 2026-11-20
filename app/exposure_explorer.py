"""Interactive Monte Carlo exposure distribution explorer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.dashboard_theme import CHART_COLORS, style_figure


def exposure_distribution_chart(
    exposures: np.ndarray,
    *,
    time_years: float,
    confidence_level: float,
) -> go.Figure:
    """Plot positive exposure across Monte Carlo paths at one date."""
    values = np.asarray(exposures, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("exposures must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("exposures must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("exposures must be nonnegative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    ee = float(values.mean())
    pfe = float(np.quantile(values, confidence_level))

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=45,
            name="Positive exposure",
            marker_color=CHART_COLORS["ee"],
            opacity=0.85,
            hovertemplate=("Exposure: %{x:,.2f} EUR<br>Paths: %{y}<extra></extra>"),
        )
    )

    fig.add_vline(
        x=ee,
        line_color=CHART_COLORS["ene"],
        line_width=2.5,
        line_dash="dash",
    )

    fig.add_vline(
        x=pfe,
        line_color=CHART_COLORS["pfe"],
        line_width=2.5,
        line_dash="solid",
    )

    fig.update_layout(
        xaxis_title="Positive exposure (EUR)",
        yaxis_title="Monte Carlo path count",
        xaxis_tickformat=",.0f",
        bargap=0.04,
        showlegend=False,
    )

    return style_figure(
        fig,
        title=(
            f"Exposure distribution at year {time_years:.2f} "
            f"| PFE {confidence_level:.0%}"
        ),
        height=440,
    )


def render_exposure_explorer(result, config) -> None:
    """Render an interactive cross-sectional exposure analysis."""
    st.subheader("Exposure Distribution Explorer")
    st.caption(
        "Inspect the distribution of simulated positive portfolio MTM "
        "at a selected simulation date. This is illustrative signed "
        "trade aggregation, not legally enforceable netting."
    )

    times = np.asarray(result.times, dtype=float)
    portfolio = np.asarray(result.portfolio_values, dtype=float)

    if times.ndim != 1 or times.size == 0:
        raise ValueError("Simulation times must be a nonempty 1D array")
    if portfolio.ndim != 2 or portfolio.shape[1] != times.size:
        raise ValueError("Expected portfolio_values shape (n_paths, n_times)")
    if not np.all(np.isfinite(portfolio)):
        raise ValueError("Portfolio paths contain nonfinite values")

    default_index = int(np.argmin(np.abs(times - 1.5)))

    left, right = st.columns(2)

    with left:
        index = st.select_slider(
            "Simulation date",
            options=list(range(times.size)),
            value=default_index,
            format_func=lambda i: f"Year {times[i]:.2f}",
            key="exposure_explorer_time",
        )

    with right:
        confidence = st.select_slider(
            "PFE confidence level",
            options=[0.90, 0.95, 0.975, 0.99],
            value=min(
                [0.90, 0.95, 0.975, 0.99],
                key=lambda level: abs(level - config.confidence_level),
            ),
            format_func=lambda level: f"{level:.1%}",
            key="exposure_explorer_confidence",
        )

    positive_exposure = np.maximum(portfolio[:, index], 0.0)

    ee = float(np.mean(positive_exposure))
    pfe = float(np.quantile(positive_exposure, confidence))
    positive_share = float(np.mean(positive_exposure > 0.0))

    col_ee, col_pfe, col_share = st.columns(3)

    col_ee.metric("Expected Exposure", f"EUR {ee:,.0f}")
    col_pfe.metric(
        f"PFE ({confidence:.1%})",
        f"EUR {pfe:,.0f}",
    )
    col_share.metric(
        "Positive-path share",
        f"{positive_share:.1%}",
    )

    st.plotly_chart(
        exposure_distribution_chart(
            positive_exposure,
            time_years=float(times[index]),
            confidence_level=float(confidence),
        ),
        width="stretch",
    )

    st.caption(
        f"Based on {positive_exposure.size:,} Monte Carlo paths. "
        "EE is the mean positive exposure; PFE is the selected "
        "cross-sectional percentile. No collateral, wrong-way risk, "
        "or correlated defaults are modeled in this scenario."
    )
