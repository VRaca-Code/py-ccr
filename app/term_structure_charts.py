"""Plotly charts for initial term structures and Hull-White diagnostics."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyccr.dashboard.term_structure import TermStructureDiagnostics


def discount_factor_chart(data: TermStructureDiagnostics) -> go.Figure:
    """Display the initial discount factor term structure."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.times,
            y=data.discount_factors,
            mode="lines",
            name="Discount factor",
        )
    )
    fig.update_layout(
        title="Initial discount factor curve",
        xaxis_title="Maturity (years)",
        yaxis_title="Discount factor",
        template="plotly_white",
    )
    return fig


def forward_curve_chart(data: TermStructureDiagnostics) -> go.Figure:
    """Compare zero rates and instantaneous forwards."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.times,
            y=100.0 * data.zero_rates,
            mode="lines",
            name="Zero rate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.times,
            y=100.0 * data.instantaneous_forwards,
            mode="lines",
            name="Instantaneous forward",
        )
    )
    fig.update_layout(
        title="Initial zero and forward curves",
        xaxis_title="Maturity (years)",
        yaxis_title="Rate (%)",
        template="plotly_white",
    )
    return fig


def forward_slope_chart(data: TermStructureDiagnostics) -> go.Figure:
    """Display maturity derivative of the instantaneous forward."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.times,
            y=10000.0 * data.forward_slopes,
            mode="lines",
            name="Forward slope",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(
        title="Instantaneous forward slope",
        xaxis_title="Maturity (years)",
        yaxis_title="Forward slope (bp/year)",
        template="plotly_white",
    )
    return fig


def hull_white_drift_chart(data: TermStructureDiagnostics) -> go.Figure:
    """Show additive contributions to Hull-White theta."""
    fig = go.Figure()
    components = (
        ("Forward slope", data.forward_slopes),
        ("Mean reversion", data.mean_reversion_contribution),
        ("Volatility", data.volatility_contribution),
        ("Total theta", data.theta),
    )
    for label, values in components:
        fig.add_trace(
            go.Scatter(
                x=data.times,
                y=10000.0 * values,
                mode="lines",
                name=label,
            )
        )
    fig.update_layout(
        title="Hull-White deterministic drift decomposition",
        xaxis_title="Maturity (years)",
        yaxis_title="Drift (bp/year)",
        template="plotly_white",
    )
    return fig


def parameter_comparison_chart(
    baseline: TermStructureDiagnostics,
    alternative: TermStructureDiagnostics,
    *,
    baseline_label: str = "Baseline",
    alternative_label: str = "Alternative",
) -> go.Figure:
    """Compare the same initial curve under two Hull-White parameter sets."""
    if baseline.times.shape != alternative.times.shape or not np.array_equal(
        baseline.times, alternative.times
    ):
        raise ValueError("Baseline and alternative must share the same time grid")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Initial instantaneous forward",
            "Hull-White deterministic drift",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=baseline.times,
            y=100.0 * baseline.instantaneous_forwards,
            name=f"{baseline_label} forward",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=baseline.times,
            y=10000.0 * baseline.theta,
            name=f"{baseline_label} theta",
            mode="lines",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=alternative.times,
            y=100.0 * alternative.instantaneous_forwards,
            name=f"{alternative_label} forward",
            mode="lines",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=alternative.times,
            y=10000.0 * alternative.theta,
            name=f"{alternative_label} theta",
            mode="lines",
            line={"dash": "dash"},
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Maturity (years)", row=1, col=1)
    fig.update_xaxes(title_text="Maturity (years)", row=1, col=2)
    fig.update_yaxes(title_text="Forward (%)", row=1, col=1)
    fig.update_yaxes(title_text="Theta (bp/year)", row=1, col=2)
    fig.update_layout(
        title="Hull-White parameter comparison",
        template="plotly_white",
    )
    return fig
