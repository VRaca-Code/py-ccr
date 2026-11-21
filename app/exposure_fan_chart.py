"""Exposure percentile fan chart for the PyCCR dashboard."""

import numpy as np
import plotly.graph_objects as go

from app.dashboard_theme import style_figure


def calculate_exposure_fan(result):
    """Calculate positive-exposure quantiles across simulation paths."""
    times = np.asarray(result.times, dtype=float)
    portfolio = np.asarray(result.portfolio_values, dtype=float)

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")

    if portfolio.ndim != 2:
        raise ValueError("portfolio_values must be two-dimensional")

    if portfolio.shape[1] != len(times):
        raise ValueError("Portfolio time dimension does not match times")

    if portfolio.shape[0] == 0 or len(times) == 0:
        raise ValueError("Exposure data must not be empty")

    if not np.all(np.isfinite(times)):
        raise ValueError("Times must be finite")

    if not np.all(np.isfinite(portfolio)):
        raise ValueError("Portfolio values must be finite")

    exposure = np.maximum(portfolio, 0.0)

    return {
        "times": times,
        "ee": np.mean(exposure, axis=0),
        "p25": np.quantile(exposure, 0.25, axis=0),
        "p50": np.quantile(exposure, 0.50, axis=0),
        "p75": np.quantile(exposure, 0.75, axis=0),
        "p95": np.quantile(exposure, 0.95, axis=0),
        "p99": np.quantile(exposure, 0.99, axis=0),
    }


def exposure_fan_chart(result):
    """Plot expected exposure and positive-exposure percentile bands."""
    data = calculate_exposure_fan(result)
    times = data["times"]

    fig = go.Figure()

    # Outer 25th–99th percentile band.
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data["p99"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=data["p25"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.12)",
            line={"width": 0},
            name="P25–P99 band",
            hoverinfo="skip",
        )
    )

    # Inner 50th–95th percentile band.
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data["p95"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=data["p50"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.24)",
            line={"width": 0},
            name="P50–P95 band",
            hoverinfo="skip",
        )
    )

    # Quantile curves.
    for key, label, color, dash in [
        ("p50", "Median (P50)", "#60A5FA", "dot"),
        ("p75", "P75", "#38BDF8", "solid"),
        ("p95", "P95", "#F59E0B", "solid"),
        ("p99", "P99", "#EF4444", "dash"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=data[key],
                mode="lines",
                name=label,
                line={"color": color, "width": 2, "dash": dash},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=data["ee"],
            mode="lines",
            name="Expected exposure (EE)",
            line={"color": "#22C55E", "width": 3},
        )
    )

    fig.update_layout(
        title="Portfolio positive-exposure percentile fan",
        xaxis_title="Time (years)",
        yaxis_title="Positive exposure (EUR)",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )

    return style_figure(fig)
