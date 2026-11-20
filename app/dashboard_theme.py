"""Shared presentation settings for the PyCCR Streamlit dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

CHART_COLORS = {
    "ee": "#14B8A6",
    "ene": "#F59E0B",
    "pfe": "#3B82F6",
    "cva": "#EF4444",
    "dva": "#22C55E",
    "xva": "#A78BFA",
}


def apply_dashboard_theme() -> None:
    """Apply lightweight visual styling to the Streamlit dashboard."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        div[data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 0.75rem;
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        div[data-testid="stTabs"] button {
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_figure(
    figure: go.Figure,
    *,
    title: str | None = None,
    height: int = 430,
) -> go.Figure:
    """Apply consistent Plotly presentation settings."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 40, "r": 30, "t": 65, "b": 45},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )

    if title is not None:
        figure.update_layout(title=title)

    return figure
