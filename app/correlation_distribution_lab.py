"""Interactive correlation and marginal/joint distribution diagnostics.

Uses existing PyCCR simulation functions. These diagnostics do not
alter the primary portfolio scenario or its XVA calculations.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.simulation.config import SimulationConfig
from pyccr.simulation.correlation import (
    cholesky_factor,
    correlate_normals,
)
from pyccr.simulation.joint_fx_hw_paths import (
    simulate_joint_fx_hull_white_paths,
)
from pyccr.simulation.time_grid import TimeGrid


def calculate_correlation_diagnostics(
    *,
    curve,
    mean_reversion: float,
    rate_volatility: float,
    correlation: float,
    fx_volatility: float,
    n_paths: int,
    seed: int,
):
    """Return matrices, Gaussian diagnostics, and joint model paths."""

    target = np.array(
        [[1.0, correlation], [correlation, 1.0]],
        dtype=np.float64,
    )

    factor = cholesky_factor(target)
    reconstructed = factor @ factor.T

    # Separate Gaussian diagnostic sample.
    # Not the exact innovations used by the path simulator.
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((n_paths, 2))
    correlated = correlate_normals(independent, target)

    empirical = np.corrcoef(correlated, rowvar=False)

    simulation_config = SimulationConfig(
        time_grid=TimeGrid.uniform(1.0, 24),
        n_paths=n_paths,
        seed=seed,
        antithetic=False,
    )

    model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=mean_reversion,
        volatility=rate_volatility,
    )

    paths = simulate_joint_fx_hull_white_paths(
        simulation_config,
        model,
        initial_spot=1.10,
        domestic_rate=0.03,
        foreign_rate=0.02,
        fx_volatility=fx_volatility,
        correlation=correlation,
    )

    fx_terminal = np.asarray(paths.fx_spots[:, -1])
    rate_terminal = np.asarray(paths.short_rates[:, -1])

    log_fx_terminal = np.log(fx_terminal)

    terminal_logfx_rate_correlation = float(
        np.corrcoef(log_fx_terminal, rate_terminal)[0, 1]
    )

    return {
        "target": target,
        "cholesky": factor,
        "reconstructed": reconstructed,
        "empirical": empirical,
        "fx_terminal": fx_terminal,
        "rate_terminal": rate_terminal,
        "terminal_correlation": terminal_logfx_rate_correlation,
    }


def correlation_heatmap(matrix, title, *, zmin=-1.0, zmax=1.0):
    """Plot a labeled 2x2 matrix."""

    labels = ["HW shock", "FX shock"]

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=zmin,
            zmax=zmax,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(matrix, 4),
            texttemplate="%{text:.4f}",
            hovertemplate="%{x} / %{y}: %{z:.4f}<extra></extra>",
            colorbar={"title": "Value"},
        )
    )

    figure.update_layout(
        title=title,
        height=330,
        margin={"l": 25, "r": 25, "t": 55, "b": 25},
    )

    return figure


def marginal_histogram(values, title, axis_label):
    """Plot an empirical terminal marginal distribution."""

    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=55,
            marker_color="#48A6C9",
            opacity=0.85,
        )
    )

    figure.update_layout(
        title=title,
        xaxis_title=axis_label,
        yaxis_title="Path count",
        height=350,
        margin={"l": 25, "r": 25, "t": 55, "b": 40},
        bargap=0.05,
    )

    return figure


def joint_distribution_chart(fx, rates):
    """Plot joint terminal FX spot and short-rate samples."""

    figure = go.Figure(
        go.Scattergl(
            x=fx,
            y=100.0 * rates,
            mode="markers",
            marker={
                "size": 4,
                "opacity": 0.35,
                "color": "#48A6C9",
            },
            name="Simulated paths",
        )
    )

    figure.update_layout(
        title="Joint terminal distribution: FX spot vs short rate",
        xaxis_title="FX spot (domestic currency per foreign unit)",
        yaxis_title="Hull–White short rate (%)",
        height=430,
        margin={"l": 25, "r": 25, "t": 55, "b": 45},
    )

    return figure


def render_correlation_distribution_lab(config, curve):
    """Render optional diagnostics inside the Hull–White tab."""

    with st.expander(
        "Correlation & Distribution Lab — Hybrid IR/FX",
        expanded=False,
    ):
        st.caption(
            "Independent diagnostic simulation. Changing these controls "
            "does not reprice the portfolio or change the headline XVA."
        )

        left, middle, right = st.columns(3)

        with left:
            rho = st.slider(
                "HW / FX Gaussian shock correlation",
                min_value=-0.95,
                max_value=0.95,
                value=0.45,
                step=0.05,
                key="lab_ir_fx_correlation",
            )

        with middle:
            fx_vol = st.slider(
                "FX annual volatility",
                min_value=0.05,
                max_value=0.40,
                value=0.15,
                step=0.01,
                key="lab_fx_volatility",
            )

        with right:
            paths_count = st.selectbox(
                "Diagnostic paths",
                options=[1024, 2048, 4096, 8192],
                index=2,
                key="lab_path_count",
            )

        diagnostics = calculate_correlation_diagnostics(
            curve=curve,
            mean_reversion=float(config.mean_reversion),
            rate_volatility=float(config.volatility),
            correlation=float(rho),
            fx_volatility=float(fx_vol),
            n_paths=int(paths_count),
            seed=int(config.seed),
        )

        st.subheader("Correlation matrix and Cholesky decomposition")

        st.write("Target correlation matrix")
        st.dataframe(
            diagnostics["target"],
            width="stretch",
            hide_index=True,
        )

        st.write("Lower-triangular Cholesky factor")
        st.dataframe(
            diagnostics["cholesky"],
            width="stretch",
            hide_index=True,
        )

        reconstruction_error = float(
            np.max(np.abs(diagnostics["reconstructed"] - diagnostics["target"]))
        )

        st.metric(
            "Maximum Cholesky reconstruction error",
            f"{reconstruction_error:.2e}",
        )

        st.subheader("Target vs empirical Gaussian correlation")

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.plotly_chart(
                correlation_heatmap(
                    diagnostics["target"],
                    "Target shock correlation",
                ),
                width="stretch",
            )

        with chart_right:
            st.plotly_chart(
                correlation_heatmap(
                    diagnostics["empirical"],
                    "Empirical shock correlation",
                ),
                width="stretch",
            )

        st.metric(
            "Maximum empirical correlation error",
            f"{np.max(np.abs(diagnostics['empirical'] - diagnostics['target'])):.4f}",
        )

        st.caption(
            "Empirical shock correlation is estimated from a separate "
            "Gaussian diagnostic sample. It is not extracted from the "
            "joint simulator's internal random innovations."
        )

        st.subheader("Marginal terminal distributions")

        marginal_left, marginal_right = st.columns(2)

        with marginal_left:
            st.plotly_chart(
                marginal_histogram(
                    diagnostics["fx_terminal"],
                    "FX spot at 1 year",
                    "FX spot",
                ),
                width="stretch",
            )

        with marginal_right:
            st.plotly_chart(
                marginal_histogram(
                    100.0 * diagnostics["rate_terminal"],
                    "Hull–White short rate at 1 year",
                    "Short rate (%)",
                ),
                width="stretch",
            )

        st.subheader("Joint terminal distribution")

        st.plotly_chart(
            joint_distribution_chart(
                diagnostics["fx_terminal"],
                diagnostics["rate_terminal"],
            ),
            width="stretch",
        )

        st.metric(
            "Terminal log-FX / short-rate correlation",
            f"{diagnostics['terminal_correlation']:+.4f}",
        )

        st.info(
            "Model interpretation: the specified correlation applies "
            "to Gaussian step innovations. Terminal log-FX/short-rate "
            "correlation generally differs because the Hull–White "
            "factor mean-reverts. The current joint simulator uses "
            "constant domestic and foreign rates in FX drift; it is "
            "not a fully stochastic-rate-coupled hybrid pricing model. "
            "These simulated market-factor distributions are not "
            "portfolio exposure distributions, calibration results, "
            "or model validation evidence."
        )
