
"""Monte Carlo convergence visualization for PyCCR."""

import numpy as np
import plotly.graph_objects as go

from pyccr.convergence.mean import monte_carlo_mean_interval
from app.dashboard_theme import style_figure


def calculate_mc_convergence(result, time_index):
    """Compute running EE estimates from complete antithetic pairs."""
    times = np.asarray(result.times, dtype=float)
    portfolio = np.asarray(result.portfolio_values, dtype=float)

    if times.ndim != 1 or portfolio.ndim != 2:
        raise ValueError("Invalid simulation dimensions")

    if portfolio.shape[1] != len(times):
        raise ValueError("Portfolio time dimension mismatch")

    n_paths = portfolio.shape[0]

    if n_paths < 4 or n_paths % 2:
        raise ValueError(
            "Antithetic convergence requires an even "
            "number of paths, at least four"
        )

    if not isinstance(time_index, (int, np.integer)):
        raise TypeError("time_index must be an integer")

    if not 0 <= time_index < len(times):
        raise IndexError("time_index out of range")

    if not np.all(np.isfinite(portfolio)):
        raise ValueError("Portfolio values must be finite")

    if not np.all(np.isfinite(times)):
        raise ValueError("Times must be finite")

    exposure = np.maximum(portfolio[:, time_index], 0.0)

    half = n_paths // 2

    # PyCCR's antithetic ordering:
    # [Z_1, ..., Z_m, -Z_1, ..., -Z_m]
    pair_means = 0.5 * (
        exposure[:half] + exposure[half:]
    )

    # At least two complete pairs are required
    # for sample-variance estimation.
    checkpoints = np.arange(2, half + 1)

    estimates = np.empty(len(checkpoints))
    lower = np.empty(len(checkpoints))
    upper = np.empty(len(checkpoints))

    for index, count in enumerate(checkpoints):
        observations = np.concatenate((
            exposure[:count],
            exposure[half:half + count],
        ))

        interval = monte_carlo_mean_interval(
            observations,
            confidence_level=0.95,
            antithetic=True,
        )

        estimates[index] = interval.estimate
        lower[index] = interval.lower_bound
        upper[index] = interval.upper_bound

    return {
        "times": times,
        "time_index": int(time_index),
        "selected_time": float(times[time_index]),
        "n_paths": 2 * checkpoints,
        "n_pairs": checkpoints,
        "running_ee": estimates,
        "lower_95": lower,
        "upper_95": upper,
        "reference_ee": float(np.mean(pair_means)),
    }


def monte_carlo_convergence_chart(result, time_index):
    """Plot pair-aware Monte Carlo expected-exposure convergence."""
    data = calculate_mc_convergence(result, time_index)

    x = data["n_paths"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=data["upper_95"],
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=data["lower_95"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.18)",
        line=dict(width=0),
        name="Approx. 95% confidence interval",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=data["running_ee"],
        mode="lines",
        name="Running expected exposure",
        line=dict(color="#38BDF8", width=3),
    ))

    fig.add_hline(
        y=data["reference_ee"],
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Full-sample EE",
        annotation_position="top right",
    )

    fig.update_layout(
        title=(
            "Monte Carlo convergence at "
            f"{data['selected_time']:.2f} years"
        ),
        xaxis_title="Number of simulated paths (complete pairs)",
        yaxis_title="Expected positive exposure (EUR)",
        hovermode="x unified",
    )

    return style_figure(fig)
