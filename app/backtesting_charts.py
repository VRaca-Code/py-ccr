"""Plotly views of historical quantile-coverage backtests.

This app-layer module is intentionally separate from the pure
pyccr statistical engine. Figures are descriptive diagnostics;
rolling coverage charts are not dependence-adjusted tests.
"""

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from scipy.stats import binom

from pyccr.backtesting.bootstrap import BootstrapCoverageResult
from pyccr.backtesting.coverage import (
    CoverageResult,
    rolling_coverage,
)
from pyccr.backtesting.dependence import BreachDependenceResult


def _aligned_values(
    realized: ArrayLike,
    quantiles: ArrayLike,
    result: CoverageResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Check that chart inputs match the supplied coverage result."""
    actual = np.asarray(realized, dtype=np.float64)
    forecast = np.asarray(quantiles, dtype=np.float64)

    if actual.shape != (result.n_observations,):
        raise ValueError("realized shape does not match coverage result")
    if forecast.shape != actual.shape:
        raise ValueError("quantiles shape does not match realized")
    if not np.all(np.isfinite(actual)):
        raise ValueError("realized values must be finite")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("quantile forecasts must be finite")

    expected = actual > forecast if result.tail == "upper" else actual < forecast
    if not np.array_equal(expected, result.breaches):
        raise ValueError("coverage result does not match chart inputs")

    return actual, forecast


def breach_timeline_chart(
    realized: ArrayLike,
    quantiles: ArrayLike,
    result: CoverageResult,
) -> go.Figure:
    """Plot aligned realizations, forecast thresholds and breaches."""
    actual, forecast = _aligned_values(realized, quantiles, result)
    x = np.arange(result.n_observations)
    flags = result.breaches

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines+markers",
            name="Realized",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name=f"{result.tail.title()} quantile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[flags],
            y=actual[flags],
            mode="markers",
            name="Breaches",
            marker={"color": "crimson", "size": 11, "symbol": "x"},
        )
    )
    fig.update_layout(
        title="Forecast quantiles vs realized observations",
        xaxis_title="Observation index (forecast maturity)",
        yaxis_title="Risk factor / exposure value",
        template="plotly_white",
    )
    return fig


def rolling_coverage_chart(
    result: CoverageResult,
    *,
    windows: Sequence[int] = (13, 26, 52),
) -> go.Figure:
    """Compare rolling breach frequencies across observation windows."""
    if not windows:
        raise ValueError("windows must be nonempty")

    fig = go.Figure()

    for window in windows:
        rolling = rolling_coverage(
            result.breaches,
            window=window,
            nominal_probability=result.nominal_probability,
        )
        fig.add_trace(
            go.Scatter(
                x=rolling.endpoints,
                y=rolling.frequencies,
                mode="lines",
                name=f"{window}-observation window",
            )
        )

    fig.add_hline(
        y=result.nominal_probability,
        line_dash="dash",
        line_color="black",
        annotation_text="Nominal breach probability",
    )
    fig.update_layout(
        title="Rolling breach frequency",
        xaxis_title="Observation index (window endpoint)",
        yaxis_title="Breach frequency",
        yaxis_tickformat=".1%",
        template="plotly_white",
    )
    return fig


def binomial_null_chart(
    result: CoverageResult,
    *,
    max_points: int = 500,
) -> go.Figure:
    """Plot the exact IID binomial null distribution.

    The support is truncated around its central mass for large
    samples; the displayed PMF is not renormalized.
    """
    if max_points < 3:
        raise ValueError("max_points must be at least three")

    n = result.n_observations
    p = result.nominal_probability

    low = int(binom.ppf(1e-6, n, p))
    high = int(binom.ppf(1 - 1e-6, n, p))

    low = min(low, result.n_breaches)
    high = max(high, result.n_breaches)

    if high - low + 1 > max_points:
        raise ValueError(
            "null distribution requires more than max_points; "
            "increase max_points to preserve exact discrete support"
        )

    counts = np.arange(low, high + 1)
    probabilities = binom.pmf(counts, n, p)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts,
            y=probabilities,
            name="IID binomial null PMF",
        )
    )
    fig.add_vline(
        x=result.n_breaches,
        line_color="crimson",
        line_width=2,
        annotation_text="Observed breaches",
    )
    fig.update_layout(
        title=(f"Exact binomial null distribution (p-value = {result.p_value:.4g})"),
        xaxis_title="Number of breaches",
        yaxis_title="Null probability",
        template="plotly_white",
    )
    return fig


def bootstrap_coverage_chart(
    iid: BootstrapCoverageResult,
    block: BootstrapCoverageResult,
    *,
    nominal_probability: float,
) -> go.Figure:
    """Compare descriptive IID and circular-block bootstrap distributions.

    These distributions resample observed breach indicators; they are
    not calibrated under the nominal-coverage null hypothesis.
    """
    if iid.method != "iid" or block.method != "circular_block":
        raise ValueError("expected IID and circular-block bootstrap results")

    if (
        iid.n_observations != block.n_observations
        or iid.n_breaches != block.n_breaches
        or iid.observed_frequency != block.observed_frequency
    ):
        raise ValueError("bootstrap results must describe the same sample")

    if iid.confidence_level != block.confidence_level:
        raise ValueError("bootstrap confidence levels must match")

    if not np.isfinite(nominal_probability) or not (0.0 < nominal_probability < 1.0):
        raise ValueError("nominal_probability must be in (0, 1)")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=iid.resampled_frequencies,
            name="IID bootstrap",
            histnorm="probability",
            opacity=0.55,
            nbinsx=35,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=block.resampled_frequencies,
            name="Circular block bootstrap",
            histnorm="probability",
            opacity=0.55,
            nbinsx=35,
        )
    )

    for result, color in (
        (iid, "#2563eb"),
        (block, "#ea580c"),
    ):
        fig.add_vline(
            x=result.lower_bound,
            line_color=color,
            line_dash="dot",
            line_width=2,
        )
        fig.add_vline(
            x=result.upper_bound,
            line_color=color,
            line_dash="dot",
            line_width=2,
        )

    fig.add_vline(
        x=iid.observed_frequency,
        line_color="black",
        line_width=2,
    )
    fig.add_vline(
        x=nominal_probability,
        line_color="#16a34a",
        line_dash="dash",
        line_width=2,
    )

    fig.update_layout(
        title="Bootstrap distributions of observed breach frequency",
        xaxis_title="Resampled breach frequency",
        xaxis_tickformat=".1%",
        yaxis_title="Relative histogram frequency",
        barmode="overlay",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def breach_autocorrelation_chart(
    result: BreachDependenceResult,
) -> go.Figure:
    """Plot descriptive lagged breach autocorrelations.

    Undefined correlations for constant sequences are omitted.
    The plot does not show hypothesis-test significance bounds.
    """
    fig = go.Figure()
    valid = np.isfinite(result.autocorrelations)

    fig.add_trace(
        go.Bar(
            x=result.lags[valid],
            y=result.autocorrelations[valid],
            name="Sample breach autocorrelation",
        )
    )

    fig.add_hline(y=0.0, line_color="black", line_width=1)

    fig.update_layout(
        title="Breach autocorrelation by observation lag",
        xaxis_title="Lag (observations)",
        yaxis_title="Sample autocorrelation",
        yaxis_range=[-1.1, 1.1],
        template="plotly_white",
    )
    return fig
