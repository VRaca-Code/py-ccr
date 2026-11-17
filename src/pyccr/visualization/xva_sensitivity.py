"""Heatmap visualization of bilateral XVA credit-spread sensitivity."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from pyccr.xva.sensitivity import BilateralXVASensitivityResult

XVA_SENSITIVITY_METRICS = (
    "naive_cva",
    "naive_dva",
    "naive_net_xva",
    "first_to_default_cva",
    "first_to_default_dva",
    "first_to_default_net_xva",
    "cva_correction",
    "dva_correction",
    "net_xva_correction",
)

_SIGNED_METRICS = frozenset(
    {
        "naive_net_xva",
        "first_to_default_net_xva",
        "cva_correction",
        "dva_correction",
        "net_xva_correction",
    }
)


def _validate_sensitivity_result(
    result: BilateralXVASensitivityResult,
) -> None:
    """Validate the spread axes and every valuation matrix."""
    if not isinstance(result, BilateralXVASensitivityResult):
        raise TypeError("result must be a BilateralXVASensitivityResult instance")

    cp_spreads = np.asarray(result.counterparty_spreads_bps)
    own_spreads = np.asarray(result.own_spreads_bps)

    for name, spreads in (
        ("counterparty_spreads_bps", cp_spreads),
        ("own_spreads_bps", own_spreads),
    ):
        if spreads.ndim != 1 or spreads.size == 0:
            raise ValueError(f"{name} must be a nonempty 1D array")

        if not np.all(np.isfinite(spreads)) or np.any(spreads < 0):
            raise ValueError(f"{name} must contain finite nonnegative values")

    expected_shape = (own_spreads.size, cp_spreads.size)

    for metric in XVA_SENSITIVITY_METRICS:
        values = np.asarray(getattr(result, metric))

        if values.shape != expected_shape:
            raise ValueError(f"{metric} must have shape {expected_shape}")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{metric} must contain finite values")


def plot_xva_sensitivity_heatmap(
    result: BilateralXVASensitivityResult,
    *,
    metric: str = "first_to_default_net_xva",
    title: str | None = None,
) -> Figure:
    """Plot a bilateral XVA metric across two CDS-spread axes.

    Columns are counterparty CDS spreads (basis points).
    Rows are own CDS spreads (basis points).

    Returns a Matplotlib Figure without displaying or saving it.
    """
    _validate_sensitivity_result(result)

    if not isinstance(metric, str) or metric not in XVA_SENSITIVITY_METRICS:
        raise ValueError("metric must be one of XVA_SENSITIVITY_METRICS")

    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")

    values = np.asarray(getattr(result, metric), dtype=np.float64)

    signed = metric in _SIGNED_METRICS

    if signed:
        bound = float(np.max(np.abs(values)))
        if bound == 0.0:
            bound = 1.0
        color_options = {
            "cmap": "RdBu_r",
            "vmin": -bound,
            "vmax": bound,
            "center": 0.0,
        }
    else:
        color_options = {
            "cmap": "viridis",
            "vmin": 0.0,
        }

    fig_width = max(7.5, 1.05 * values.shape[1] + 3.0)
    fig_height = max(5.0, 0.65 * values.shape[0] + 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        values,
        ax=ax,
        annot=True,
        fmt=",.2f",
        linewidths=0.5,
        linecolor="white",
        xticklabels=[f"{spread:g}" for spread in result.counterparty_spreads_bps],
        yticklabels=[f"{spread:g}" for spread in result.own_spreads_bps],
        cbar_kws={"label": "XVA adjustment (domestic currency)"},
        **color_options,
    )

    ax.set_title(title if title is not None else metric.replace("_", " ").title())
    ax.set_xlabel("Counterparty CDS spread (bps)")
    ax.set_ylabel("Own CDS spread (bps)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig
