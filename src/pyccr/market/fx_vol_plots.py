"""Visualization utilities for FX implied-volatility surfaces."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyccr.market.fx_vol_diagnostics import calendar_variance_violations
from pyccr.market.fx_vol_surface import FXVolSurface


def plot_fx_calendar_variance(
    surface: FXVolSurface,
    *,
    tolerance: float = 1e-12,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot total variance by maturity and mark diagnostic violations.

    Each quoted strike gets a separate line. Red markers identify the
    later maturity of any interval with decreasing total variance.
    This is a diagnostic visualization, not an arbitrage-free guarantee.
    """
    violations = calendar_variance_violations(
        surface,
        tolerance=tolerance,
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    maturities = np.asarray(surface.maturities, dtype=float)
    strikes = np.asarray(surface.strikes, dtype=float)
    vols = np.asarray(surface.volatilities, dtype=float)

    total_variances = maturities[:, None] * vols**2

    for strike_index, strike in enumerate(strikes):
        ax.plot(
            maturities,
            total_variances[:, strike_index],
            marker="o",
            label=f"K={strike:g}",
        )

    for violation in violations:
        ax.scatter(
            violation.later_maturity,
            violation.later_variance,
            color="red",
            marker="x",
            s=80,
            zorder=5,
        )

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Total implied variance")
    ax.set_title("FX volatility surface: calendar-variance diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax
