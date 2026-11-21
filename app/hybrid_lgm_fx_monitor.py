"""Hybrid LGM-FX dashboard monitoring.

Reuses PyCCR's existing Hull-White calibration,
FX Black smile construction, and joint FX/Hull-White
correlation diagnostics.

Does not change numerical models or portfolio XVA.
"""

import streamlit as st

from app.correlation_distribution_lab import (
    render_correlation_distribution_lab,
)
from app.fx_black_calibration_monitor import (
    calculate_fx_black_calibration_monitor,
)
from app.hull_white_calibration_monitor import (
    calculate_hull_white_calibration_monitor,
)


def render_hybrid_lgm_fx_monitor(config, curve):
    """Render the existing correlated IR/FX diagnostic workflow."""

    st.subheader("Hybrid LGM–FX — Cross-Asset Monitoring")

    st.caption(
        "Illustrative one-factor Hull–White/LGM interest-rate "
        "model combined with Black lognormal FX dynamics. "
        "The existing PyCCR joint simulator correlates the "
        "Gaussian shocks driving both factors."
    )

    hw_result, _hw_diagnostics, hw_rmse = calculate_hull_white_calibration_monitor()

    fx_result = calculate_fx_black_calibration_monitor()

    st.markdown("**Underlying model inputs**")

    left, middle, right, fourth = st.columns(4)

    left.metric(
        "Calibrated HW mean reversion",
        f"{hw_result.mean_reversion:.6f}",
    )

    middle.metric(
        "Calibrated HW volatility",
        f"{hw_result.volatility:.6f}",
    )

    right.metric(
        "FX ATM Black volatility",
        f"{100 * fx_result['quotes'].atm_vol:.2f}%",
    )

    fourth.metric(
        "HW normal-vol RMSE",
        f"{hw_rmse:.4f} bp",
    )

    st.caption(
        "Hull–White parameters come from the existing "
        "16-swaption illustrative calibration. FX Black "
        "quotes come from the existing illustrative "
        "USD/EUR smile fixture. These are separate "
        "diagnostic inputs, not a joint calibration."
    )

    st.info(
        "Current model scope: the joint simulator uses "
        "correlated Gaussian shocks and a stochastic "
        "domestic Hull–White short rate. FX spot uses "
        "constant domestic and foreign rates in its drift. "
        "It is not yet a fully coupled stochastic-rate "
        "two-curve LGM–FX model."
    )

    st.markdown("**Correlation and distribution diagnostics**")

    st.caption(
        "The interactive lab below uses the dashboard's "
        "existing scenario configuration and monitoring "
        "curve. Its simulation parameters are controlled "
        "independently of the illustrative calibration "
        "results shown above."
    )

    render_correlation_distribution_lab(config, curve)
