"""Read-only monitoring of PyCCR's existing Hull-White calibration.

The illustrative curve, quotes, and optimizer settings reproduce
examples/hull_white_normal_vol_calibration.py.

This module does not alter the calibration or pricing engines.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pyccr.calibration.hull_white_calibration import (
    SwaptionQuote,
    calibrate_hull_white,
)
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.pricing.bachelier_swaption import (
    bachelier_swaption_price,
    implied_normal_volatility,
)
from pyccr.pricing.hull_white_swaption import (
    hull_white_swaption_price,
)


@lru_cache(maxsize=1)
def calculate_hull_white_calibration_monitor():
    """Reproduce the repository's 16-quote illustrative calibration."""

    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        zero_rates=(0.025, 0.027, 0.029, 0.030, 0.032, 0.034, 0.035, 0.036),
    )

    expiries = (1.0, 2.0, 3.0, 5.0)
    tenors = (2, 3, 5, 7)

    normal_vol_bp = np.array(
        [
            [82, 80, 77, 74],
            [79, 77, 74, 71],
            [76, 74, 71, 68],
            [72, 70, 67, 64],
        ],
        dtype=float,
    )

    quotes = []

    for i, expiry in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            payment_times = tuple(expiry + year for year in range(1, tenor + 1))

            discounts = np.asarray(
                curve.discount_factor(payment_times),
                dtype=float,
            )
            annuity = float(np.sum(discounts))

            forward = (
                float(curve.discount_factor(expiry))
                - float(curve.discount_factor(payment_times[-1]))
            ) / annuity

            quotes.append(
                SwaptionQuote(
                    expiry=expiry,
                    payment_times=payment_times,
                    normal_volatility=normal_vol_bp[i, j] / 10000.0,
                    strike=forward,
                    option_type="payer",
                )
            )

    result = calibrate_hull_white(
        curve=curve,
        quotes=tuple(quotes),
        initial_mean_reversion=0.05,
        initial_volatility=0.01,
    )

    if not result.success:
        raise RuntimeError(f"Hull-White calibration failed: {result.message}")

    records = []

    for quote in quotes:
        times = np.asarray(quote.payment_times, dtype=float)
        accruals = np.diff(np.concatenate(([quote.expiry], times)))

        discounts = np.asarray(
            curve.discount_factor(times),
            dtype=float,
        )
        annuity = float(np.dot(accruals, discounts))

        forward = (
            float(curve.discount_factor(quote.expiry))
            - float(curve.discount_factor(times[-1]))
        ) / annuity

        strike = forward if quote.strike is None else quote.strike

        market_price = bachelier_swaption_price(
            forward=forward,
            strike=strike,
            expiry=quote.expiry,
            normal_volatility=quote.normal_volatility,
            annuity=annuity,
            option_type=quote.option_type,
        )

        model_price = hull_white_swaption_price(
            model=result.model,
            expiry=quote.expiry,
            payment_times=quote.payment_times,
            fixed_rate=strike,
            option_type=quote.option_type,
        )

        model_vol = implied_normal_volatility(
            price=model_price,
            forward=forward,
            strike=strike,
            expiry=quote.expiry,
            annuity=annuity,
            option_type=quote.option_type,
        )

        market_bp = quote.normal_volatility * 10000.0
        model_bp = model_vol * 10000.0

        records.append(
            {
                "Expiry": quote.expiry,
                "Swap tenor": len(times),
                "Market vol (bp)": market_bp,
                "HW implied vol (bp)": model_bp,
                "Vol error (bp)": model_bp - market_bp,
                "Normalized price error": (model_price - market_price) / annuity,
            }
        )

    diagnostics = pd.DataFrame(records)

    normal_vol_rmse_bp = float(
        np.sqrt(np.mean(diagnostics["Vol error (bp)"].to_numpy() ** 2))
    )

    return result, diagnostics, normal_vol_rmse_bp


def render_hull_white_calibration_monitor():
    """Render calibration results without changing dashboard scenarios."""

    st.subheader("Hull–White calibration monitoring")

    st.caption(
        "Illustrative 16-swaption calibration from PyCCR's "
        "existing reproducible example. Not live market quotes. "
        "Uses the unchanged production calibration procedure."
    )

    result, diagnostics, normal_vol_rmse_bp = calculate_hull_white_calibration_monitor()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mean reversion a",
        f"{result.mean_reversion:.8f}",
    )
    c2.metric(
        "Hull–White sigma",
        f"{result.volatility:.8f}",
    )
    c3.metric(
        "Normal-vol RMSE",
        f"{normal_vol_rmse_bp:.4f} bp",
    )
    c4.metric(
        "Swaptions",
        str(len(diagnostics)),
    )

    st.caption(
        "Optimizer objective: annuity-normalized swaption "
        "price residuals. "
        f"Objective RMSE: {result.rmse:.8f}. "
        f"Optimizer status: {'success' if result.success else 'failure'}."
    )

    with st.expander("Optimizer message"):
        st.write(result.message)

    st.markdown("**Market versus model normal volatility**")

    comparison = diagnostics.melt(
        id_vars=["Expiry", "Swap tenor"],
        value_vars=[
            "Market vol (bp)",
            "HW implied vol (bp)",
        ],
        var_name="Series",
        value_name="Normal volatility (bp)",
    )

    fig = px.line(
        comparison,
        x="Expiry",
        y="Normal volatility (bp)",
        color="Series",
        line_dash="Swap tenor",
        markers=True,
        title="Swaption normal-volatility fit by expiry and tenor",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Calibration residuals**")

    residual_fig = px.bar(
        diagnostics,
        x="Expiry",
        y="Vol error (bp)",
        color="Swap tenor",
        barmode="group",
        title="Model minus illustrative quote (normal-vol bp)",
    )
    residual_fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(residual_fig, width="stretch")

    st.markdown("**Volatility fit heatmap**")

    heatmap_data = diagnostics.pivot(
        index="Expiry",
        columns="Swap tenor",
        values="Vol error (bp)",
    )

    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.to_numpy(),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Error (bp)"},
        )
    )
    heatmap.update_layout(
        title="Normal-volatility residuals by expiry and swap tenor",
        xaxis_title="Swap tenor (years)",
        yaxis_title="Expiry (years)",
    )
    st.plotly_chart(heatmap, width="stretch")

    st.markdown("**Hull–White short-rate volatility parameter**")
    st.caption(
        "The calibrated sigma is constant in the existing "
        "one-factor Hull–White model. The varying swaption "
        "normal volatilities above are market/model instrument "
        "volatilities, not a calibrated sigma(t) function."
    )

    st.dataframe(
        diagnostics,
        width="stretch",
        hide_index=True,
    )
