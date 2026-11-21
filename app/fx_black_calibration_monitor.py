"""FX Black smile and implied-volatility monitoring.

Uses existing PyCCR FX quote conventions, total-variance
interpolation, Garman-Kohlhagen pricing and implied-volatility
inversion.

All inputs are illustrative and derived from existing test fixtures.
No new calibration optimizer is introduced.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve
from pyccr.market.fx_smile_interpolation import (
    fx_25delta_smile_surface,
)
from pyccr.market.fx_smile_quotes import FXSmileQuotes
from pyccr.pricing.fx_forward import FXForward
from pyccr.pricing.fx_implied_vol import implied_volatility
from pyccr.pricing.fx_option import FXOption
from pyccr.pricing.fx_smile_option import (
    price_fx_option_with_smile,
)


@lru_cache(maxsize=1)
def calculate_fx_black_calibration_monitor():
    """Calculate illustrative FX smile and pricing diagnostics."""

    domestic_currency = "USD"
    foreign_currency = "EUR"
    spot = 1.10
    maturity = 2.0

    curves = CurveSet(
        discount_curves={
            "USD": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(0.04, 0.04),
            ),
            "EUR": DiscountCurve(
                tenors=(1.0, 5.0),
                zero_rates=(0.02, 0.02),
            ),
        },
        projection_curves={},
    )

    quotes = FXSmileQuotes(
        atm_vol=0.12,
        risk_reversal=0.02,
        butterfly=0.005,
    )

    forward = FXForward(
        domestic_currency,
        foreign_currency,
        spot,
    ).forward_rate(curves, maturity)

    surface = fx_25delta_smile_surface(
        forward=forward,
        maturity=maturity,
        quotes=quotes,
    )

    labels = (
        "25Δ put",
        "ATM forward",
        "25Δ call",
    )

    rows = []

    for label, strike, quoted_vol in zip(
        labels,
        surface.strikes,
        surface.volatilities[0],
    ):
        option_type = "put" if label == "25Δ put" else "call"

        option = FXOption(
            domestic_currency=domestic_currency,
            foreign_currency=foreign_currency,
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=quoted_vol,
            option_type=option_type,
        )

        direct_price = option.price(curves)

        smile_price = price_fx_option_with_smile(
            domestic_currency=domestic_currency,
            foreign_currency=foreign_currency,
            spot=spot,
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            curves=curves,
            surface=surface,
        )

        recovered_vol = implied_volatility(
            option,
            curves,
            smile_price,
        )

        rows.append(
            {
                "Node": label,
                "Option type": option_type,
                "Strike": float(strike),
                "Quoted volatility": float(quoted_vol),
                "Surface volatility": float(surface.volatility(maturity, strike)),
                "Direct Black price": float(direct_price),
                "Smile Black price": float(smile_price),
                "Recovered implied volatility": float(recovered_vol),
                "Price difference": float(smile_price - direct_price),
                "Volatility difference": float(recovered_vol - quoted_vol),
            }
        )

    table = pd.DataFrame(rows)

    strikes = np.linspace(
        surface.strikes[0],
        surface.strikes[-1],
        151,
    )

    smile_vols = np.array(
        [surface.volatility(maturity, float(strike)) for strike in strikes]
    )

    assert len(table) == 3
    assert np.all(np.isfinite(smile_vols))
    assert np.all(smile_vols > 0)

    assert np.allclose(
        table["Quoted volatility"],
        table["Surface volatility"],
        atol=1e-12,
        rtol=0,
    )

    assert np.max(np.abs(table["Price difference"])) < 1e-10

    assert np.max(np.abs(table["Volatility difference"])) < 1e-9

    return {
        "spot": spot,
        "forward": float(forward),
        "maturity": maturity,
        "quotes": quotes,
        "surface": surface,
        "table": table,
        "strikes": strikes,
        "smile_vols": smile_vols,
        "max_price_error": float(np.max(np.abs(table["Price difference"]))),
        "max_vol_error_bp": float(
            10000 * np.max(np.abs(table["Volatility difference"]))
        ),
    }


def render_fx_black_calibration_monitor():
    """Render the FX Black smile and pricing checks."""

    result = calculate_fx_black_calibration_monitor()

    st.subheader("FX Black — Smile & Implied Volatility")
    st.caption(
        "Illustrative USD/EUR market quotes from PyCCR test "
        "fixtures. Black/Garman-Kohlhagen pricing; "
        "unadjusted forward delta; ATM-forward strike; "
        "volatility butterfly. No optimizer-based smile fit."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ATM volatility",
        f"{100 * result['quotes'].atm_vol:.2f}%",
    )

    col2.metric(
        "25Δ risk reversal",
        f"{100 * result['quotes'].risk_reversal:.2f}%",
    )

    col3.metric(
        "25Δ butterfly",
        f"{100 * result['quotes'].butterfly:.2f}%",
    )

    col4.metric(
        "2Y FX forward",
        f"{result['forward']:.5f}",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result["strikes"],
            y=100 * result["smile_vols"],
            mode="lines",
            name="Total-variance interpolation",
        )
    )

    table = result["table"]

    fig.add_trace(
        go.Scatter(
            x=table["Strike"],
            y=100 * table["Quoted volatility"],
            mode="markers+text",
            text=table["Node"],
            textposition="top center",
            marker={"size": 10},
            name="Quoted smile nodes",
        )
    )

    fig.update_layout(
        title="FX Black volatility smile — 2Y",
        xaxis_title="FX strike (USD per EUR)",
        yaxis_title="Black implied volatility (%)",
        legend_title="",
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("**Black pricing and implied-volatility checks**")

    display = table.copy()

    for column in [
        "Quoted volatility",
        "Surface volatility",
        "Recovered implied volatility",
    ]:
        display[column] = 100 * display[column]

    display = display.rename(
        columns={
            "Quoted volatility": "Quoted vol (%)",
            "Surface volatility": "Surface vol (%)",
            "Recovered implied volatility": "Recovered vol (%)",
        }
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Maximum pricing difference",
        f"{result['max_price_error']:.2e}",
    )

    c2.metric(
        "Maximum implied-vol difference",
        f"{result['max_vol_error_bp']:.2e} bp",
    )

    st.info(
        "The differences above are numerical consistency "
        "checks. They are not calibration residuals against "
        "independent market option prices."
    )
