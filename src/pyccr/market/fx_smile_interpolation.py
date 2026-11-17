"""Interpolate FX 25-delta smile quotes using total variance.

The resulting surface contains one maturity and three strike nodes.
Interpolation is restricted to the quoted 25-delta wing interval.

Conventions:
    - Black lognormal volatility.
    - Unadjusted forward delta.
    - ATM-forward strike.
    - Volatility butterfly.

No extrapolation or static-arbitrage guarantee is provided.
"""

from pyccr.market.fx_smile import fx_25delta_smile_nodes
from pyccr.market.fx_smile_quotes import FXSmileQuotes
from pyccr.market.fx_vol_surface import FXVolSurface


def fx_25delta_smile_surface(
    *,
    forward: float,
    maturity: float,
    quotes: FXSmileQuotes,
) -> FXVolSurface:
    """Return a single-maturity FX smile with variance interpolation.

    The three nodes are the 25-delta put, ATM-forward, and
    25-delta call. Strike interpolation follows FXVolSurface's
    existing total-implied-variance convention.

    Queries outside the quoted strike interval or at a different
    maturity raise ValueError through FXVolSurface.
    """
    nodes = fx_25delta_smile_nodes(
        forward=forward,
        maturity=maturity,
        quotes=quotes,
    )

    return FXVolSurface(
        maturities=(float(maturity),),
        strikes=nodes.strikes,
        volatilities=(nodes.volatilities,),
    )
