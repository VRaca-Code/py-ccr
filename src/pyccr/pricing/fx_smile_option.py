"""European FX option pricing with strike-dependent implied volatility.

Uses the existing Garman-Kohlhagen pricer and FX volatility surface.
No strike or maturity extrapolation is performed.
"""

from pyccr.market.curve_set import CurveSet
from pyccr.market.fx_vol_surface import FXVolSurface
from pyccr.pricing.fx_option import FXOption


def price_fx_option_with_smile(
    *,
    domestic_currency: str,
    foreign_currency: str,
    spot: float,
    strike: float,
    maturity: float,
    option_type: str,
    curves: CurveSet,
    surface: FXVolSurface,
) -> float:
    """Price a European FX option using strike-specific implied volatility.

    The FXVolSurface supplies the volatility at the requested maturity
    and strike. The existing FXOption pricer handles forward calculation,
    discounting and Garman-Kohlhagen valuation.

    Queries outside the quoted volatility domain raise ValueError.
    """
    if not isinstance(surface, FXVolSurface):
        raise TypeError("surface must be an FXVolSurface instance")

    if not isinstance(curves, CurveSet):
        raise TypeError("curves must be a CurveSet instance")

    # Validate option parameters before accessing market volatility.
    # A placeholder volatility is used only for construction.
    option = FXOption(
        domestic_currency=domestic_currency,
        foreign_currency=foreign_currency,
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=0.0,
        option_type=option_type,
    )

    implied_volatility = surface.volatility(
        option.maturity,
        option.strike,
    )

    smile_option = FXOption(
        domestic_currency=option.domestic_currency,
        foreign_currency=option.foreign_currency,
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        volatility=implied_volatility,
        option_type=option.option_type,
    )

    return smile_option.price(curves)
