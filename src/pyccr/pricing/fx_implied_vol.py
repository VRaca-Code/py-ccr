"""Garman-Kohlhagen European FX implied-volatility solver."""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from pyccr.market.curve_set import CurveSet
from pyccr.pricing.fx_option import FXOption


def implied_volatility(
    option: FXOption,
    curves: CurveSet,
    market_price: float,
    *,
    max_volatility: float = 5.0,
    tolerance: float = 1e-12,
) -> float:
    """Recover annualized FX volatility from a domestic-currency option price.

    Prices are quoted per unit of foreign currency. A finite maximum
    volatility sets the numerical search domain, not a market convention.
    """
    if not np.isfinite(market_price) or market_price < 0:
        raise ValueError("market_price must be finite and non-negative")

    if not np.isfinite(max_volatility) or max_volatility <= 0:
        raise ValueError("max_volatility must be positive and finite")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    if option.maturity == 0:
        raise ValueError("implied volatility is undefined at zero maturity")

    domestic_df = float(
        curves.discount_curve(option.domestic_currency).discount_factor(option.maturity)
    )
    foreign_df = float(
        curves.discount_curve(option.foreign_currency).discount_factor(option.maturity)
    )

    discounted_spot = option.spot * foreign_df
    discounted_strike = option.strike * domestic_df

    if option.option_type == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    else:
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike

    price_scale = max(discounted_spot, discounted_strike)
    price_tolerance = tolerance * price_scale

    if market_price < lower - price_tolerance:
        raise ValueError("market_price is below the no-arbitrage lower bound")

    if market_price > upper + price_tolerance:
        raise ValueError("market_price exceeds the no-arbitrage upper bound")

    if abs(market_price - lower) <= price_tolerance:
        return 0.0

    if market_price >= upper:
        raise ValueError(
            "market_price at the upper bound has no finite implied volatility"
        )

    def price_error(volatility: float) -> float:
        candidate = replace(option, volatility=volatility)
        return candidate.price(curves) - market_price

    upper_error = price_error(max_volatility)

    if upper_error < 0:
        raise ValueError("market_price requires volatility above max_volatility")

    return float(
        brentq(
            price_error,
            0.0,
            max_volatility,
            xtol=tolerance,
            rtol=1e-14,
        )
    )
