"""European swaption pricing with the Bachelier normal model."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bachelier_swaption_price(
    forward: float,
    strike: float,
    expiry: float,
    normal_volatility: float,
    annuity: float,
    option_type: str = "payer",
    notional: float = 1.0,
) -> float:
    """Price a European swaption using absolute normal volatility."""
    values = np.asarray(
        [forward, strike, expiry, normal_volatility, annuity, notional],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("All numeric inputs must be finite")
    if expiry < 0:
        raise ValueError("expiry must be non-negative")
    if normal_volatility < 0:
        raise ValueError("normal_volatility must be non-negative")
    if annuity <= 0 or notional <= 0:
        raise ValueError("annuity and notional must be positive")
    if option_type not in ("payer", "receiver"):
        raise ValueError("option_type must be 'payer' or 'receiver'")

    direction = 1.0 if option_type == "payer" else -1.0
    intrinsic = direction * (forward - strike)
    standard_deviation = normal_volatility * np.sqrt(expiry)

    if standard_deviation == 0:
        return float(notional * annuity * max(intrinsic, 0.0))

    d = intrinsic / standard_deviation
    price_per_annuity = intrinsic * norm.cdf(d) + standard_deviation * norm.pdf(d)

    return float(notional * annuity * price_per_annuity)


def implied_normal_volatility(
    price: float,
    forward: float,
    strike: float,
    expiry: float,
    annuity: float,
    option_type: str = "payer",
    notional: float = 1.0,
) -> float:
    """Recover Bachelier normal volatility in absolute rate units."""
    values = np.asarray(
        [price, forward, strike, expiry, annuity, notional],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("All numeric inputs must be finite")
    if price < 0 or expiry <= 0:
        raise ValueError("price must be non-negative and expiry positive")
    if annuity <= 0 or notional <= 0:
        raise ValueError("annuity and notional must be positive")
    if option_type not in ("payer", "receiver"):
        raise ValueError("Invalid option_type")

    intrinsic = bachelier_swaption_price(
        forward=forward,
        strike=strike,
        expiry=expiry,
        normal_volatility=0.0,
        annuity=annuity,
        option_type=option_type,
        notional=notional,
    )

    tolerance = 1e-12 * max(1.0, abs(price), abs(intrinsic))

    if price < intrinsic - tolerance:
        raise ValueError("Price is below intrinsic value")

    if abs(price - intrinsic) <= tolerance:
        return 0.0

    def objective(volatility: float) -> float:
        return (
            bachelier_swaption_price(
                forward=forward,
                strike=strike,
                expiry=expiry,
                normal_volatility=volatility,
                annuity=annuity,
                option_type=option_type,
                notional=notional,
            )
            - price
        )

    upper = 0.01

    for _ in range(60):
        if objective(upper) >= 0:
            return float(brentq(objective, 0.0, upper, xtol=1e-14))
        upper *= 2.0

    raise RuntimeError("Could not bracket implied normal volatility")
