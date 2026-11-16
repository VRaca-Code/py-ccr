"""Analytical zero-coupon bond options in Hull-White 1F."""

import numpy as np
from scipy.stats import norm

from pyccr.models.hull_white_curve import HullWhiteCurveModel


def bond_option_volatility(
    model: HullWhiteCurveModel,
    expiry: float,
    maturity: float,
) -> float:
    """Effective lognormal volatility of the bond forward price."""
    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")

    expiry = float(expiry)
    maturity = float(maturity)

    if not np.isfinite(expiry) or not np.isfinite(maturity):
        raise ValueError("expiry and maturity must be finite")

    if expiry < 0 or maturity <= expiry:
        raise ValueError("require 0 <= expiry < maturity")

    if maturity > model.curve.tenors[-1]:
        raise ValueError("maturity exceeds curve horizon")

    a = model.mean_reversion
    sigma = model.volatility

    b = -np.expm1(-a * (maturity - expiry)) / a
    variance = -np.expm1(-2.0 * a * expiry) / (2.0 * a)

    return float(sigma * b * np.sqrt(variance))


def bond_option_price(
    model: HullWhiteCurveModel,
    expiry: float,
    maturity: float,
    strike: float,
    option_type: str = "call",
) -> float:
    """Price a European call or put on a zero-coupon bond."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be positive and finite")

    v = bond_option_volatility(model, expiry, maturity)

    p_s = float(model.curve.discount_factor(expiry))
    p_t = float(model.curve.discount_factor(maturity))

    forward = p_t / p_s

    if v == 0.0:
        intrinsic = (
            max(forward - strike, 0.0)
            if option_type == "call"
            else max(strike - forward, 0.0)
        )
        return float(p_s * intrinsic)

    d1 = np.log(forward / strike) / v + 0.5 * v
    d2 = d1 - v

    if option_type == "call":
        return float(p_t * norm.cdf(d1) - strike * p_s * norm.cdf(d2))

    return float(strike * p_s * norm.cdf(-d2) - p_t * norm.cdf(-d1))
