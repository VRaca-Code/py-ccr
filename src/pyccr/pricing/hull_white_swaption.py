"""European swaptions in the one-factor Hull-White model.

Uses Jamshidian's decomposition under single-curve assumptions.
"""

import numpy as np
from scipy.optimize import brentq

from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.hull_white_bond_option import bond_option_price


def hull_white_swaption_price(
    model: HullWhiteCurveModel,
    expiry: float,
    payment_times: tuple[float, ...],
    fixed_rate: float,
    notional: float = 1.0,
    option_type: str = "payer",
) -> float:
    """Price a European payer or receiver swaption.

    The underlying swap starts at expiry and pays at payment_times.
    The first accrual period begins at expiry.

    Payer swaptions are calls on the swap rate and decompose into
    puts on zero-coupon bonds. Receiver swaptions decompose into calls.
    """
    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")

    if option_type not in ("payer", "receiver"):
        raise ValueError("option_type must be 'payer' or 'receiver'")

    if not np.isfinite(expiry) or expiry < 0:
        raise ValueError("expiry must be non-negative and finite")

    if not np.isfinite(fixed_rate) or fixed_rate < 0:
        raise ValueError("fixed_rate must be non-negative and finite")

    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("notional must be positive and finite")

    times = np.asarray(payment_times, dtype=float)

    if times.ndim != 1 or len(times) == 0:
        raise ValueError("payment_times must be non-empty")

    if not np.all(np.isfinite(times)):
        raise ValueError("payment_times must be finite")

    if times[0] <= expiry or np.any(np.diff(times) <= 0):
        raise ValueError("payment_times must be strictly increasing and after expiry")

    if times[-1] > model.curve.tenors[-1]:
        raise ValueError("payment schedule exceeds curve horizon")

    accruals = np.diff(np.concatenate(([expiry], times)))

    # At expiry, the fixed leg plus redemption is a coupon bond.
    coupons = fixed_rate * accruals
    coupons[-1] += 1.0

    def coupon_bond_value(short_rate: float) -> float:
        prices = np.asarray(
            model.bond_price(expiry, times, short_rate),
            dtype=float,
        )
        return float(np.dot(coupons, prices))

    # Find r* such that the coupon bond is worth par at expiry.
    # Hull-White bond prices are decreasing in the short rate.
    lower, upper = -1.0, 1.0

    while coupon_bond_value(lower) < 1.0:
        lower *= 2.0

    while coupon_bond_value(upper) > 1.0:
        upper *= 2.0

    critical_rate = brentq(
        lambda r: coupon_bond_value(r) - 1.0,
        lower,
        upper,
        xtol=1e-12,
    )

    strikes = np.asarray(
        model.bond_price(expiry, times, critical_rate),
        dtype=float,
    )

    bond_option_type = "put" if option_type == "payer" else "call"

    prices = [
        bond_option_price(
            model=model,
            expiry=expiry,
            maturity=float(maturity),
            strike=float(strike),
            option_type=bond_option_type,
        )
        for maturity, strike in zip(times, strikes, strict=True)
    ]

    return float(notional * np.dot(coupons, prices))
