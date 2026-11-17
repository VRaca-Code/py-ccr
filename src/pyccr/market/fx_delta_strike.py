"""FX strike conversion using unadjusted forward delta.

Delta conventions:
    call: N(d1), between 0 and 1
    put:  N(d1) - 1, between -1 and 0

Spot delta and premium-adjusted delta are not supported here.
"""

import math
from typing import Literal

from scipy.stats import norm


def fx_forward_delta_to_strike(
    *,
    forward: float,
    maturity: float,
    volatility: float,
    delta: float,
    option_type: Literal["call", "put"],
) -> float:
    """Convert signed unadjusted FX forward delta to strike.

    Args:
        forward: Positive FX forward exchange rate.
        maturity: Positive time to expiry in years.
        volatility: Positive annualized Black implied volatility.
        delta: Signed forward delta; calls positive, puts negative.
        option_type: Either "call" or "put".

    Returns:
        Positive strike corresponding to the specified forward delta.

    Raises:
        ValueError: If inputs or delta conventions are invalid.
        OverflowError: If the resulting strike is not representable.
    """
    values = (forward, maturity, volatility, delta)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Inputs must be finite")

    if forward <= 0:
        raise ValueError("Forward must be positive")

    if maturity <= 0:
        raise ValueError("Maturity must be positive")

    if volatility <= 0:
        raise ValueError("Volatility must be positive")

    if option_type == "call":
        if not 0 < delta < 1:
            raise ValueError("Call forward delta must be between 0 and 1")
        probability = delta
    elif option_type == "put":
        if not -1 < delta < 0:
            raise ValueError("Put forward delta must be between -1 and 0")
        probability = 1.0 + delta
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if not 0 < probability < 1:
        raise ValueError("Delta is too close to a numerical boundary")

    sigma_sqrt_t = volatility * math.sqrt(maturity)

    if not math.isfinite(sigma_sqrt_t):
        raise ValueError("Volatility-time product is not finite")

    d1 = norm.ppf(probability)

    exponent = -d1 * sigma_sqrt_t + 0.5 * sigma_sqrt_t**2

    if not math.isfinite(exponent):
        raise OverflowError("Strike exponent is not finite")

    try:
        strike = forward * math.exp(exponent)
    except OverflowError as exc:
        raise OverflowError("Strike exceeds floating-point range") from exc

    if not math.isfinite(strike) or strike <= 0:
        raise OverflowError("Strike is outside representable positive range")

    return strike
