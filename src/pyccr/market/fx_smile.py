"""FX 25-delta volatility smile construction.

Conventions:
    - Black lognormal implied volatility.
    - Unadjusted forward delta.
    - ATM-forward strike (K = F).
    - Volatility butterfly (not market-strangle butterfly).

This module converts one maturity's market quotes into three
strike-volatility nodes. It does not fit a continuous smile.
"""

import math
from dataclasses import dataclass

from pyccr.market.fx_delta_strike import fx_forward_delta_to_strike
from pyccr.market.fx_smile_quotes import FXSmileQuotes


@dataclass(frozen=True)
class FXSmileNodes:
    """Three FX smile nodes ordered by strike."""

    strikes: tuple[float, float, float]
    volatilities: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.strikes) != 3 or len(self.volatilities) != 3:
            raise ValueError("Exactly three smile nodes are required")

        if not all(math.isfinite(k) and k > 0 for k in self.strikes):
            raise ValueError("Smile strikes must be finite and positive")

        if not (self.strikes[0] < self.strikes[1] < self.strikes[2]):
            raise ValueError("Smile strikes must be strictly increasing")

        if not all(math.isfinite(vol) and vol >= 0 for vol in self.volatilities):
            raise ValueError("Smile volatilities must be finite and non-negative")


def fx_25delta_smile_nodes(
    *,
    forward: float,
    maturity: float,
    quotes: FXSmileQuotes,
) -> FXSmileNodes:
    """Construct 25-delta put, ATM-forward and 25-delta call nodes."""
    if not isinstance(quotes, FXSmileQuotes):
        raise TypeError("quotes must be an FXSmileQuotes instance")

    if not math.isfinite(forward) or forward <= 0:
        raise ValueError("Forward must be finite and positive")

    if not math.isfinite(maturity) or maturity <= 0:
        raise ValueError("Maturity must be finite and positive")

    put_vol, call_vol = quotes.wing_volatilities()

    if min(put_vol, call_vol) <= 0:
        raise ValueError("Positive wing volatilities are required for delta inversion")

    put_strike = fx_forward_delta_to_strike(
        forward=forward,
        maturity=maturity,
        volatility=put_vol,
        delta=-0.25,
        option_type="put",
    )

    call_strike = fx_forward_delta_to_strike(
        forward=forward,
        maturity=maturity,
        volatility=call_vol,
        delta=0.25,
        option_type="call",
    )

    return FXSmileNodes(
        strikes=(put_strike, forward, call_strike),
        volatilities=(put_vol, quotes.atm_vol, call_vol),
    )
