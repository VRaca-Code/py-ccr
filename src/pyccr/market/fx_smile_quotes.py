"""FX volatility smile quote conventions.

Supports the volatility-butterfly convention, not market-strangle
or premium-adjusted delta conventions.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FXSmileQuotes:
    """ATM, risk reversal and volatility butterfly at one maturity."""

    atm_vol: float
    risk_reversal: float
    butterfly: float

    def __post_init__(self) -> None:
        values = (
            self.atm_vol,
            self.risk_reversal,
            self.butterfly,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Smile quotes must be finite")

        if self.atm_vol < 0:
            raise ValueError("ATM volatility must be non-negative")

        put_vol, call_vol = self.wing_volatilities()

        if put_vol < 0 or call_vol < 0:
            raise ValueError("Wing volatilities must be non-negative")

    def wing_volatilities(self) -> tuple[float, float]:
        """Return (put_vol, call_vol) for the quoted delta wings."""
        common = self.atm_vol + self.butterfly

        put_vol = common - 0.5 * self.risk_reversal
        call_vol = common + 0.5 * self.risk_reversal

        return put_vol, call_vol

    @classmethod
    def from_wing_volatilities(
        cls,
        *,
        atm_vol: float,
        put_vol: float,
        call_vol: float,
    ) -> "FXSmileQuotes":
        """Convert ATM and wing volatilities to ATM/RR/BF quotes."""
        if not all(math.isfinite(value) for value in (atm_vol, put_vol, call_vol)):
            raise ValueError("Volatilities must be finite")

        if min(atm_vol, put_vol, call_vol) < 0:
            raise ValueError("Volatilities must be non-negative")

        return cls(
            atm_vol=atm_vol,
            risk_reversal=call_vol - put_vol,
            butterfly=0.5 * (call_vol + put_vol) - atm_vol,
        )
