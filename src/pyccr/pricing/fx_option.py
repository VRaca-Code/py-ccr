"""European FX option pricing under Garman-Kohlhagen."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pyccr.market.curve_set import CurveSet
from pyccr.pricing.fx_forward import FXForward


@dataclass(frozen=True)
class FXOption:
    """European FX option, priced per unit of foreign currency."""

    domestic_currency: str
    foreign_currency: str
    spot: float
    strike: float
    maturity: float
    volatility: float
    option_type: str = "call"

    def __post_init__(self) -> None:
        fx = FXForward(
            self.domestic_currency,
            self.foreign_currency,
            self.spot,
        )

        object.__setattr__(self, "domestic_currency", fx.domestic_currency)
        object.__setattr__(self, "foreign_currency", fx.foreign_currency)

        if not np.isfinite(self.strike) or self.strike <= 0:
            raise ValueError("strike must be positive and finite")

        if not np.isfinite(self.maturity) or self.maturity < 0:
            raise ValueError("maturity must be finite and non-negative")

        if not np.isfinite(self.volatility) or self.volatility < 0:
            raise ValueError("volatility must be finite and non-negative")

        if not isinstance(self.option_type, str):
            raise TypeError("option_type must be call or put")

        option_type = self.option_type.strip().lower()

        if option_type not in ("call", "put"):
            raise ValueError("option_type must be call or put")

        object.__setattr__(self, "option_type", option_type)

    def price(self, curves: CurveSet) -> float:
        """Return discounted option value in domestic currency."""
        fx = FXForward(
            self.domestic_currency,
            self.foreign_currency,
            self.spot,
        )

        forward = fx.forward_rate(curves, self.maturity)
        domestic_df = float(
            curves.discount_curve(self.domestic_currency).discount_factor(self.maturity)
        )

        if self.maturity == 0 or self.volatility == 0:
            intrinsic = (
                max(forward - self.strike, 0.0)
                if self.option_type == "call"
                else max(self.strike - forward, 0.0)
            )
            return float(domestic_df * intrinsic)

        total_vol = self.volatility * np.sqrt(self.maturity)

        d1 = (np.log(forward / self.strike) + 0.5 * total_vol**2) / total_vol
        d2 = d1 - total_vol

        if self.option_type == "call":
            value = domestic_df * (forward * norm.cdf(d1) - self.strike * norm.cdf(d2))
        else:
            value = domestic_df * (
                self.strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
            )

        return float(value)
