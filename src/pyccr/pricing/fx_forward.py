"""FX forward pricing from domestic and foreign discount curves."""

from dataclasses import dataclass

import numpy as np

from pyccr.market.curve_set import CurveSet


@dataclass(frozen=True)
class FXForward:
    """FX forward with spot quoted in domestic currency per foreign unit."""

    domestic_currency: str
    foreign_currency: str
    spot: float

    def __post_init__(self) -> None:
        for field_name in ("domestic_currency", "foreign_currency"):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

            object.__setattr__(self, field_name, value.strip().upper())

        if self.domestic_currency == self.foreign_currency:
            raise ValueError("domestic and foreign currencies must differ")

        if not np.isfinite(self.spot) or self.spot <= 0:
            raise ValueError("spot must be positive and finite")

    def forward_rate(self, curves: CurveSet, maturity: float) -> float:
        """Return the no-basis FX forward rate at maturity."""
        if not np.isfinite(maturity) or maturity < 0:
            raise ValueError("maturity must be finite and non-negative")

        domestic = curves.discount_curve(self.domestic_currency)
        foreign = curves.discount_curve(self.foreign_currency)

        domestic_df = float(domestic.discount_factor(maturity))
        foreign_df = float(foreign.discount_factor(maturity))

        return self.spot * foreign_df / domestic_df
