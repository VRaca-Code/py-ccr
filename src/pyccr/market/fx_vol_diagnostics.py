"""Diagnostic checks for FX implied-volatility surfaces."""

from dataclasses import dataclass

import numpy as np

from pyccr.market.fx_vol_surface import FXVolSurface


@dataclass(frozen=True)
class CalendarVarianceViolation:
    """A decrease in total implied variance between adjacent maturities."""

    earlier_maturity: float
    later_maturity: float
    strike: float
    earlier_variance: float
    later_variance: float

    @property
    def variance_change(self) -> float:
        """Return later total variance minus earlier total variance."""
        return self.later_variance - self.earlier_variance


def calendar_variance_violations(
    surface: FXVolSurface,
    *,
    tolerance: float = 1e-12,
) -> tuple[CalendarVarianceViolation, ...]:
    """Find decreases in total variance at fixed quoted strikes.

    This is a diagnostic, not a complete static-arbitrage test.
    In particular, FX calendar arbitrage depends on the discount
    curves and the treatment of forward moneyness.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    maturities = np.asarray(surface.maturities, dtype=float)
    strikes = np.asarray(surface.strikes, dtype=float)
    vols = np.asarray(surface.volatilities, dtype=float)

    total_variances = maturities[:, None] * vols**2

    violations = []

    for maturity_index in range(len(maturities) - 1):
        for strike_index, strike in enumerate(strikes):
            earlier = float(total_variances[maturity_index, strike_index])
            later = float(total_variances[maturity_index + 1, strike_index])

            if later - earlier < -tolerance:
                violations.append(
                    CalendarVarianceViolation(
                        earlier_maturity=float(maturities[maturity_index]),
                        later_maturity=float(maturities[maturity_index + 1]),
                        strike=float(strike),
                        earlier_variance=earlier,
                        later_variance=later,
                    )
                )

    return tuple(violations)
