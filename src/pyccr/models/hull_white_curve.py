"""Hull-White one-factor bond pricing with a non-flat initial curve."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve


@dataclass(frozen=True)
class HullWhiteCurveModel:
    """One-factor Hull-White model fitted to an initial discount curve."""

    curve: SmoothDiscountCurve
    mean_reversion: float
    volatility: float

    def __post_init__(self) -> None:
        if not isinstance(self.curve, SmoothDiscountCurve):
            raise TypeError("curve must be a SmoothDiscountCurve")

        if not np.isfinite(self.mean_reversion) or self.mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive and finite")

        if not np.isfinite(self.volatility) or self.volatility < 0:
            raise ValueError("volatility must be non-negative and finite")

    def b_factor(
        self,
        t: float,
        maturity: ArrayLike,
    ) -> NDArray[np.float64]:
        """Calculate B(t,T)."""
        maturities = np.asarray(maturity, dtype=float)
        self._validate_times(t, maturities)

        a = self.mean_reversion
        return -np.expm1(-a * (maturities - t)) / a

    def variance_adjustment(
        self,
        t: float,
        maturity: ArrayLike,
    ) -> NDArray[np.float64]:
        """Calculate the convexity term in log A(t,T)."""
        b = self.b_factor(t, maturity)
        a = self.mean_reversion

        return self.volatility**2 * (-np.expm1(-2.0 * a * t)) * b**2 / (4.0 * a)

    def a_factor(
        self,
        t: float,
        maturity: ArrayLike,
    ) -> NDArray[np.float64]:
        """Calculate A(t,T) using the initial discount curve."""
        maturities = np.asarray(maturity, dtype=float)
        b = self.b_factor(t, maturities)

        initial_ratio = self.curve.discount_factor(
            maturities
        ) / self.curve.discount_factor(t)

        return initial_ratio * np.exp(
            b * self.curve.instantaneous_forward(t)
            - self.variance_adjustment(t, maturities)
        )

    def bond_price(
        self,
        t: float,
        maturity: ArrayLike,
        short_rate: ArrayLike,
    ) -> NDArray[np.float64]:
        """Calculate P(t,T) for a given short rate."""
        rates = np.asarray(short_rate, dtype=float)

        if not np.all(np.isfinite(rates)):
            raise ValueError("short_rate must be finite")

        return self.a_factor(t, maturity) * np.exp(-self.b_factor(t, maturity) * rates)

    def theta(self, t: ArrayLike) -> NDArray[np.float64]:
        """Calculate the deterministic Hull-White drift."""
        times = np.asarray(t, dtype=float)
        a = self.mean_reversion

        return (
            self.curve.forward_derivative(times)
            + a * self.curve.instantaneous_forward(times)
            + self.volatility**2 * (-np.expm1(-2.0 * a * times)) / (2.0 * a)
        )

    def _validate_times(
        self,
        t: float,
        maturities: NDArray[np.float64],
    ) -> None:
        """Validate valuation and maturity times."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("valuation time must be non-negative and finite")

        if not np.all(np.isfinite(maturities)):
            raise ValueError("maturities must be finite")

        if np.any(maturities < t):
            raise ValueError("maturity must not precede valuation time")

        if np.any(maturities > self.curve.tenors[-1]):
            raise ValueError("maturity exceeds curve horizon")

        if t > self.curve.tenors[-1]:
            raise ValueError("valuation time exceeds curve horizon")
