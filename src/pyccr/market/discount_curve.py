"""Discount curves using piecewise-linear zero-rate interpolation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DiscountCurve:
    """Continuously compounded zero rates at increasing maturities."""

    tenors: tuple[float, ...]
    zero_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        tenors = np.asarray(self.tenors, dtype=np.float64)
        rates = np.asarray(self.zero_rates, dtype=np.float64)

        if tenors.ndim != 1 or rates.ndim != 1:
            raise ValueError("tenors and zero_rates must be one-dimensional")

        if len(tenors) == 0 or len(tenors) != len(rates):
            raise ValueError("tenors and zero_rates must have equal nonzero length")

        if not np.all(np.isfinite(tenors)):
            raise ValueError("tenors must be finite")

        if not np.all(np.isfinite(rates)):
            raise ValueError("zero_rates must be finite")

        if np.any(tenors <= 0):
            raise ValueError("tenors must be strictly positive")

        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")

        object.__setattr__(self, "tenors", tuple(float(x) for x in tenors))
        object.__setattr__(self, "zero_rates", tuple(float(x) for x in rates))

    def zero_rate(self, maturity: ArrayLike) -> NDArray[np.float64]:
        """Interpolate zero rates with flat endpoint extrapolation."""
        t = np.asarray(maturity, dtype=np.float64)

        if not np.all(np.isfinite(t)) or np.any(t < 0):
            raise ValueError("maturity must be finite and non-negative")

        return np.asarray(
            np.interp(t, self.tenors, self.zero_rates),
            dtype=np.float64,
        )

    def discount_factor(self, maturity: ArrayLike) -> NDArray[np.float64]:
        """Calculate discount factors P(0,T)."""
        t = np.asarray(maturity, dtype=np.float64)
        return np.exp(-self.zero_rate(t) * t)

    def instantaneous_forward(
        self,
        maturity: ArrayLike,
    ) -> NDArray[np.float64]:
        """Calculate forwards using the right derivative at interior knots."""
        t = np.asarray(maturity, dtype=np.float64)

        if not np.all(np.isfinite(t)) or np.any(t < 0):
            raise ValueError("maturity must be finite and non-negative")

        tenors = np.asarray(self.tenors)
        rates = np.asarray(self.zero_rates)

        slopes = np.diff(rates) / np.diff(tenors)
        segment = np.searchsorted(tenors, t, side="right")

        slope = np.zeros_like(t, dtype=np.float64)

        if len(slopes):
            inside = (segment > 0) & (segment < len(tenors))
            slope = np.where(
                inside,
                slopes[np.clip(segment - 1, 0, len(slopes) - 1)],
                0.0,
            )

        return self.zero_rate(t) + t * slope
