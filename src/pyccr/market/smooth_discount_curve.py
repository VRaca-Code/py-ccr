"""Smooth continuously compounded zero-rate discount curve."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class SmoothDiscountCurve:
    """Natural cubic-spline interpolation of zero rates."""

    tenors: tuple[float, ...]
    zero_rates: tuple[float, ...]
    _spline: CubicSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        times = np.asarray(self.tenors, dtype=float)
        rates = np.asarray(self.zero_rates, dtype=float)

        if times.ndim != 1 or rates.ndim != 1:
            raise ValueError("tenors and zero_rates must be one-dimensional")
        if len(times) < 2 or times.shape != rates.shape:
            raise ValueError("At least two matching curve nodes are required")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(rates)):
            raise ValueError("Curve inputs must be finite")
        if times[0] != 0.0 or np.any(np.diff(times) <= 0):
            raise ValueError("Tenors must start at zero and increase strictly")

        object.__setattr__(self, "tenors", tuple(float(x) for x in times))
        object.__setattr__(self, "zero_rates", tuple(float(x) for x in rates))
        object.__setattr__(
            self,
            "_spline",
            CubicSpline(times, rates, bc_type="natural", extrapolate=False),
        )

    def _time(self, maturity: ArrayLike) -> np.ndarray:
        t = np.asarray(maturity, dtype=float)
        if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > self.tenors[-1]):
            raise ValueError("Maturity must lie within the curve domain")
        return t

    def zero_rate(self, maturity: ArrayLike) -> np.ndarray:
        t = self._time(maturity)
        return np.asarray(self._spline(t), dtype=float)

    def discount_factor(self, maturity: ArrayLike) -> np.ndarray:
        t = self._time(maturity)
        return np.exp(-t * self.zero_rate(t))

    def instantaneous_forward(self, maturity: ArrayLike) -> np.ndarray:
        t = self._time(maturity)
        return self.zero_rate(t) + t * self._spline(t, 1)

    def forward_derivative(self, maturity: ArrayLike) -> np.ndarray:
        t = self._time(maturity)
        return 2.0 * self._spline(t, 1) + t * self._spline(t, 2)
