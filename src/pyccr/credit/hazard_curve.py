"""Constant-hazard counterparty survival probabilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ConstantHazardCurve:
    """Exponential survival curve with a non-negative constant hazard rate."""

    hazard_rate: float

    def __post_init__(self) -> None:
        rate = float(self.hazard_rate)
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("hazard_rate must be finite and non-negative")
        object.__setattr__(self, "hazard_rate", rate)

    def survival_probability(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return unconditional survival probabilities S(t)."""
        t = np.asarray(times, dtype=np.float64)
        if not np.all(np.isfinite(t)) or np.any(t < 0.0):
            raise ValueError("times must be finite and non-negative")
        return np.asarray(np.exp(-self.hazard_rate * t), dtype=np.float64)

    def default_probability(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return cumulative unconditional default probabilities."""
        return np.asarray(
            -np.expm1(-self.hazard_rate * self._validated_times(times)),
            dtype=np.float64,
        )

    @staticmethod
    def _validated_times(times: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(times, dtype=np.float64)
        if not np.all(np.isfinite(t)) or np.any(t < 0.0):
            raise ValueError("times must be finite and non-negative")
        return t

    def marginal_default_probabilities(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return default probabilities over consecutive grid intervals."""
        t = self._validated_times(times)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("times must be a one-dimensional grid")
        if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must start at zero and increase strictly")
        survival = self.survival_probability(t)
        return np.asarray(
            -np.expm1(-self.hazard_rate * np.diff(t)) * survival[:-1],
            dtype=np.float64,
        )
