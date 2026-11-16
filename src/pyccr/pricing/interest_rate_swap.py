"""Multi-curve fixed-for-floating interest-rate swap pricing."""

from dataclasses import dataclass

import numpy as np

from pyccr.market.curve_set import CurveSet


@dataclass(frozen=True)
class InterestRateSwap:
    """Spot-starting, fixed-for-floating interest-rate swap.

    Assumptions:
    - Payment times are year fractions from valuation date.
    - Fixed and floating payments share the same schedule.
    - Accrual fractions equal differences between payment times.
    - Floating coupons are projected from a pseudo-discount curve.
    - No historical fixings, stubs, payment lags or notionals exchanges.
    """

    currency: str
    floating_index: str
    notional: float
    fixed_rate: float
    payment_times: tuple[float, ...]
    pay_fixed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be a non-empty string")

        if not isinstance(self.floating_index, str) or not self.floating_index.strip():
            raise ValueError("floating_index must be a non-empty string")

        if not np.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("notional must be positive and finite")

        if not np.isfinite(self.fixed_rate):
            raise ValueError("fixed_rate must be finite")

        if not isinstance(self.pay_fixed, bool):
            raise TypeError("pay_fixed must be a boolean")

        times = np.asarray(self.payment_times, dtype=float)

        if times.ndim != 1 or len(times) < 2:
            raise ValueError("payment_times must contain at least two times")

        if not np.all(np.isfinite(times)):
            raise ValueError("payment_times must be finite")

        if times[0] != 0.0:
            raise ValueError("payment_times must start at zero")

        if np.any(np.diff(times) <= 0):
            raise ValueError("payment_times must be strictly increasing")

        object.__setattr__(self, "currency", self.currency.strip().upper())
        object.__setattr__(
            self,
            "floating_index",
            self.floating_index.strip().upper(),
        )
        object.__setattr__(
            self,
            "payment_times",
            tuple(float(t) for t in times),
        )

    def _cashflow_components(
        self,
        curves: CurveSet,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return accrual fractions, discount factors and forwards."""
        discount = curves.discount_curve(self.currency)
        projection = curves.projection_curve(
            self.currency,
            self.floating_index,
        )

        times = np.asarray(self.payment_times)
        starts = times[:-1]
        ends = times[1:]
        accruals = ends - starts

        discount_factors = np.asarray(
            discount.discount_factor(ends),
            dtype=float,
        )

        projection_starts = np.asarray(
            projection.discount_factor(starts),
            dtype=float,
        )
        projection_ends = np.asarray(
            projection.discount_factor(ends),
            dtype=float,
        )

        forwards = (projection_starts / projection_ends - 1.0) / accruals

        return accruals, discount_factors, forwards

    def annuity(self, curves: CurveSet) -> float:
        """Present value of one unit of fixed coupon per unit notional."""
        accruals, discount_factors, _ = self._cashflow_components(curves)
        return float(np.sum(accruals * discount_factors))

    def par_rate(self, curves: CurveSet) -> float:
        """Fixed rate that makes the swap's initial NPV zero."""
        accruals, discount_factors, forwards = self._cashflow_components(curves)
        numerator = np.sum(accruals * discount_factors * forwards)
        denominator = np.sum(accruals * discount_factors)
        return float(numerator / denominator)

    def npv(self, curves: CurveSet) -> float:
        """Swap NPV, positive when favorable to the holder."""
        par = self.par_rate(curves)
        annuity = self.annuity(curves)

        receive_float = self.notional * annuity * (par - self.fixed_rate)

        return receive_float if self.pay_fixed else -receive_float
