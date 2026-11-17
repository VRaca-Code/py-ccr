"""Pathwise mark-to-market valuation of FX forward contracts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FXForwardExposureTrade:
    """FX forward receiving foreign and paying domestic currency.

    Spot and strike are domestic currency per foreign unit.
    A positive foreign notional means receiving foreign currency.
    A negative foreign notional reverses the trade direction.
    """

    foreign_notional: float
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        for name in ("foreign_notional", "strike", "maturity"):
            value = getattr(self, name)

            if not np.isscalar(value):
                raise ValueError(f"{name} must be a finite scalar")

            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite scalar") from exc

            if not np.isfinite(number):
                raise ValueError(f"{name} must be finite")

            object.__setattr__(self, name, number)

        if self.foreign_notional == 0.0:
            raise ValueError("foreign_notional must be nonzero")

        if self.strike <= 0.0:
            raise ValueError("strike must be positive")

        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative")


def value_fx_forward_paths(
    trade: FXForwardExposureTrade,
    times: ArrayLike,
    fx_spots: ArrayLike,
    *,
    domestic_rate: float,
    foreign_rate: float,
) -> NDArray[np.float64]:
    """Return pathwise domestic-currency FX-forward values.

    Inputs:
        times: shape (n_times,), strictly increasing from zero.
        fx_spots: shape (n_paths, n_times).
        domestic_rate: constant continuously compounded rate.
        foreign_rate: constant continuously compounded rate.

    The contract is valued before maturity using remaining-time
    domestic and foreign discount factors. Values at and after
    maturity are zero, excluding settled cashflows.

    Returns:
        Array with shape (n_paths, n_times).
    """
    if not isinstance(trade, FXForwardExposureTrade):
        raise TypeError("trade must be an FXForwardExposureTrade")

    for name, value in (
        ("domestic_rate", domestic_rate),
        ("foreign_rate", foreign_rate),
    ):
        if not np.isscalar(value):
            raise ValueError(f"{name} must be a finite scalar")

        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc

        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")

    rd = float(domestic_rate)
    rf = float(foreign_rate)

    t = np.asarray(times, dtype=np.float64)
    spots = np.asarray(fx_spots, dtype=np.float64)

    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a nonempty 1D array")

    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")

    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and strictly increase")

    if spots.ndim != 2 or spots.shape[1] != t.size:
        raise ValueError("fx_spots must have shape (n_paths, n_times)")

    if spots.shape[0] == 0:
        raise ValueError("fx_spots must contain at least one path")

    if not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
        raise ValueError("fx_spots must be positive and finite")

    remaining = np.maximum(trade.maturity - t, 0.0)

    domestic_df = np.exp(-rd * remaining)
    foreign_df = np.exp(-rf * remaining)

    values = trade.foreign_notional * (
        spots * foreign_df[None, :] - trade.strike * domestic_df[None, :]
    )

    # Exposure stops after contractual settlement.
    values[:, t >= trade.maturity] = 0.0

    return np.asarray(values, dtype=np.float64)
