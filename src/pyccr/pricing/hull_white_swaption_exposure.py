"""Pathwise European swaption valuation under single-curve Hull-White 1F."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from pyccr.models.hull_white_curve import HullWhiteCurveModel


@dataclass(frozen=True)
class HullWhiteSwaptionTrade:
    """Physically settled European swaption on a forward-starting IRS."""

    expiry: float
    payment_times: tuple[float, ...]
    fixed_rate: float
    notional: float = 1.0
    option_type: str = "payer"

    def __post_init__(self) -> None:
        if not np.isfinite(self.expiry) or self.expiry <= 0:
            raise ValueError("expiry must be positive and finite")

        if not np.isfinite(self.fixed_rate) or self.fixed_rate < 0:
            raise ValueError("fixed_rate must be non-negative and finite")

        if not np.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("notional must be positive and finite")

        if self.option_type not in ("payer", "receiver"):
            raise ValueError("option_type must be 'payer' or 'receiver'")

        schedule = np.asarray(self.payment_times, dtype=float)

        if (
            schedule.ndim != 1
            or schedule.size == 0
            or not np.all(np.isfinite(schedule))
            or schedule[0] <= self.expiry
            or np.any(np.diff(schedule) <= 0)
        ):
            raise ValueError("payment_times must be strictly increasing after expiry")

        object.__setattr__(
            self,
            "payment_times",
            tuple(float(t) for t in schedule),
        )


def value_hull_white_swaption_paths(
    trade: HullWhiteSwaptionTrade,
    model: HullWhiteCurveModel,
    times: ArrayLike,
    short_rates: ArrayLike,
) -> NDArray[np.float64]:
    """Value a physically settled European swaption along HW paths.

    The simulation grid must contain expiry and all swap payment dates.
    Pre-expiry values are conditional option values. At expiry, exercise
    is determined by intrinsic value. An exercised option subsequently
    carries the signed MTM of the underlying swap.

    Values exclude cashflows already settled at the valuation time.
    """
    if not isinstance(trade, HullWhiteSwaptionTrade):
        raise TypeError("trade must be a HullWhiteSwaptionTrade")

    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")

    grid = np.asarray(times, dtype=float)
    rates = np.asarray(short_rates, dtype=float)

    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or grid[0] != 0.0
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("times must start at zero and increase strictly")

    if (
        rates.ndim != 2
        or rates.shape[0] < 1
        or rates.shape[1] != grid.size
        or not np.all(np.isfinite(rates))
    ):
        raise ValueError("short_rates must be a finite (n_paths, n_times) matrix")

    schedule = np.asarray(
        (trade.expiry, *trade.payment_times),
        dtype=float,
    )

    if schedule[-1] > model.curve.tenors[-1]:
        raise ValueError("swaption schedule exceeds model curve horizon")

    if grid[-1] > model.curve.tenors[-1]:
        raise ValueError("simulation horizon exceeds model curve horizon")

    indices = np.searchsorted(grid, schedule)

    if np.any(indices >= grid.size) or not np.array_equal(grid[indices], schedule):
        raise ValueError("simulation grid must contain expiry and all payment dates")

    n_paths, n_times = rates.shape
    expiry_index = int(indices[0])
    expiry = float(trade.expiry)

    payments = np.asarray(trade.payment_times, dtype=float)
    accruals = np.diff(schedule)

    # Jamshidian coupon-bond representation:
    # underlying payer swap = 1 - sum_i c_i P(expiry, T_i).
    coupons = trade.fixed_rate * accruals
    coupons[-1] += 1.0

    def coupon_bond_value(rate: float) -> float:
        bonds = np.asarray(
            model.bond_price(expiry, payments, rate),
            dtype=float,
        )
        return float(np.dot(coupons, bonds))

    lower, upper = -1.0, 1.0

    while coupon_bond_value(lower) < 1.0:
        lower *= 2.0

    while coupon_bond_value(upper) > 1.0:
        upper *= 2.0

    critical_rate = brentq(
        lambda rate: coupon_bond_value(rate) - 1.0,
        lower,
        upper,
        xtol=1e-12,
    )

    strikes = np.asarray(
        model.bond_price(expiry, payments, critical_rate),
        dtype=float,
    )

    values = np.zeros((n_paths, n_times), dtype=float)

    a = model.mean_reversion
    sigma = model.volatility

    # Conditional Jamshidian decomposition before expiry.
    for column in range(expiry_index):
        t = float(grid[column])
        tau = expiry - t

        discount_expiry = np.asarray(
            model.bond_price(t, expiry, rates[:, column]),
            dtype=float,
        )

        remaining_variance = -np.expm1(-2.0 * a * tau) / (2.0 * a)

        for coupon, maturity, strike in zip(coupons, payments, strikes, strict=True):
            discount_maturity = np.asarray(
                model.bond_price(t, float(maturity), rates[:, column]),
                dtype=float,
            )

            forward_bond = discount_maturity / discount_expiry

            b = -np.expm1(-a * (float(maturity) - expiry)) / a

            vol = sigma * b * np.sqrt(remaining_variance)

            if vol == 0.0:
                if trade.option_type == "payer":
                    payoff = np.maximum(strike - forward_bond, 0.0)
                else:
                    payoff = np.maximum(forward_bond - strike, 0.0)

                option_value = discount_expiry * payoff

            else:
                d1 = np.log(forward_bond / strike) / vol + 0.5 * vol
                d2 = d1 - vol

                if trade.option_type == "payer":
                    option_value = strike * discount_expiry * norm.cdf(
                        -d2
                    ) - discount_maturity * norm.cdf(-d1)
                else:
                    option_value = discount_maturity * norm.cdf(
                        d1
                    ) - strike * discount_expiry * norm.cdf(d2)

            values[:, column] += coupon * option_value

    values[:, :expiry_index] *= trade.notional

    # Value the underlying forward-starting swap from expiry onward.
    # Floating coupons are fixed at their actual reset dates.
    swap_values = np.zeros_like(values)

    for column in range(expiry_index, n_times):
        t = float(grid[column])

        if t >= payments[-1]:
            continue

        fixed_leg = np.zeros(n_paths, dtype=float)
        floating_leg = np.zeros(n_paths, dtype=float)

        for period, (start, end, accrual) in enumerate(
            zip(
                schedule[:-1],
                schedule[1:],
                accruals,
                strict=True,
            )
        ):
            if end <= t:
                continue

            payment_bond = np.asarray(
                model.bond_price(t, float(end), rates[:, column]),
                dtype=float,
            )

            fixed_leg += trade.fixed_rate * accrual * payment_bond

            if t >= start:
                reset_index = int(indices[period])
                reset_bond = np.asarray(
                    model.bond_price(
                        float(start),
                        float(end),
                        rates[:, reset_index],
                    ),
                    dtype=float,
                )
                floating_leg += (1.0 / reset_bond - 1.0) * payment_bond
            else:
                reset_bond = np.asarray(
                    model.bond_price(t, float(start), rates[:, column]),
                    dtype=float,
                )
                floating_leg += reset_bond - payment_bond

        payer_mtm = trade.notional * (floating_leg - fixed_leg)

        swap_values[:, column] = (
            payer_mtm if trade.option_type == "payer" else -payer_mtm
        )

    # Exercise decision is made once at expiry, without look-ahead.
    exercised = swap_values[:, expiry_index] > 0.0

    values[:, expiry_index:] = np.where(
        exercised[:, None],
        swap_values[:, expiry_index:],
        0.0,
    )

    return values
