"""Pathwise single-curve interest-rate swap valuation under Hull-White 1F."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.interest_rate_swap import InterestRateSwap


def value_hull_white_swap_paths(
    swap: InterestRateSwap,
    model: HullWhiteCurveModel,
    times: ArrayLike,
    short_rates: ArrayLike,
) -> NDArray[np.float64]:
    """Value an existing fixed-for-floating swap along short-rate paths.

    Assumptions:
    - Single-curve discounting and floating-rate projection.
    - Floating coupons reset at the start of each accrual period.
    - No payment lag, historical fixings, collateral, or cashflow accumulation.
    - Simulation grid contains every swap reset/payment date.
    - Values at payment times exclude cashflows paid at that time.

    Returns a matrix with shape (n_paths, n_times).
    """
    if not isinstance(swap, InterestRateSwap):
        raise TypeError("swap must be an InterestRateSwap")

    if not isinstance(model, HullWhiteCurveModel):
        raise TypeError("model must be a HullWhiteCurveModel")

    grid = np.asarray(times, dtype=np.float64)
    rates = np.asarray(short_rates, dtype=np.float64)

    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or grid[0] != 0.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("times must start at zero and increase strictly")

    if (
        rates.ndim != 2
        or rates.shape[1] != grid.size
        or rates.shape[0] < 1
        or not np.all(np.isfinite(rates))
    ):
        raise ValueError("short_rates must be a finite (n_paths, n_times) matrix")

    schedule = np.asarray(swap.payment_times, dtype=np.float64)

    if schedule[-1] > model.curve.tenors[-1]:
        raise ValueError("swap maturity exceeds model curve horizon")

    if grid[-1] > model.curve.tenors[-1]:
        raise ValueError("simulation horizon exceeds model curve horizon")

    indices = np.searchsorted(grid, schedule)

    # Avoid indexing outside the simulation grid.
    if np.any(indices >= grid.size):
        raise ValueError("simulation grid must contain all swap schedule dates")

    if not np.array_equal(grid[indices], schedule):
        raise ValueError("simulation grid must contain all swap schedule dates")

    n_paths, n_times = rates.shape
    values = np.zeros((n_paths, n_times), dtype=np.float64)

    accruals = np.diff(schedule)
    starts = schedule[:-1]
    ends = schedule[1:]

    # Determine each coupon at its reset date.
    # This information is used only at valuations on or after reset.
    floating_amounts = np.empty(
        (n_paths, accruals.size),
        dtype=np.float64,
    )

    for period, (start, end) in enumerate(zip(starts, ends, strict=True)):
        reset_index = int(indices[period])

        reset_bond = np.asarray(
            model.bond_price(
                float(start),
                float(end),
                rates[:, reset_index],
            ),
            dtype=np.float64,
        )

        floating_amounts[:, period] = 1.0 / reset_bond - 1.0

    for column, t in enumerate(grid):
        if t >= schedule[-1]:
            continue

        fixed_leg = np.zeros(n_paths, dtype=np.float64)

        floating_leg = np.zeros(n_paths, dtype=np.float64)

        for period, (start, end, accrual) in enumerate(
            zip(starts, ends, accruals, strict=True)
        ):
            # Cashflow already paid.
            if end <= t:
                continue

            payment_bond = np.asarray(
                model.bond_price(
                    float(t),
                    float(end),
                    rates[:, column],
                ),
                dtype=np.float64,
            )

            fixed_leg += swap.fixed_rate * accrual * payment_bond

            if t >= start:
                # The floating rate is already fixed.
                floating_leg += floating_amounts[:, period] * payment_bond
            else:
                # Future reset: use conditional bond prices,
                # not future simulated reset rates.
                reset_bond = np.asarray(
                    model.bond_price(
                        float(t),
                        float(start),
                        rates[:, column],
                    ),
                    dtype=np.float64,
                )

                floating_leg += reset_bond - payment_bond

        payer_value = swap.notional * (floating_leg - fixed_leg)

        values[:, column] = payer_value if swap.pay_fixed else -payer_value

    return values
