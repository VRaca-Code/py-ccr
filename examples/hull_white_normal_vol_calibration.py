"""Reproduce the illustrative Hull-White normal-volatility calibration."""

from pathlib import Path

import numpy as np
import pandas as pd

from pyccr.calibration.hull_white_calibration import (
    SwaptionQuote,
    calibrate_hull_white,
)
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.pricing.bachelier_swaption import implied_normal_volatility
from pyccr.pricing.hull_white_swaption import hull_white_swaption_price


def main() -> None:
    """Calibrate to 16 illustrative swaption normal-volatility quotes."""
    curve = SmoothDiscountCurve(
        tenors=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        zero_rates=(0.025, 0.027, 0.029, 0.030, 0.032, 0.034, 0.035, 0.036),
    )

    expiries = (1.0, 2.0, 3.0, 5.0)
    tenors = (2, 3, 5, 7)

    normal_vol_bp = np.array(
        [
            [82, 80, 77, 74],
            [79, 77, 74, 71],
            [76, 74, 71, 68],
            [72, 70, 67, 64],
        ],
        dtype=float,
    )

    quotes = []

    for i, expiry in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            payment_times = tuple(expiry + year for year in range(1, tenor + 1))

            discounts = np.asarray(curve.discount_factor(payment_times), dtype=float)
            annuity = float(np.sum(discounts))

            forward = (
                float(curve.discount_factor(expiry))
                - float(curve.discount_factor(payment_times[-1]))
            ) / annuity

            quotes.append(
                SwaptionQuote(
                    expiry=expiry,
                    payment_times=payment_times,
                    normal_volatility=normal_vol_bp[i, j] / 10000.0,
                    strike=forward,
                    option_type="payer",
                )
            )

    result = calibrate_hull_white(
        curve=curve,
        quotes=tuple(quotes),
        initial_mean_reversion=0.05,
        initial_volatility=0.01,
    )

    if not result.success:
        raise RuntimeError(f"Calibration failed: {result.message}")

    records = []

    for quote in quotes:
        times = np.asarray(quote.payment_times, dtype=float)
        accruals = np.diff(np.concatenate(([quote.expiry], times)))

        discounts = np.asarray(curve.discount_factor(times), dtype=float)
        annuity = float(np.dot(accruals, discounts))

        forward = (
            float(curve.discount_factor(quote.expiry))
            - float(curve.discount_factor(times[-1]))
        ) / annuity

        strike = forward if quote.strike is None else quote.strike

        model_price = hull_white_swaption_price(
            model=result.model,
            expiry=quote.expiry,
            payment_times=quote.payment_times,
            fixed_rate=strike,
            option_type=quote.option_type,
        )

        model_vol = implied_normal_volatility(
            price=model_price,
            forward=forward,
            strike=strike,
            expiry=quote.expiry,
            annuity=annuity,
            option_type=quote.option_type,
        )

        market_bp = quote.normal_volatility * 10000.0
        model_bp = model_vol * 10000.0

        records.append(
            {
                "Expiry": quote.expiry,
                "Swap tenor": len(times),
                "Market vol (bp)": market_bp,
                "HW implied vol (bp)": model_bp,
                "Vol error (bp)": model_bp - market_bp,
            }
        )

    diagnostics = pd.DataFrame(records)

    rmse = float(np.sqrt(np.mean(diagnostics["Vol error (bp)"].to_numpy() ** 2)))

    reference_path = (
        Path(__file__).resolve().parent
        / "results"
        / "hull_white_normal_vol_comparison.csv"
    )

    reference = pd.read_csv(reference_path)

    if len(reference) != len(diagnostics):
        raise AssertionError("Reference row count does not match")

    np.testing.assert_allclose(
        diagnostics.to_numpy(dtype=float),
        reference.to_numpy(dtype=float),
        rtol=0,
        atol=1e-5,
    )

    print("=== Reproducible Hull-White Calibration ===")
    print(f"Swaptions: {len(quotes)}")
    print(f"Mean reversion a: {result.mean_reversion:.8f}")
    print(f"Hull-White sigma: {result.volatility:.8f}")
    print(f"Normal-volatility RMSE: {rmse:.4f} bp")
    print("Saved CSV comparison: PASSED")


if __name__ == "__main__":
    main()
