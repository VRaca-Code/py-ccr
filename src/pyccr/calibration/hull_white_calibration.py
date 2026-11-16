"""Hull-White one-factor calibration to normal swaption volatilities."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pyccr.market.smooth_discount_curve import SmoothDiscountCurve
from pyccr.models.hull_white_curve import HullWhiteCurveModel
from pyccr.pricing.bachelier_swaption import bachelier_swaption_price
from pyccr.pricing.hull_white_swaption import hull_white_swaption_price


@dataclass(frozen=True)
class SwaptionQuote:
    """European swaption quote with normal volatility in absolute rate units."""

    expiry: float
    payment_times: tuple[float, ...]
    normal_volatility: float
    strike: float | None = None
    option_type: str = "payer"

    def __post_init__(self) -> None:
        if not np.isfinite(self.expiry) or self.expiry <= 0:
            raise ValueError("expiry must be positive and finite")

        if not np.isfinite(self.normal_volatility) or self.normal_volatility < 0:
            raise ValueError("normal_volatility must be non-negative and finite")

        if self.strike is not None and (
            not np.isfinite(self.strike) or self.strike < 0
        ):
            raise ValueError("strike must be non-negative and finite")

        if self.option_type not in ("payer", "receiver"):
            raise ValueError("option_type must be 'payer' or 'receiver'")

        times = np.asarray(self.payment_times, dtype=float)

        if (
            times.ndim != 1
            or len(times) == 0
            or not np.all(np.isfinite(times))
            or times[0] <= self.expiry
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "payment_times must be strictly increasing and after expiry"
            )

        object.__setattr__(
            self,
            "payment_times",
            tuple(float(t) for t in times),
        )


@dataclass(frozen=True)
class HullWhiteCalibrationResult:
    """Calibrated parameters and numerical optimization diagnostics."""

    model: HullWhiteCurveModel
    mean_reversion: float
    volatility: float
    rmse: float
    success: bool
    message: str


def calibrate_hull_white(
    curve: SmoothDiscountCurve,
    quotes: tuple[SwaptionQuote, ...],
    initial_mean_reversion: float = 0.05,
    initial_volatility: float = 0.01,
) -> HullWhiteCalibrationResult:
    """Fit Hull-White parameters to European swaption normal volatilities.

    Assumes single-curve pricing and deterministic accrual schedules.
    """
    if not isinstance(curve, SmoothDiscountCurve):
        raise TypeError("curve must be a SmoothDiscountCurve")

    if not quotes:
        raise ValueError("at least one swaption quote is required")

    if not all(isinstance(q, SwaptionQuote) for q in quotes):
        raise TypeError("quotes must contain SwaptionQuote objects")

    if any(q.payment_times[-1] > curve.tenors[-1] for q in quotes):
        raise ValueError("swaption maturity exceeds curve horizon")

    initial = np.asarray(
        [initial_mean_reversion, initial_volatility],
        dtype=float,
    )

    if not np.all(np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError("initial parameters must be positive and finite")

    market_prices = []
    annuities = []

    for quote in quotes:
        times = np.asarray(quote.payment_times)
        accruals = np.diff(np.concatenate(([quote.expiry], times)))
        discounts = np.asarray(curve.discount_factor(times))

        annuity = float(np.dot(accruals, discounts))
        forward = float(
            (curve.discount_factor(quote.expiry) - curve.discount_factor(times[-1]))
            / annuity
        )

        strike = forward if quote.strike is None else quote.strike

        market_price = bachelier_swaption_price(
            forward=forward,
            strike=strike,
            expiry=quote.expiry,
            normal_volatility=quote.normal_volatility,
            annuity=annuity,
            option_type=quote.option_type,
        )

        market_prices.append(market_price)
        annuities.append(annuity)

    market_prices = np.asarray(market_prices)
    annuities = np.asarray(annuities)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        model = HullWhiteCurveModel(
            curve=curve,
            mean_reversion=float(parameters[0]),
            volatility=float(parameters[1]),
        )

        prices = []

        for quote in quotes:
            times = np.asarray(quote.payment_times)
            accruals = np.diff(np.concatenate(([quote.expiry], times)))
            discounts = np.asarray(curve.discount_factor(times))
            annuity = float(np.dot(accruals, discounts))

            forward = float(
                (curve.discount_factor(quote.expiry) - curve.discount_factor(times[-1]))
                / annuity
            )

            strike = forward if quote.strike is None else quote.strike

            prices.append(
                hull_white_swaption_price(
                    model=model,
                    expiry=quote.expiry,
                    payment_times=quote.payment_times,
                    fixed_rate=strike,
                    option_type=quote.option_type,
                )
            )

        return (np.asarray(prices) - market_prices) / annuities

    optimization = least_squares(
        residuals,
        x0=initial,
        bounds=([1e-5, 1e-8], [2.0, 1.0]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=300,
    )

    calibrated_model = HullWhiteCurveModel(
        curve=curve,
        mean_reversion=float(optimization.x[0]),
        volatility=float(optimization.x[1]),
    )

    final_residuals = residuals(optimization.x)

    return HullWhiteCalibrationResult(
        model=calibrated_model,
        mean_reversion=calibrated_model.mean_reversion,
        volatility=calibrated_model.volatility,
        rmse=float(np.sqrt(np.mean(final_residuals**2))),
        success=bool(optimization.success),
        message=str(optimization.message),
    )
