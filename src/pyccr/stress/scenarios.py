"""Pure market-parameter shocks, without repricing or simulation.

Basis-point shocks are additive to continuously compounded
zero rates or constant hazard rates. FX spot shocks are relative.
Volatility and correlation shocks are additive decimal changes.

Each function returns a new object or scalar. No input is mutated.
"""

import math

from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.market.discount_curve import DiscountCurve
from pyccr.market.smooth_discount_curve import SmoothDiscountCurve

BASIS_POINT = 1.0e-4


def _finite_scalar(value: float, name: str) -> float:
    """Accept finite real scalars, excluding booleans."""
    try:
        if isinstance(value, bool):
            raise TypeError("boolean values are not valid real scalars")
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _shifted_rates(
    rates: tuple[float, ...],
    shift_bps: float,
) -> tuple[float, ...]:
    shift = _finite_scalar(shift_bps, "shift_bps") * BASIS_POINT
    if not math.isfinite(shift):
        raise ValueError("rate shift exceeds finite range")

    shifted = tuple(rate + shift for rate in rates)
    if not all(math.isfinite(rate) for rate in shifted):
        raise ValueError("shocked rates must be finite")
    return shifted


def shock_discount_curve(
    curve: DiscountCurve,
    shift_bps: float,
) -> DiscountCurve:
    """Add a parallel zero-rate shift in basis points."""
    if not isinstance(curve, DiscountCurve):
        raise TypeError("curve must be a DiscountCurve")
    return DiscountCurve(
        tenors=curve.tenors,
        zero_rates=_shifted_rates(curve.zero_rates, shift_bps),
    )


def shock_smooth_discount_curve(
    curve: SmoothDiscountCurve,
    shift_bps: float,
) -> SmoothDiscountCurve:
    """Parallel shift preserving the smooth curve's node grid."""
    if not isinstance(curve, SmoothDiscountCurve):
        raise TypeError("curve must be a SmoothDiscountCurve")
    return SmoothDiscountCurve(
        tenors=curve.tenors,
        zero_rates=_shifted_rates(curve.zero_rates, shift_bps),
    )


def shock_fx_spot(
    initial_spot: float,
    relative_shock: float,
) -> float:
    """Apply S_shocked = S_base * (1 + relative_shock)."""
    spot = _finite_scalar(initial_spot, "initial_spot")
    shock = _finite_scalar(relative_shock, "relative_shock")

    if spot <= 0.0:
        raise ValueError("initial_spot must be positive")
    if shock <= -1.0:
        raise ValueError("relative_shock must be greater than -1")

    shocked = spot * (1.0 + shock)
    if not math.isfinite(shocked) or shocked <= 0.0:
        raise ValueError("shocked FX spot must be positive and finite")
    return shocked


def shock_volatility(
    volatility: float,
    absolute_shock: float,
) -> float:
    """Add an absolute decimal volatility shock."""
    base = _finite_scalar(volatility, "volatility")
    shock = _finite_scalar(absolute_shock, "absolute_shock")

    if base < 0.0:
        raise ValueError("volatility must be non-negative")

    shocked = base + shock
    if not math.isfinite(shocked) or shocked < 0.0:
        raise ValueError("shocked volatility must be finite and non-negative")
    return shocked


def shock_hazard_curve(
    curve: ConstantHazardCurve,
    shift_bps: float,
) -> ConstantHazardCurve:
    """Add basis points directly to the constant hazard rate.

    This is a hazard-rate shock, not a CDS-spread shock.
    No spread-to-hazard conversion is implied.
    """
    if not isinstance(curve, ConstantHazardCurve):
        raise TypeError("curve must be a ConstantHazardCurve")

    shift = _finite_scalar(shift_bps, "shift_bps") * BASIS_POINT
    shocked = curve.hazard_rate + shift

    if not math.isfinite(shocked) or shocked < 0.0:
        raise ValueError("shocked hazard rate must be finite and non-negative")
    return ConstantHazardCurve(hazard_rate=shocked)


def shock_correlation(
    correlation: float,
    absolute_shock: float,
) -> float:
    """Shock a two-factor correlation coefficient.

    Strict bounds follow the joint FX/Hull-White simulator's
    positive-definite correlation convention. This is not
    a general correlation-matrix stress transformation.
    """
    base = _finite_scalar(correlation, "correlation")
    shock = _finite_scalar(absolute_shock, "absolute_shock")

    if not -1.0 < base < 1.0:
        raise ValueError("correlation must be strictly between -1 and 1")

    shocked = base + shock
    if not math.isfinite(shocked) or not -1.0 < shocked < 1.0:
        raise ValueError("shocked correlation must be strictly between -1 and 1")
    return shocked
