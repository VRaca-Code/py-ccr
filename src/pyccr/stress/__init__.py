"""Deterministic market-parameter stress transformations."""

from pyccr.stress.scenarios import (
    shock_correlation,
    shock_discount_curve,
    shock_fx_spot,
    shock_hazard_curve,
    shock_smooth_discount_curve,
    shock_volatility,
)

__all__ = [
    "shock_correlation",
    "shock_discount_curve",
    "shock_fx_spot",
    "shock_hazard_curve",
    "shock_smooth_discount_curve",
    "shock_volatility",
]
