"""Collateral modelling for PyCCR."""

from pyccr.collateral.csa import CSAConfig
from pyccr.collateral.initial_margin import (
    InitialMarginResult,
    calculate_initial_margin_exposures,
)
from pyccr.collateral.margin import (
    MarginResult,
    simulate_variation_margin,
    variation_margin_target,
)
from pyccr.collateral.mpor import MPORResult, calculate_mpor_exposures

__all__ = [
    "CSAConfig",
    "InitialMarginResult",
    "MPORResult",
    "MarginResult",
    "calculate_initial_margin_exposures",
    "calculate_mpor_exposures",
    "simulate_variation_margin",
    "variation_margin_target",
]
