"""Flat CDS-implied hazard calibration using the credit triangle."""

import numpy as np

from pyccr.credit.hazard_curve import ConstantHazardCurve


def calibrate_flat_hazard_from_cds(
    cds_spread_bps: float,
    recovery_rate: float = 0.4,
) -> ConstantHazardCurve:
    """Infer a flat hazard rate from a CDS spread in basis points.

    Uses the credit-triangle approximation:

        hazard_rate = (cds_spread_bps / 10000) / (1 - recovery_rate)

    This is not a full CDS bootstrap or exact CDS repricing model.
    """
    spread = float(cds_spread_bps)
    recovery = float(recovery_rate)

    if not np.isfinite(spread) or spread < 0.0:
        raise ValueError("cds_spread_bps must be finite and non-negative")

    if not np.isfinite(recovery) or not 0.0 <= recovery < 1.0:
        raise ValueError("recovery_rate must be finite and lie in [0, 1)")

    hazard_rate = (spread / 10_000.0) / (1.0 - recovery)

    if not np.isfinite(hazard_rate):
        raise ValueError("implied hazard_rate must be finite")

    return ConstantHazardCurve(hazard_rate=hazard_rate)


def implied_cds_spread_bps(
    credit_curve: ConstantHazardCurve,
    recovery_rate: float = 0.4,
) -> float:
    """Return the credit-triangle CDS spread in basis points."""
    if not isinstance(credit_curve, ConstantHazardCurve):
        raise TypeError("credit_curve must be a ConstantHazardCurve")

    recovery = float(recovery_rate)

    if not np.isfinite(recovery) or not 0.0 <= recovery <= 1.0:
        raise ValueError("recovery_rate must be finite and lie in [0, 1]")

    return float(10_000.0 * credit_curve.hazard_rate * (1.0 - recovery))
