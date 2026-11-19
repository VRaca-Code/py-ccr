"""Collateral-aware exposure profiles on the hypothetical default-time grid.

The existing VM, MPOR and segregated IM engines are composed without
changing their individual conventions.

Exposure at default index d uses settled VM and eligible IM frozen at d,
and portfolio closeout value at d + mpor_steps.

For positive MPOR, the available default-time horizon is shorter than
the supplied simulation horizon. No terminal exposure extrapolation is
performed.

This module does not model funding costs, collateral interest, haircuts,
wrong-way risk, or legal closeout enforceability.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.collateral.csa import CSAConfig
from pyccr.collateral.initial_margin import (
    InitialMarginResult,
    calculate_initial_margin_exposures,
)
from pyccr.collateral.margin import MarginResult, simulate_variation_margin
from pyccr.collateral.mpor import MPORResult, calculate_mpor_exposures
from pyccr.exposure.profiles import ExposureProfiles

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CollateralExposureResult:
    """Exposure profiles and underlying collateral calculation results."""

    profiles: ExposureProfiles
    margin_result: MarginResult
    mpor_result: MPORResult
    initial_margin_result: InitialMarginResult
    closeout_times: FloatArray


def _validate_times_and_values(
    times: ArrayLike,
    portfolio_values: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    t = np.asarray(times, dtype=np.float64)
    values = np.asarray(portfolio_values, dtype=np.float64)

    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a nonempty 1D array")

    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")

    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and strictly increase")

    if values.ndim != 2 or values.shape[1] != t.size:
        raise ValueError("portfolio_values must have shape (n_paths, n_times)")

    if values.shape[0] == 0:
        raise ValueError("portfolio_values must contain at least one path")

    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio_values must be finite")

    return t.copy(), values.copy()


def _exposure_profiles_from_paths(
    times: FloatArray,
    positive_exposures: FloatArray,
    negative_exposures: FloatArray,
    confidence_level: float,
) -> ExposureProfiles:
    """Aggregate already-clipped bilateral pathwise exposures."""

    positive = np.asarray(positive_exposures, dtype=np.float64)
    negative = np.asarray(negative_exposures, dtype=np.float64)

    if (
        positive.ndim != 2
        or positive.shape != negative.shape
        or positive.shape[0] == 0
        or positive.shape[1] != times.size
    ):
        raise ValueError("collateral exposure arrays have invalid shape")

    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ValueError("collateral exposures must be finite")

    if np.any(positive < 0.0) or np.any(negative < 0.0):
        raise ValueError("collateral exposures must be nonnegative")

    if not np.isscalar(confidence_level):
        raise ValueError("confidence_level must be a finite scalar")

    try:
        level = float(np.asarray(confidence_level, dtype=np.float64))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence_level must be a finite scalar") from exc

    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    ee = np.mean(positive, axis=0)
    ene = np.mean(negative, axis=0)
    pfe = np.quantile(positive, level, axis=0)

    if times.size == 1:
        epe = float(ee[0])
    else:
        integrated = np.sum(0.5 * (ee[:-1] + ee[1:]) * np.diff(times))
        epe = float(integrated / (times[-1] - times[0]))

    return ExposureProfiles(
        times=times.copy(),
        positive_exposures=positive.copy(),
        negative_exposures=negative.copy(),
        ee=np.asarray(ee, dtype=np.float64),
        ene=np.asarray(ene, dtype=np.float64),
        pfe=np.asarray(pfe, dtype=np.float64),
        epe=epe,
        confidence_level=level,
    )


def calculate_collateral_exposure_profiles(
    times: ArrayLike,
    portfolio_values: ArrayLike,
    csa: CSAConfig,
    *,
    mpor_steps: int = 0,
    received_im: ArrayLike = 0.0,
    posted_im: ArrayLike = 0.0,
    confidence_level: float = 0.95,
) -> CollateralExposureResult:
    """Calculate bilateral collateral-adjusted EE, ENE, PFE and EPE.

    All amounts must use the same currency units.

    ``profiles.times`` contains hypothetical default times, whereas
    ``closeout_times`` contains the corresponding closeout times.

    When MPOR is positive, ``profiles.times`` ends before the final
    simulation time. XVA callers must explicitly account for this
    shortened horizon or supply an extended simulation grid.

    The returned profiles can be passed to existing CVA/DVA/XVA
    functions using ``profiles.ee`` and ``profiles.ene``.
    """

    t, values = _validate_times_and_values(times, portfolio_values)

    if not isinstance(csa, CSAConfig):
        raise TypeError("csa must be a CSAConfig")

    margin = simulate_variation_margin(values, csa)

    mpor = calculate_mpor_exposures(
        values,
        margin.vm_balance,
        mpor_steps=mpor_steps,
    )

    initial_margin = calculate_initial_margin_exposures(
        mpor,
        received_im=received_im,
        posted_im=posted_im,
    )

    default_times = t[initial_margin.default_indices].copy()
    closeout_times = t[initial_margin.closeout_indices].copy()

    profiles = _exposure_profiles_from_paths(
        default_times,
        initial_margin.positive_exposure,
        initial_margin.negative_exposure,
        confidence_level,
    )

    return CollateralExposureResult(
        profiles=profiles,
        margin_result=margin,
        mpor_result=mpor,
        initial_margin_result=initial_margin,
        closeout_times=closeout_times,
    )
