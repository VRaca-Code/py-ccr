"""Uncollateralized exposure profiles from pathwise portfolio values."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ExposureProfiles:
    """Exposure metrics for one aggregated portfolio.

    positive_exposures and negative_exposures have shape
    (n_paths, n_times). EE, ENE and PFE have shape (n_times,).

    EPE is the time-weighted average of EE over the supplied grid.
    """

    times: NDArray[np.float64]
    positive_exposures: NDArray[np.float64]
    negative_exposures: NDArray[np.float64]
    ee: NDArray[np.float64]
    ene: NDArray[np.float64]
    pfe: NDArray[np.float64]
    epe: float
    confidence_level: float


def calculate_exposure_profiles(
    times: ArrayLike,
    portfolio_values: ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> ExposureProfiles:
    """Calculate uncollateralized EE, ENE, PFE and EPE.

    portfolio_values must have shape (n_paths, n_times) and
    represent signed values in one consistent currency.

    The caller is responsible for any legally valid netting.
    Collateral, discounting and credit risk are not modeled.

    PFE uses NumPy's linear quantile interpolation.
    EPE uses trapezoidal time integration over the supplied grid.
    """
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

    if not np.isscalar(confidence_level):
        raise ValueError("confidence_level must be a finite scalar")

    try:
        level = float(confidence_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence_level must be a finite scalar") from exc

    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    positive = np.maximum(values, 0.0)
    negative = np.maximum(-values, 0.0)

    ee = np.mean(positive, axis=0)
    ene = np.mean(negative, axis=0)
    pfe = np.quantile(positive, level, axis=0)

    if t.size == 1:
        epe = float(ee[0])
    else:
        dt = np.diff(t)
        integrated_ee = np.sum(0.5 * (ee[:-1] + ee[1:]) * dt)
        epe = float(integrated_ee / (t[-1] - t[0]))

    return ExposureProfiles(
        times=t.copy(),
        positive_exposures=np.asarray(positive, dtype=np.float64),
        negative_exposures=np.asarray(negative, dtype=np.float64),
        ee=np.asarray(ee, dtype=np.float64),
        ene=np.asarray(ene, dtype=np.float64),
        pfe=np.asarray(pfe, dtype=np.float64),
        epe=epe,
        confidence_level=level,
    )
