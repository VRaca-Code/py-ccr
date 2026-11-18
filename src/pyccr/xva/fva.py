"""Deterministic-spread funding valuation adjustment.

This simplified, uncollateralized calculation uses expected positive
and negative portfolio exposures with deterministic discount factors.
It does not model collateral, funding constraints, stochastic spreads,
default-survival weighting, closeout, or interactions with CVA/DVA.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FVAResult:
    """Positive FCA/FBA magnitudes and signed FVA = FBA - FCA.

    All contribution arrays have length ``len(times) - 1``.
    """

    fca: float
    fba: float
    fva: float
    times: NDArray[np.float64]
    borrowing_spreads: NDArray[np.float64]
    lending_spreads: NDArray[np.float64]
    discounted_expected_exposures: NDArray[np.float64]
    discounted_expected_negative_exposures: NDArray[np.float64]
    fca_contributions: NDArray[np.float64]
    fba_contributions: NDArray[np.float64]
    net_contributions: NDArray[np.float64]


def _validated_spread(
    spread: ArrayLike,
    name: str,
    n_intervals: int,
) -> NDArray[np.float64]:
    """Expand a constant spread or validate interval-specific spreads."""
    try:
        values = np.asarray(spread, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc

    if values.ndim == 0:
        values = np.full(n_intervals, float(values), dtype=np.float64)
    elif values.ndim != 1 or values.shape != (n_intervals,):
        raise ValueError(f"{name} must be a scalar or have length len(times) - 1")

    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")

    return values.copy()


def calculate_fva(
    times: ArrayLike,
    expected_exposure: ArrayLike,
    expected_negative_exposure: ArrayLike,
    discount_factors: ArrayLike,
    borrowing_spread: ArrayLike,
    lending_spread: ArrayLike = 0.0,
) -> FVAResult:
    """Calculate endpoint-discretized deterministic-spread FVA.

    Expected exposure (EE) and expected negative exposure (ENE) must
    be nonnegative magnitudes, in one consistent currency.

    At interval endpoint t[i], funding cost is
    DF[i] * EE[i] * borrowing_spread[i-1] * (t[i] - t[i-1]).

    Funding benefit uses ENE and the lending spread analogously.
    Spreads are continuously quoted annualized decimal rates, such
    that 100 basis points equals 0.01.

    FCA and FBA are nonnegative. Signed FVA equals FBA minus FCA,
    so a net funding cost reduces portfolio value.

    No default-survival weighting is applied. This is not a
    collateralized, stochastic-funding, or joint XVA valuation.
    """
    try:
        t = np.asarray(times, dtype=np.float64)
        ee = np.asarray(expected_exposure, dtype=np.float64)
        ene = np.asarray(expected_negative_exposure, dtype=np.float64)
        df = np.asarray(discount_factors, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be finite numeric arrays") from exc

    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a one-dimensional grid")

    if not np.all(np.isfinite(t)) or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and increase strictly")

    if ee.shape != t.shape or not np.all(np.isfinite(ee)):
        raise ValueError("expected_exposure must be finite and match times")

    if np.any(ee < 0.0):
        raise ValueError("expected_exposure must be non-negative")

    if ene.shape != t.shape or not np.all(np.isfinite(ene)):
        raise ValueError("expected_negative_exposure must be finite and match times")

    if np.any(ene < 0.0):
        raise ValueError("expected_negative_exposure must be non-negative")

    if df.shape != t.shape or not np.all(np.isfinite(df)):
        raise ValueError("discount_factors must be finite and match times")

    if np.any(df <= 0.0):
        raise ValueError("discount_factors must be strictly positive")

    borrow = _validated_spread(borrowing_spread, "borrowing_spread", t.size - 1)
    lend = _validated_spread(lending_spread, "lending_spread", t.size - 1)

    dt = np.diff(t)
    discounted_ee = df[1:] * ee[1:]
    discounted_ene = df[1:] * ene[1:]

    fca_contributions = discounted_ee * borrow * dt
    fba_contributions = discounted_ene * lend * dt
    net_contributions = fba_contributions - fca_contributions

    fca = float(np.sum(fca_contributions))
    fba = float(np.sum(fba_contributions))

    return FVAResult(
        fca=fca,
        fba=fba,
        fva=float(fba - fca),
        times=t.copy(),
        borrowing_spreads=borrow,
        lending_spreads=lend,
        discounted_expected_exposures=discounted_ee,
        discounted_expected_negative_exposures=discounted_ene,
        fca_contributions=fca_contributions,
        fba_contributions=fba_contributions,
        net_contributions=net_contributions,
    )
