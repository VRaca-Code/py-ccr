"""FX implied-volatility surface on a strike-maturity grid."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FXVolSurface:
    """FX volatility surface interpolated in total implied variance.

    Volatility quotes are annualized decimals. Rows correspond to
    maturities and columns correspond to absolute FX strikes.

    Interpolation is linear in strike and total variance.
    Extrapolation beyond the quoted domain is not supported.
    """

    maturities: tuple[float, ...]
    strikes: tuple[float, ...]
    volatilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        maturities = np.asarray(self.maturities, dtype=float)
        strikes = np.asarray(self.strikes, dtype=float)
        vols = np.asarray(self.volatilities, dtype=float)

        if maturities.ndim != 1 or maturities.size == 0:
            raise ValueError("maturities must be a non-empty 1D sequence")

        if strikes.ndim != 1 or strikes.size == 0:
            raise ValueError("strikes must be a non-empty 1D sequence")

        if not np.all(np.isfinite(maturities)) or np.any(maturities <= 0):
            raise ValueError("maturities must be positive and finite")

        if not np.all(np.isfinite(strikes)) or np.any(strikes <= 0):
            raise ValueError("strikes must be positive and finite")

        if np.any(np.diff(maturities) <= 0):
            raise ValueError("maturities must be strictly increasing")

        if np.any(np.diff(strikes) <= 0):
            raise ValueError("strikes must be strictly increasing")

        expected_shape = (maturities.size, strikes.size)

        if vols.shape != expected_shape:
            raise ValueError(f"volatilities must have shape {expected_shape}")

        if not np.all(np.isfinite(vols)) or np.any(vols < 0):
            raise ValueError("volatilities must be finite and non-negative")

        object.__setattr__(self, "maturities", tuple(float(x) for x in maturities))
        object.__setattr__(self, "strikes", tuple(float(x) for x in strikes))
        object.__setattr__(
            self,
            "volatilities",
            tuple(tuple(float(x) for x in row) for row in vols),
        )

    def total_variance(self, maturity: float, strike: float) -> float:
        """Interpolate total implied variance within the quoted grid."""
        if not np.isfinite(maturity) or not np.isfinite(strike):
            raise ValueError("maturity and strike must be finite")

        if maturity <= 0 or strike <= 0:
            raise ValueError("maturity and strike must be positive")

        times = np.asarray(self.maturities)
        strikes = np.asarray(self.strikes)
        vols = np.asarray(self.volatilities)

        if maturity < times[0] or maturity > times[-1]:
            raise ValueError("maturity outside quoted surface domain")

        if strike < strikes[0] or strike > strikes[-1]:
            raise ValueError("strike outside quoted surface domain")

        variance_grid = times[:, None] * vols**2

        # First interpolate total variance across strike for each maturity.
        strike_variances = np.array(
            [np.interp(strike, strikes, row) for row in variance_grid]
        )

        # Then interpolate total variance across maturity.
        return float(np.interp(maturity, times, strike_variances))

    def volatility(self, maturity: float, strike: float) -> float:
        """Return annualized implied volatility at maturity and strike."""
        variance = self.total_variance(maturity, strike)
        return float(np.sqrt(variance / maturity))
