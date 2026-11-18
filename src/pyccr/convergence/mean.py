"""Confidence intervals for Monte Carlo sample means.

Standard errors are computed from independent sampling units.
With antithetic sampling, each positive/negative path pair is
reduced to one average before estimating uncertainty.

Student-t intervals are finite-sample approximations, not
guaranteed coverage intervals for arbitrary payoff distributions.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t


@dataclass(frozen=True)
class MonteCarloMeanResult:
    """Estimated mean and uncertainty from independent units."""

    estimate: float
    standard_error: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    n_paths: int
    n_independent_units: int
    antithetic: bool


def monte_carlo_mean_interval(
    observations: ArrayLike,
    *,
    confidence_level: float = 0.95,
    antithetic: bool = False,
) -> MonteCarloMeanResult:
    """Estimate a mean and Student-t confidence interval.

    In antithetic mode, observations must be ordered as
    [Z_1, ..., Z_m, -Z_1, ..., -Z_m] in matching path order.
    Pair averages are treated as independent across pairs.

    The function cannot verify that the supplied observations
    actually arise from independent paths or antithetic pairs.
    """
    values = np.asarray(observations, dtype=np.float64)

    if values.ndim != 1 or values.size < 2:
        raise ValueError("observations must be a 1D array with at least two paths")

    if not np.all(np.isfinite(values)):
        raise ValueError("observations must be finite")

    if not isinstance(antithetic, (bool, np.bool_)):
        raise TypeError("antithetic must be boolean")

    if isinstance(confidence_level, (bool, np.bool_)):
        raise TypeError("confidence_level must be numeric")

    if not np.isscalar(confidence_level):
        raise TypeError("confidence_level must be scalar")

    try:
        level = float(confidence_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("confidence_level must be numeric") from exc

    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    n_paths = int(values.size)

    if antithetic:
        if n_paths < 4 or n_paths % 2:
            raise ValueError(
                "antithetic sampling requires at least four "
                "paths and an even path count"
            )

        half = n_paths // 2
        units = 0.5 * (values[:half] + values[half:])
    else:
        units = values

    n_units = int(units.size)
    estimate = float(np.mean(units))
    standard_error = float(np.std(units, ddof=1) / np.sqrt(n_units))

    critical = float(t.ppf(0.5 + level / 2.0, df=n_units - 1))
    half_width = critical * standard_error

    return MonteCarloMeanResult(
        estimate=estimate,
        standard_error=standard_error,
        lower_bound=estimate - half_width,
        upper_bound=estimate + half_width,
        confidence_level=level,
        n_paths=n_paths,
        n_independent_units=n_units,
        antithetic=bool(antithetic),
    )
