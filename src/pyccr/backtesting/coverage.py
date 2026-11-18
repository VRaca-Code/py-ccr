"""Historical forecast-quantile coverage backtesting.

Inputs are realized historical observations and forecasts issued before
those observations became known. Monte Carlo simulation paths alone
are not independent historical validation observations.

Exact binomial tests assume independent Bernoulli breach indicators.
For overlapping forecast horizons or serial dependence, use a
dependence-aware procedure instead.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

Tail = Literal["upper", "lower"]
Alternative = Literal["greater", "less", "two-sided"]


@dataclass(frozen=True)
class CoverageResult:
    """Observed breach frequency and exact-binomial coverage statistics."""

    tail: Tail
    nominal_probability: float
    significance_level: float
    alternative: Alternative
    breaches: NDArray[np.bool_]
    n_observations: int
    n_breaches: int
    expected_breaches: float
    observed_frequency: float
    p_value: float
    reject_null: bool


@dataclass(frozen=True)
class RollingCoverageResult:
    """Trailing-window breach counts and frequencies."""

    window: int
    endpoints: NDArray[np.int64]
    counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    expected_counts: NDArray[np.float64]


def _probability(value: float, name: str) -> float:
    """Validate a probability strictly between zero and one."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be strictly between zero and one")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be strictly between zero and one") from exc
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one")
    return result


def breach_indicators(
    realized: ArrayLike,
    quantiles: ArrayLike,
    *,
    tail: Tail = "upper",
) -> NDArray[np.bool_]:
    """Flag strict breaches of forecasts aligned with realizations.

    Equality is not a breach. For discrete distributions, strict
    quantile exceedance need not have exactly the nominal probability.
    """
    if tail not in ("upper", "lower"):
        raise ValueError("tail must be 'upper' or 'lower'")

    observed = np.asarray(realized, dtype=np.float64)
    forecast = np.asarray(quantiles, dtype=np.float64)

    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("realized must be a nonempty 1D array")
    if forecast.shape != observed.shape:
        raise ValueError("quantiles must match realized shape")
    if not np.all(np.isfinite(observed)):
        raise ValueError("realized must contain only finite values")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("quantiles must contain only finite values")

    if tail == "upper":
        return np.asarray(observed > forecast, dtype=np.bool_)
    return np.asarray(observed < forecast, dtype=np.bool_)


def exact_binomial_p_value(
    n_breaches: int,
    n_observations: int,
    nominal_probability: float,
    *,
    alternative: Alternative = "greater",
) -> float:
    """Compute an exact binomial p-value under independent breaches.

    The two-sided probability sums outcomes whose null PMF is no
    greater than that of the observed count, matching the usual
    probability-ordering definition of an exact binomial test.
    """
    p = _probability(nominal_probability, "nominal_probability")

    if (
        isinstance(n_observations, (bool, np.bool_))
        or not isinstance(n_observations, (int, np.integer))
        or n_observations < 1
    ):
        raise ValueError("n_observations must be a positive integer")

    if (
        isinstance(n_breaches, (bool, np.bool_))
        or not isinstance(n_breaches, (int, np.integer))
        or not 0 <= n_breaches <= n_observations
    ):
        raise ValueError("n_breaches must be between zero and n_observations")

    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("unsupported alternative")

    n = int(n_observations)
    k = int(n_breaches)

    if alternative == "greater":
        return float(binom.sf(k - 1, n, p))
    if alternative == "less":
        return float(binom.cdf(k, n, p))

    from scipy.stats import binomtest

    return float(binomtest(k, n, p, alternative="two-sided").pvalue)


def test_coverage(
    realized: ArrayLike,
    quantiles: ArrayLike,
    *,
    tail: Tail = "upper",
    nominal_probability: float = 0.01,
    significance_level: float = 0.05,
    alternative: Alternative = "greater",
) -> CoverageResult:
    """Test historical quantile coverage using the exact binomial law."""
    p = _probability(nominal_probability, "nominal_probability")
    alpha = _probability(significance_level, "significance_level")
    breaches = breach_indicators(realized, quantiles, tail=tail)

    n = int(breaches.size)
    k = int(np.count_nonzero(breaches))
    p_value = exact_binomial_p_value(k, n, p, alternative=alternative)

    return CoverageResult(
        tail=tail,
        nominal_probability=p,
        significance_level=alpha,
        alternative=alternative,
        breaches=breaches.copy(),
        n_observations=n,
        n_breaches=k,
        expected_breaches=n * p,
        observed_frequency=k / n,
        p_value=p_value,
        reject_null=p_value < alpha,
    )


def rolling_coverage(
    breaches: ArrayLike,
    *,
    window: int,
    nominal_probability: float = 0.01,
) -> RollingCoverageResult:
    """Calculate trailing-window statistics without statistical testing.

    Endpoints are zero-based, inclusive observation indices.
    """
    p = _probability(nominal_probability, "nominal_probability")
    values = np.asarray(breaches)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("breaches must be a nonempty 1D array")
    if values.dtype.kind != "b":
        raise ValueError("breaches must be a boolean array")
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or not 1 <= window <= values.size
    ):
        raise ValueError("window must be between one and sample size")

    width = int(window)
    cumulative = np.concatenate(([0], np.cumsum(values.astype(np.int64))))
    counts = cumulative[width:] - cumulative[:-width]

    return RollingCoverageResult(
        window=width,
        endpoints=np.arange(width - 1, values.size, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        frequencies=np.asarray(counts / width, dtype=np.float64),
        expected_counts=np.full(counts.size, width * p, dtype=np.float64),
    )
