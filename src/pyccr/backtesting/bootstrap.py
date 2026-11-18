"""Bootstrap confidence intervals for historical breach frequencies.

Resampling is centered on the empirical breach process, not on the
coverage null hypothesis. Consequently these intervals do not supply
null-calibrated bootstrap p-values.

IID bootstrap assumes exchangeable independent observations. Circular
block bootstrap retains local dependence within sampled blocks, but
its validity requires suitable weak-dependence and block-length
assumptions. Neither method fixes overlapping forecast horizons.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

BootstrapMethod = Literal["iid", "circular_block"]


@dataclass(frozen=True)
class BootstrapCoverageResult:
    """Percentile confidence interval for observed breach frequency."""

    method: BootstrapMethod
    n_observations: int
    n_breaches: int
    observed_frequency: float
    confidence_level: float
    n_resamples: int
    block_length: int | None
    lower_bound: float
    upper_bound: float
    resampled_frequencies: NDArray[np.float64]


def bootstrap_coverage_interval(
    breaches: ArrayLike,
    *,
    method: BootstrapMethod = "iid",
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    block_length: int | None = None,
    seed: int = 42,
) -> BootstrapCoverageResult:
    """Estimate a percentile interval by resampling breach indicators.

    The resampling distribution is centered on observed data.
    The returned interval is descriptive and is not a coverage-null
    bootstrap test. Percentile intervals can be degenerate for rare
    events, especially when no breaches were observed.
    """
    indicators = np.asarray(breaches)

    if indicators.ndim != 1 or indicators.size == 0 or indicators.dtype.kind != "b":
        raise ValueError("breaches must be a nonempty 1D boolean array")

    if method not in ("iid", "circular_block"):
        raise ValueError("unsupported bootstrap method")

    if isinstance(confidence_level, (bool, np.bool_)):
        raise TypeError("confidence_level must be in (0, 1)")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    if (
        isinstance(n_resamples, (bool, np.bool_))
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 100
    ):
        raise ValueError("n_resamples must be an integer >= 100")

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be a valid nonnegative integer")

    n = int(indicators.size)

    if method == "iid":
        if block_length is not None:
            raise ValueError("block_length is only valid for block bootstrap")
    else:
        if (
            isinstance(block_length, (bool, np.bool_))
            or not isinstance(block_length, (int, np.integer))
            or not 1 <= block_length <= n
        ):
            raise ValueError(
                "block_length must be an integer between 1 and sample size"
            )

    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_resamples), dtype=np.float64)

    if method == "iid":
        for index in range(int(n_resamples)):
            draw = rng.integers(0, n, size=n)
            samples[index] = np.mean(indicators[draw])
    else:
        length = int(block_length)
        n_blocks = (n + length - 1) // length
        offsets = np.arange(length)

        for index in range(int(n_resamples)):
            starts = rng.integers(0, n, size=n_blocks)
            draw = ((starts[:, None] + offsets) % n).ravel()[:n]
            samples[index] = np.mean(indicators[draw])

    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(
        samples,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return BootstrapCoverageResult(
        method=method,
        n_observations=n,
        n_breaches=int(np.count_nonzero(indicators)),
        observed_frequency=float(np.mean(indicators)),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        block_length=int(block_length) if block_length is not None else None,
        lower_bound=float(lower),
        upper_bound=float(upper),
        resampled_frequencies=samples,
    )
