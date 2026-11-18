"""Christoffersen likelihood-ratio diagnostics for breach sequences.

LRuc tests unconditional coverage against a specified nominal
breach probability. LRind compares a first-order Markov model
with a constant transition probability. LRcc = LRuc + LRind.

P-values use asymptotic chi-square reference distributions.
Sparse transitions, boundary estimates, and overlapping forecast
horizons can invalidate these approximations. These tests do
not replace independent historical out-of-sample validation.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import xlogy
from scipy.stats import chi2


@dataclass(frozen=True)
class ChristoffersenResult:
    """Likelihood-ratio statistics and asymptotic p-values."""

    n_observations: int
    n_breaches: int
    nominal_probability: float
    observed_frequency: float
    transition_counts: NDArray[np.int64]
    probability_after_no_breach: float | None
    probability_after_breach: float | None
    lr_unconditional: float
    p_unconditional: float
    lr_independence: float | None
    p_independence: float | None
    lr_conditional: float | None
    p_conditional: float | None
    sparse_transitions: bool
    independence_identifiable: bool


def _bernoulli_log_likelihood(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    """Bernoulli log likelihood with correct boundary limits."""
    failures = trials - successes
    return float(xlogy(successes, probability) + xlogy(failures, 1.0 - probability))


def christoffersen_coverage_test(
    breaches: ArrayLike,
    *,
    nominal_probability: float,
) -> ChristoffersenResult:
    """Calculate LRuc, LRind and LRcc for a binary breach sequence.

    Independence is not identifiable if either preceding-state
    row has no observations. In that case LRind and LRcc are
    reported as None rather than fabricating a test statistic.
    """
    indicators = np.asarray(breaches)

    if indicators.ndim != 1 or indicators.size < 2 or indicators.dtype.kind != "b":
        raise ValueError(
            "breaches must be a 1D boolean array with at least two observations"
        )

    if isinstance(nominal_probability, (bool, np.bool_)):
        raise TypeError("nominal_probability must be in (0, 1)")

    if not np.isscalar(nominal_probability):
        raise TypeError("nominal_probability must be a scalar")

    try:
        probability = float(nominal_probability)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("nominal_probability must be numeric") from exc

    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("nominal_probability must be in (0, 1)")

    n = int(indicators.size)
    breaches_count = int(np.count_nonzero(indicators))
    frequency = breaches_count / n

    null_log = _bernoulli_log_likelihood(breaches_count, n, probability)
    fitted_log = _bernoulli_log_likelihood(breaches_count, n, frequency)

    lr_uc = max(0.0, 2.0 * (fitted_log - null_log))

    counts = np.zeros((2, 2), dtype=np.int64)
    for previous, current in pairwise(indicators):
        counts[int(previous), int(current)] += 1

    n00, n01 = (int(value) for value in counts[0])
    n10, n11 = (int(value) for value in counts[1])

    row0 = n00 + n01
    row1 = n10 + n11
    identifiable = row0 > 0 and row1 > 0

    probability_01 = n01 / row0 if row0 else None
    probability_11 = n11 / row1 if row1 else None

    # The pooled transition likelihood uses n - 1 transitions,
    # not the n observations used by LRuc.
    pooled_successes = n01 + n11
    pooled_probability = pooled_successes / (n - 1)

    sparse = bool(np.any(counts < 5))

    if identifiable:
        markov_log = _bernoulli_log_likelihood(
            n01, row0, probability_01
        ) + _bernoulli_log_likelihood(n11, row1, probability_11)
        independent_log = _bernoulli_log_likelihood(
            pooled_successes, n - 1, pooled_probability
        )

        lr_ind = max(0.0, 2.0 * (markov_log - independent_log))
        p_ind = float(chi2.sf(lr_ind, df=1))

        lr_cc = lr_uc + lr_ind
        p_cc = float(chi2.sf(lr_cc, df=2))
    else:
        lr_ind = None
        p_ind = None
        lr_cc = None
        p_cc = None

    return ChristoffersenResult(
        n_observations=n,
        n_breaches=breaches_count,
        nominal_probability=probability,
        observed_frequency=frequency,
        transition_counts=counts,
        probability_after_no_breach=probability_01,
        probability_after_breach=probability_11,
        lr_unconditional=lr_uc,
        p_unconditional=float(chi2.sf(lr_uc, df=1)),
        lr_independence=lr_ind,
        p_independence=p_ind,
        lr_conditional=lr_cc,
        p_conditional=p_cc,
        sparse_transitions=sparse,
        independence_identifiable=identifiable,
    )
