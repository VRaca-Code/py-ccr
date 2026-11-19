"""Bilateral segregated initial margin for MPOR closeout exposure.

Received and posted initial margin are separate nonnegative balances.

Received IM may reduce counterparty-default positive exposure.
Posted IM may reduce own-default negative exposure.

IM is frozen at each hypothetical default index. This module
assumes the specified IM is eligible and legally enforceable for
the corresponding closeout loss.

It does not model IM calls, custody, segregation mechanics,
funding costs, interest, or IM methodology calibration.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyccr.collateral.mpor import MPORResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InitialMarginResult:
    """Bilateral IM-adjusted pathwise MPOR closeout exposures."""

    default_indices: NDArray[np.int64]
    closeout_indices: NDArray[np.int64]
    frozen_vm_balance: FloatArray
    frozen_received_im: FloatArray
    frozen_posted_im: FloatArray
    closeout_values: FloatArray
    positive_exposure: FloatArray
    negative_exposure: FloatArray


def _im_balance(
    balance: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int],
) -> FloatArray:
    """Normalize scalar or full-grid IM balances."""

    array = np.asarray(balance, dtype=np.float64)

    if array.ndim == 0:
        if not np.isfinite(array):
            raise ValueError(f"{name} must be finite")
        if array < 0.0:
            raise ValueError(f"{name} must be nonnegative")

        return np.full(shape, float(array), dtype=np.float64)

    if array.shape != shape:
        raise ValueError(f"{name} must be a scalar or have shape {shape}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")

    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")

    return array.copy()


def calculate_initial_margin_exposures(
    mpor_result: MPORResult,
    *,
    received_im: ArrayLike = 0.0,
    posted_im: ArrayLike = 0.0,
) -> InitialMarginResult:
    """Adjust frozen-VM MPOR exposures for segregated bilateral IM.

    IM balances may be nonnegative scalars or full-grid arrays
    with shape (n_paths, n_times).

    For each hypothetical default index d, IM is frozen at d,
    while portfolio closeout value comes from d + mpor_steps.

    The MPOR result is not modified.

    Positive exposure:
        max(closeout_value - frozen_vm - received_im_at_default, 0)

    Negative exposure:
        max(frozen_vm - closeout_value - posted_im_at_default, 0)
    """

    if not isinstance(mpor_result, MPORResult):
        raise TypeError("mpor_result must be an MPORResult")

    default_indices = np.asarray(
        mpor_result.default_indices,
        dtype=np.int64,
    )
    closeout_indices = np.asarray(
        mpor_result.closeout_indices,
        dtype=np.int64,
    )

    values = np.asarray(
        mpor_result.closeout_values,
        dtype=np.float64,
    )
    vm = np.asarray(
        mpor_result.frozen_vm_balance,
        dtype=np.float64,
    )

    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("MPOR closeout values must be nonempty 2D")

    if vm.shape != values.shape:
        raise ValueError("MPOR VM must match closeout values")

    n_paths, n_defaults = values.shape

    if default_indices.shape != (n_defaults,):
        raise ValueError("MPOR default indices have invalid shape")

    if closeout_indices.shape != (n_defaults,):
        raise ValueError("MPOR closeout indices have invalid shape")

    if not np.all(np.isfinite(values)):
        raise ValueError("MPOR closeout values must be finite")

    if not np.all(np.isfinite(vm)):
        raise ValueError("MPOR frozen VM must be finite")

    if default_indices[0] != 0 or not np.array_equal(
        default_indices,
        np.arange(n_defaults, dtype=np.int64),
    ):
        raise ValueError("MPOR default indices must be consecutive")

    if np.any(closeout_indices < default_indices):
        raise ValueError("MPOR closeout cannot precede default")

    if not np.all(
        closeout_indices - default_indices == closeout_indices[0] - default_indices[0]
    ):
        raise ValueError("MPOR horizon must be constant")

    n_times = int(closeout_indices[-1]) + 1

    if n_times < n_defaults:
        raise ValueError("Invalid MPOR simulation horizon")

    full_shape = (n_paths, n_times)

    received = _im_balance(
        received_im,
        name="received_im",
        shape=full_shape,
    )

    posted = _im_balance(
        posted_im,
        name="posted_im",
        shape=full_shape,
    )

    frozen_received = received[:, default_indices].copy()
    frozen_posted = posted[:, default_indices].copy()

    residual = values - vm

    positive = np.maximum(
        residual - frozen_received,
        0.0,
    )

    negative = np.maximum(
        -residual - frozen_posted,
        0.0,
    )

    return InitialMarginResult(
        default_indices=default_indices.copy(),
        closeout_indices=closeout_indices.copy(),
        frozen_vm_balance=vm.copy(),
        frozen_received_im=frozen_received,
        frozen_posted_im=frozen_posted,
        closeout_values=values.copy(),
        positive_exposure=positive,
        negative_exposure=negative,
    )
