"""Pathwise margin-period-of-risk closeout exposures.

Collateral is frozen at each hypothetical default index.
Portfolio value is evaluated at the corresponding closeout index.
MPOR is measured in simulation-grid steps.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MPORResult:
    """Pathwise exposures indexed by hypothetical default time."""

    default_indices: NDArray[np.int64]
    closeout_indices: NDArray[np.int64]
    frozen_vm_balance: FloatArray
    closeout_values: FloatArray
    positive_exposure: FloatArray
    negative_exposure: FloatArray


def calculate_mpor_exposures(
    portfolio_values: FloatArray,
    settled_vm_balance: FloatArray,
    *,
    mpor_steps: int,
) -> MPORResult:
    """Calculate bilateral closeout exposures with frozen settled VM.

    Inputs have shape (n_paths, n_times). Output exposure arrays
    have shape (n_paths, n_times - mpor_steps).

    A positive VM balance represents collateral held.
    Pending transfers must not be included in settled_vm_balance.

    No default simulation, IM, interest, or liquidation costs
    are modelled.
    """
    if isinstance(mpor_steps, (bool, np.bool_)) or not isinstance(
        mpor_steps, (int, np.integer)
    ):
        raise TypeError("mpor_steps must be an integer")

    if mpor_steps < 0:
        raise ValueError("mpor_steps must be nonnegative")

    values = np.asarray(portfolio_values, dtype=np.float64)
    vm = np.asarray(settled_vm_balance, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("portfolio_values must have shape (n_paths, n_times)")

    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("portfolio_values axes must be nonempty")

    if vm.shape != values.shape:
        raise ValueError("settled_vm_balance must match portfolio_values shape")

    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio_values must be finite")

    if not np.all(np.isfinite(vm)):
        raise ValueError("settled_vm_balance must be finite")

    n_times = values.shape[1]

    if mpor_steps >= n_times:
        raise ValueError("MPOR closeout exceeds the simulation horizon")

    n_defaults = n_times - mpor_steps

    default_indices = np.arange(n_defaults, dtype=np.int64)
    closeout_indices = default_indices + mpor_steps

    frozen_vm = vm[:, :n_defaults].copy()
    closeout_values = values[:, mpor_steps:].copy()

    residual = closeout_values - frozen_vm

    positive = np.maximum(residual, 0.0)
    negative = np.maximum(-residual, 0.0)

    return MPORResult(
        default_indices=default_indices,
        closeout_indices=closeout_indices,
        frozen_vm_balance=frozen_vm,
        closeout_values=closeout_values,
        positive_exposure=positive,
        negative_exposure=negative,
    )
