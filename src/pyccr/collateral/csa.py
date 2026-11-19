"""Validated bilateral variation-margin agreement configuration.

Time intervals are expressed in simulation-grid steps, not business days.
Amounts are in the same currency units as the portfolio value paths.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CSAConfig:
    """Simple bilateral CSA for variation margin.

    counterparty_threshold:
        Positive exposure permitted before the counterparty must post VM.

    own_threshold:
        Negative exposure permitted before we must post VM.

    minimum_transfer_amount:
        A margin transfer is initiated only when the absolute difference
        between target collateral and effective collateral plus pending
        transfers is at least this amount.

    call_frequency_steps:
        Calls occur at grid indices 0, frequency, 2*frequency, ...

    settlement_lag_steps:
        Number of grid intervals between a call and its settlement.
        Zero means settlement is effective at the call grid point.

    initial_vm_balance:
        Signed VM balance at grid index zero, before any margin call.
        Positive means collateral held; negative means collateral posted.
    """

    counterparty_threshold: float = 0.0
    own_threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    call_frequency_steps: int = 1
    settlement_lag_steps: int = 0
    initial_vm_balance: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "counterparty_threshold",
            "own_threshold",
            "minimum_transfer_amount",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        for name in ("call_frequency_steps", "settlement_lag_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")

        if self.call_frequency_steps < 1:
            raise ValueError("call_frequency_steps must be >= 1")

        if self.settlement_lag_steps < 0:
            raise ValueError("settlement_lag_steps must be >= 0")

        if not isinstance(self.initial_vm_balance, (int, float)):
            raise TypeError("initial_vm_balance must be numeric")

        if not math.isfinite(self.initial_vm_balance):
            raise ValueError("initial_vm_balance must be finite")
