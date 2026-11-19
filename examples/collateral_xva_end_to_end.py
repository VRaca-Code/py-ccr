"""Reproducible end-to-end collateral-aware XVA example.

Run from the repository root after installing PyCCR:

    python examples/collateral_xva_end_to_end.py

This example demonstrates bilateral variation margin, frozen
collateral during MPOR, segregated initial margin, exposure profiles,
and unilateral/bilateral/first-to-default XVA.

It is an illustrative deterministic scenario, not a calibrated
market-risk simulation or a production CSA implementation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyccr.collateral.csa import CSAConfig
from pyccr.credit.hazard_curve import ConstantHazardCurve
from pyccr.exposure.collateral_profiles import (
    CollateralExposureResult,
    calculate_collateral_exposure_profiles,
)
from pyccr.xva.collateral_xva import (
    CollateralXVAResult,
    calculate_collateral_xva,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExampleResult:
    """Inputs and outputs for the reproducible example."""

    simulation_times: FloatArray
    portfolio_values: FloatArray
    discount_factors: FloatArray
    exposures: CollateralExposureResult
    xva: CollateralXVAResult


def run_example() -> ExampleResult:
    """Execute the complete collateral-aware exposure and XVA workflow."""
    simulation_times = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float64,
    )

    # Each row represents one illustrative portfolio-value path.
    # Positive values mean the counterparty owes us.
    portfolio_values = np.array(
        [
            [0.0, 10.0, 14.0, 7.0, 3.0],
            [0.0, -8.0, -12.0, -5.0, -2.0],
            [0.0, 5.0, -3.0, 9.0, 11.0],
        ],
        dtype=np.float64,
    )

    csa = CSAConfig(
        counterparty_threshold=2.0,
        own_threshold=3.0,
        minimum_transfer_amount=1.0,
        call_frequency_steps=1,
        settlement_lag_steps=0,
    )

    # One time step of MPOR means that collateral is frozen at
    # hypothetical default and closeout is valued one step later.
    exposures = calculate_collateral_exposure_profiles(
        simulation_times,
        portfolio_values,
        csa,
        mpor_steps=1,
        received_im=1.5,
        posted_im=2.0,
        confidence_level=0.95,
    )

    # IMPORTANT:
    # XVA discount factors align to default times, NOT closeout times.
    # The last default time is 0.75, while its closeout is at 1.0.
    default_times = exposures.profiles.times

    discount_factors = np.exp(-0.04 * default_times)

    xva = calculate_collateral_xva(
        exposures,
        discount_factors,
        ConstantHazardCurve(0.03),
        ConstantHazardCurve(0.02),
        counterparty_recovery_rate=0.4,
        own_recovery_rate=0.4,
    )

    return ExampleResult(
        simulation_times=simulation_times.copy(),
        portfolio_values=portfolio_values.copy(),
        discount_factors=discount_factors.copy(),
        exposures=exposures,
        xva=xva,
    )


def main() -> None:
    """Print a concise numerical summary of the example."""
    result = run_example()

    profiles = result.exposures.profiles
    xva = result.xva

    print("PyCCR — Collateral-Aware XVA End-to-End Example")
    print("=" * 49)

    print(
        "Simulation maturity:",
        f"{result.simulation_times[-1]:.2f} years",
    )
    print(
        "Last hypothetical default time:",
        f"{profiles.times[-1]:.2f} years",
    )
    print(
        "Last MPOR closeout time:",
        f"{result.exposures.closeout_times[-1]:.2f} years",
    )

    print(
        "Expected positive exposure:",
        np.array2string(profiles.ee, precision=6),
    )
    print(
        "Expected negative exposure:",
        np.array2string(profiles.ene, precision=6),
    )
    print(
        "95% potential future exposure:",
        np.array2string(profiles.pfe, precision=6),
    )

    print(f"CVA: {xva.cva_result.cva:.12f}")
    print(f"DVA: {xva.dva_result.dva:.12f}")

    print(
        "Bilateral net XVA:",
        f"{xva.bilateral_result.net_xva:.12f}",
    )
    print(
        "First-to-default net XVA:",
        f"{xva.first_to_default_result.net_xva:.12f}",
    )

    print()
    print(
        "Convention: MPOR closeout exposures are assigned "
        "to hypothetical default times."
    )
    print(
        "Limitation: The final default time is earlier than "
        "simulation maturity when MPOR is positive."
    )


if __name__ == "__main__":
    main()
