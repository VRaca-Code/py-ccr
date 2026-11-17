"""Visualization of simulated counterparty exposure profiles."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from pyccr.exposure.profiles import ExposureProfiles


def _validate_profiles(profiles: ExposureProfiles) -> None:
    """Validate the data needed by exposure plotting functions."""
    if not isinstance(profiles, ExposureProfiles):
        raise TypeError("profiles must be an ExposureProfiles instance")

    times = np.asarray(profiles.times)
    positive = np.asarray(profiles.positive_exposures)
    negative = np.asarray(profiles.negative_exposures)
    ee = np.asarray(profiles.ee)
    ene = np.asarray(profiles.ene)
    pfe = np.asarray(profiles.pfe)

    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty 1D array")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")

    if positive.ndim != 2 or positive.shape[0] == 0:
        raise ValueError("positive exposures must be a nonempty 2D array")

    if positive.shape[1] != times.size:
        raise ValueError("exposure time dimension must match times")

    if negative.shape != positive.shape:
        raise ValueError("negative exposure shape must match positive exposure")

    if any(array.shape != times.shape for array in (ee, ene, pfe)):
        raise ValueError("profile shapes must match times")

    if not all(
        np.all(np.isfinite(array)) for array in (positive, negative, ee, ene, pfe)
    ):
        raise ValueError("exposure data must be finite")

    if not np.isfinite(profiles.epe):
        raise ValueError("EPE must be finite")

    if not 0.0 < profiles.confidence_level < 1.0:
        raise ValueError("confidence level must be between 0 and 1")


def plot_exposure_profiles(
    profiles: ExposureProfiles,
    *,
    title: str = "Counterparty exposure profiles",
) -> Figure:
    """Plot expected exposure, negative exposure, and potential exposure."""
    _validate_profiles(profiles)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        profiles.times,
        profiles.ee,
        label="Expected exposure (EE)",
        linewidth=2.2,
    )
    ax.plot(
        profiles.times,
        profiles.ene,
        label="Expected negative exposure (ENE)",
        linewidth=2.0,
        linestyle="--",
    )
    ax.plot(
        profiles.times,
        profiles.pfe,
        label=f"PFE ({profiles.confidence_level:.0%})",
        linewidth=2.0,
        linestyle="-.",
    )

    ax.set_title(title)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure (domestic currency)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax.text(
        0.98,
        0.97,
        f"EPE = {profiles.epe:,.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    return fig


def plot_exposure_distribution(
    profiles: ExposureProfiles,
    *,
    time_index: int = -1,
    bins: int = 30,
) -> Figure:
    """Plot positive exposure distribution at one simulation time."""
    _validate_profiles(profiles)

    if isinstance(time_index, (bool, np.bool_)) or not isinstance(
        time_index, (int, np.integer)
    ):
        raise TypeError("time_index must be an integer")

    n_times = profiles.times.size

    if not -n_times <= time_index < n_times:
        raise IndexError("time_index is outside the simulation grid")

    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)):
        raise TypeError("bins must be an integer")

    if bins < 1:
        raise ValueError("bins must be positive")

    index = int(time_index) % n_times
    values = profiles.positive_exposures[:, index]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.hist(
        values,
        bins=int(bins),
        alpha=0.75,
        edgecolor="white",
        label="Simulated positive exposure",
    )

    ax.axvline(
        profiles.ee[index],
        color="darkgreen",
        linewidth=2,
        label=f"EE = {profiles.ee[index]:,.4f}",
    )
    ax.axvline(
        profiles.pfe[index],
        color="darkred",
        linewidth=2,
        linestyle="--",
        label=(f"PFE {profiles.confidence_level:.0%} = {profiles.pfe[index]:,.4f}"),
    )

    ax.set_title(f"Positive exposure distribution at t = {profiles.times[index]:.3f}")
    ax.set_xlabel("Positive exposure (domestic currency)")
    ax.set_ylabel("Number of simulation paths")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_exposure_heatmap(
    profiles: ExposureProfiles,
    *,
    max_paths: int = 100,
) -> Figure:
    """Plot pathwise positive exposures on the simulation time grid."""
    _validate_profiles(profiles)

    if isinstance(max_paths, (bool, np.bool_)) or not isinstance(
        max_paths, (int, np.integer)
    ):
        raise TypeError("max_paths must be an integer")

    if max_paths < 1:
        raise ValueError("max_paths must be positive")

    values = profiles.positive_exposures[:max_paths, :]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        values,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Positive exposure (domestic currency)"},
        xticklabels=False,
        yticklabels=False,
    )

    # Put time labels at actual grid columns, including both endpoints.
    n_times = profiles.times.size
    tick_indices = np.unique(np.linspace(0, n_times - 1, min(n_times, 8), dtype=int))

    ax.set_xticks(tick_indices + 0.5)
    ax.set_xticklabels([f"{profiles.times[i]:.2f}" for i in tick_indices])

    ax.set_title(f"Pathwise positive exposure ({values.shape[0]} simulation paths)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Simulation path")

    fig.tight_layout()
    return fig
