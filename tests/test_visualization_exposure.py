"""Regression tests for counterparty exposure visualizations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pyccr.exposure.profiles import calculate_exposure_profiles
from pyccr.visualization.exposure import (
    plot_exposure_distribution,
    plot_exposure_heatmap,
    plot_exposure_profiles,
)


@pytest.fixture
def profiles():
    times = np.array([0.0, 0.5, 1.5, 3.0])

    values = np.array(
        [
            [0.0, 10.0, 20.0, 0.0],
            [0.0, -5.0, 30.0, 0.0],
            [0.0, 15.0, -10.0, 0.0],
            [0.0, -20.0, 40.0, 0.0],
        ]
    )

    return calculate_exposure_profiles(
        times,
        values,
        confidence_level=0.95,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_profile_plot_returns_figure(profiles):
    fig = plot_exposure_profiles(profiles)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_profile_plot_contains_three_curves(profiles):
    fig = plot_exposure_profiles(profiles)
    ax = fig.axes[0]

    assert len(ax.lines) == 3

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        profiles.ee,
    )
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(),
        profiles.ene,
    )
    np.testing.assert_allclose(
        ax.lines[2].get_ydata(),
        profiles.pfe,
    )


def test_profile_plot_uses_simulation_times(profiles):
    fig = plot_exposure_profiles(profiles)

    for line in fig.axes[0].lines:
        np.testing.assert_allclose(
            line.get_xdata(),
            profiles.times,
        )


def test_profile_plot_custom_title(profiles):
    fig = plot_exposure_profiles(
        profiles,
        title="PyCCR exposure analysis",
    )

    assert fig.axes[0].get_title() == "PyCCR exposure analysis"


def test_profile_plot_displays_epe(profiles):
    fig = plot_exposure_profiles(profiles)

    assert any("EPE" in item.get_text() for item in fig.axes[0].texts)


def test_profile_plot_displays_confidence_level(profiles):
    fig = plot_exposure_profiles(profiles)

    labels = [line.get_label() for line in fig.axes[0].lines]

    assert any("95%" in label for label in labels)


@pytest.mark.parametrize("time_index", [0, 1, 2, 3, -1, -2, -4])
def test_distribution_valid_time_indices(profiles, time_index):
    fig = plot_exposure_distribution(
        profiles,
        time_index=time_index,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_distribution_default_selects_last_time(profiles):
    fig = plot_exposure_distribution(profiles)

    assert "3.000" in fig.axes[0].get_title()


def test_distribution_has_histogram(profiles):
    fig = plot_exposure_distribution(
        profiles,
        time_index=2,
        bins=8,
    )

    assert len(fig.axes[0].patches) > 0


def test_distribution_ee_marker_matches_profile(profiles):
    fig = plot_exposure_distribution(
        profiles,
        time_index=2,
    )

    ee_line = fig.axes[0].lines[0]

    np.testing.assert_allclose(
        ee_line.get_xdata(),
        [profiles.ee[2], profiles.ee[2]],
    )


def test_distribution_pfe_marker_matches_profile(profiles):
    fig = plot_exposure_distribution(
        profiles,
        time_index=2,
    )

    pfe_line = fig.axes[0].lines[1]

    np.testing.assert_allclose(
        pfe_line.get_xdata(),
        [profiles.pfe[2], profiles.pfe[2]],
    )


@pytest.mark.parametrize("time_index", [-5, 4, 100])
def test_distribution_rejects_out_of_range_index(
    profiles,
    time_index,
):
    with pytest.raises(IndexError):
        plot_exposure_distribution(
            profiles,
            time_index=time_index,
        )


@pytest.mark.parametrize(
    "time_index",
    [1.5, "1", True, None],
)
def test_distribution_rejects_noninteger_index(
    profiles,
    time_index,
):
    with pytest.raises(TypeError):
        plot_exposure_distribution(
            profiles,
            time_index=time_index,
        )


@pytest.mark.parametrize("bins", [0, -1, -10])
def test_distribution_rejects_nonpositive_bins(profiles, bins):
    with pytest.raises(ValueError):
        plot_exposure_distribution(
            profiles,
            bins=bins,
        )


@pytest.mark.parametrize("bins", [1.5, "10", True, None])
def test_distribution_rejects_noninteger_bins(profiles, bins):
    with pytest.raises(TypeError):
        plot_exposure_distribution(
            profiles,
            bins=bins,
        )


def test_heatmap_returns_figure(profiles):
    fig = plot_exposure_heatmap(profiles)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_heatmap_contains_correct_exposure_matrix(profiles):
    fig = plot_exposure_heatmap(profiles)

    matrix = fig.axes[0].collections[0].get_array()

    np.testing.assert_allclose(
        np.asarray(matrix),
        profiles.positive_exposures,
    )


def test_heatmap_respects_max_paths(profiles):
    fig = plot_exposure_heatmap(
        profiles,
        max_paths=2,
    )

    matrix = fig.axes[0].collections[0].get_array()

    assert np.asarray(matrix).shape == (2, 4)


def test_heatmap_caps_at_available_paths(profiles):
    fig = plot_exposure_heatmap(
        profiles,
        max_paths=100,
    )

    matrix = fig.axes[0].collections[0].get_array()

    assert np.asarray(matrix).shape == (4, 4)


@pytest.mark.parametrize("max_paths", [0, -1, -10])
def test_heatmap_rejects_nonpositive_max_paths(
    profiles,
    max_paths,
):
    with pytest.raises(ValueError):
        plot_exposure_heatmap(
            profiles,
            max_paths=max_paths,
        )


@pytest.mark.parametrize("max_paths", [1.5, "10", True, None])
def test_heatmap_rejects_noninteger_max_paths(
    profiles,
    max_paths,
):
    with pytest.raises(TypeError):
        plot_exposure_heatmap(
            profiles,
            max_paths=max_paths,
        )


@pytest.mark.parametrize(
    "plotter",
    [
        plot_exposure_profiles,
        plot_exposure_distribution,
        plot_exposure_heatmap,
    ],
)
def test_plotters_reject_invalid_profile_type(plotter):
    with pytest.raises(TypeError):
        plotter(None)


def test_plotting_does_not_modify_exposure_data(profiles):
    arrays = {
        name: getattr(profiles, name).copy()
        for name in (
            "times",
            "positive_exposures",
            "negative_exposures",
            "ee",
            "ene",
            "pfe",
        )
    }

    plot_exposure_profiles(profiles)
    plot_exposure_distribution(profiles, time_index=2)
    plot_exposure_heatmap(profiles)

    for name, original in arrays.items():
        np.testing.assert_array_equal(
            getattr(profiles, name),
            original,
        )


def test_single_time_grid_supported():
    profiles = calculate_exposure_profiles(
        np.array([0.0]),
        np.array([[10.0], [-5.0]]),
    )

    figures = [
        plot_exposure_profiles(profiles),
        plot_exposure_distribution(profiles),
        plot_exposure_heatmap(profiles),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)


def test_zero_exposure_supported():
    profiles = calculate_exposure_profiles(
        np.array([0.0, 1.0]),
        np.zeros((3, 2)),
    )

    figures = [
        plot_exposure_profiles(profiles),
        plot_exposure_distribution(profiles),
        plot_exposure_heatmap(profiles),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
