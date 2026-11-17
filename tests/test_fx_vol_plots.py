"""Regression tests for FX calendar-variance visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from pyccr.market.fx_vol_plots import plot_fx_calendar_variance
from pyccr.market.fx_vol_surface import FXVolSurface


@pytest.fixture(autouse=True)
def close_figures():
    """Prevent Matplotlib figure leakage between tests."""
    yield
    plt.close("all")


def make_surface(
    maturities=(1.0, 2.0, 3.0),
    strikes=(0.9, 1.1),
    volatilities=(
        (0.20, 0.25),
        (0.20, 0.25),
        (0.20, 0.25),
    ),
):
    return FXVolSurface(
        maturities=maturities,
        strikes=strikes,
        volatilities=volatilities,
    )


def test_returns_matplotlib_figure_and_axes():
    fig, ax = plot_fx_calendar_variance(make_surface())

    assert isinstance(fig, Figure)
    assert ax.figure is fig


def test_one_line_per_quoted_strike():
    fig, ax = plot_fx_calendar_variance(make_surface())

    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == [
        "K=0.9",
        "K=1.1",
    ]
    assert fig is ax.figure


def test_plotted_total_variance_values():
    surface = make_surface()
    _, ax = plot_fx_calendar_variance(surface)

    np.testing.assert_allclose(
        ax.lines[0].get_xdata(),
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        [0.04, 0.08, 0.12],
    )
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(),
        [0.0625, 0.125, 0.1875],
    )


def test_clean_surface_has_no_violation_markers():
    _, ax = plot_fx_calendar_variance(make_surface())

    assert not any(
        isinstance(collection, PathCollection) for collection in ax.collections
    )


def test_decreasing_variance_creates_violation_marker():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.30,), (0.10,)),
    )

    _, ax = plot_fx_calendar_variance(surface)

    markers = [
        collection
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    ]

    assert len(markers) == 1
    np.testing.assert_allclose(
        markers[0].get_offsets(),
        [[2.0, 0.02]],
    )


def test_multiple_violations_create_multiple_markers():
    surface = make_surface(
        maturities=(1.0, 2.0, 3.0),
        strikes=(0.9, 1.1),
        volatilities=(
            (0.30, 0.10),
            (0.10, 0.30),
            (0.30, 0.10),
        ),
    )

    _, ax = plot_fx_calendar_variance(surface)

    markers = [
        collection
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    ]

    assert len(markers) == 2

    offsets = [marker.get_offsets()[0] for marker in markers]

    np.testing.assert_allclose(
        offsets,
        [[2.0, 0.02], [3.0, 0.03]],
    )


def test_reuses_supplied_axes():
    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_fx_calendar_variance(
        make_surface(),
        ax=ax,
    )

    assert returned_fig is fig
    assert returned_ax is ax


def test_plot_labels_and_title():
    _, ax = plot_fx_calendar_variance(make_surface())

    assert ax.get_xlabel() == "Maturity (years)"
    assert ax.get_ylabel() == "Total implied variance"
    assert "calendar-variance diagnostics" in ax.get_title()


def test_legend_contains_quoted_strikes():
    _, ax = plot_fx_calendar_variance(make_surface())

    legend = ax.get_legend()
    assert legend is not None

    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["K=0.9", "K=1.1"]


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        plot_fx_calendar_variance(
            make_surface(),
            tolerance=tolerance,
        )


def test_single_maturity_surface_plots():
    surface = make_surface(
        maturities=(1.0,),
        strikes=(1.0,),
        volatilities=((0.20,),),
    )

    _, ax = plot_fx_calendar_variance(surface)

    assert len(ax.lines) == 1
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        [0.04],
    )


def test_zero_volatility_surface_plots():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.0,), (0.0,)),
    )

    _, ax = plot_fx_calendar_variance(surface)

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        [0.0, 0.0],
    )


def test_surface_is_not_modified():
    surface = make_surface()

    before = (
        surface.maturities,
        surface.strikes,
        surface.volatilities,
    )

    plot_fx_calendar_variance(surface)

    after = (
        surface.maturities,
        surface.strikes,
        surface.volatilities,
    )

    assert after == before
