"""Regression tests for bilateral XVA sensitivity heatmaps."""

from dataclasses import fields

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pyccr.visualization.xva_sensitivity import (
    XVA_SENSITIVITY_METRICS,
    plot_xva_sensitivity_heatmap,
)
from pyccr.xva.sensitivity import (
    BilateralXVASensitivityResult,
    calculate_bilateral_xva_sensitivity,
)


@pytest.fixture
def result():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    ee = np.array([0.0, 100.0, 100.0, 100.0])
    ene = np.array([0.0, 50.0, 50.0, 50.0])

    return calculate_bilateral_xva_sensitivity(
        times=times,
        expected_exposure=ee,
        expected_negative_exposure=ene,
        discount_factors=np.ones_like(times),
        counterparty_spreads_bps=[25.0, 120.0, 200.0],
        own_spreads_bps=[50.0, 180.0],
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def replace_field(result, name, value):
    values = {
        field.name: getattr(result, field.name)
        for field in fields(BilateralXVASensitivityResult)
    }
    values[name] = value
    return BilateralXVASensitivityResult(**values)


def heatmap_values(fig):
    return np.asarray(fig.axes[0].collections[0].get_array()).reshape(2, 3)


def test_metric_catalog_contains_nine_unique_metrics():
    assert len(XVA_SENSITIVITY_METRICS) == 9
    assert len(set(XVA_SENSITIVITY_METRICS)) == 9


def test_default_heatmap_returns_figure(result):
    fig = plot_xva_sensitivity_heatmap(result)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_default_metric_is_first_to_default_net_xva(result):
    fig = plot_xva_sensitivity_heatmap(result)

    np.testing.assert_allclose(
        heatmap_values(fig),
        result.first_to_default_net_xva,
    )


@pytest.mark.parametrize("metric", XVA_SENSITIVITY_METRICS)
def test_each_metric_plots_correct_matrix(result, metric):
    fig = plot_xva_sensitivity_heatmap(result, metric=metric)

    np.testing.assert_allclose(
        heatmap_values(fig),
        getattr(result, metric),
    )


@pytest.mark.parametrize("metric", XVA_SENSITIVITY_METRICS)
def test_each_metric_returns_figure(result, metric):
    fig = plot_xva_sensitivity_heatmap(result, metric=metric)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_counterparty_spreads_are_columns(result):
    fig = plot_xva_sensitivity_heatmap(result)
    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]

    assert labels == ["25", "120", "200"]


def test_own_spreads_are_rows(result):
    fig = plot_xva_sensitivity_heatmap(result)
    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]

    assert labels == ["50", "180"]


def test_axis_labels(result):
    fig = plot_xva_sensitivity_heatmap(result)
    ax = fig.axes[0]

    assert ax.get_xlabel() == "Counterparty CDS spread (bps)"
    assert ax.get_ylabel() == "Own CDS spread (bps)"


def test_default_title(result):
    fig = plot_xva_sensitivity_heatmap(result)

    assert fig.axes[0].get_title() == ("First To Default Net Xva")


def test_custom_title(result):
    fig = plot_xva_sensitivity_heatmap(
        result,
        title="PyCCR bilateral credit sensitivity",
    )

    assert fig.axes[0].get_title() == ("PyCCR bilateral credit sensitivity")


def test_colorbar_label(result):
    fig = plot_xva_sensitivity_heatmap(result)

    assert fig.axes[1].get_ylabel() == ("XVA adjustment (domestic currency)")


@pytest.mark.parametrize(
    "metric",
    [
        "naive_net_xva",
        "first_to_default_net_xva",
        "cva_correction",
        "dva_correction",
        "net_xva_correction",
    ],
)
def test_signed_metrics_have_symmetric_color_limits(
    result,
    metric,
):
    fig = plot_xva_sensitivity_heatmap(result, metric=metric)
    vmin, vmax = fig.axes[0].collections[0].get_clim()

    assert vmin < 0.0
    assert vmax > 0.0
    assert vmin == pytest.approx(-vmax)


@pytest.mark.parametrize(
    "metric",
    [
        "naive_cva",
        "naive_dva",
        "first_to_default_cva",
        "first_to_default_dva",
    ],
)
def test_unsigned_metrics_start_color_scale_at_zero(
    result,
    metric,
):
    fig = plot_xva_sensitivity_heatmap(result, metric=metric)
    vmin, _ = fig.axes[0].collections[0].get_clim()

    assert vmin == pytest.approx(0.0)


@pytest.mark.parametrize(
    "metric",
    [
        "naive_net_xva",
        "first_to_default_net_xva",
        "cva_correction",
        "dva_correction",
        "net_xva_correction",
    ],
)
def test_zero_signed_metrics_supported(result, metric):
    zero = replace_field(
        result,
        metric,
        np.zeros((2, 3)),
    )

    fig = plot_xva_sensitivity_heatmap(zero, metric=metric)
    vmin, vmax = fig.axes[0].collections[0].get_clim()

    assert vmin == pytest.approx(-1.0)
    assert vmax == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric",
    ["invalid", "", None, 123, True],
)
def test_invalid_metric_rejected(result, metric):
    with pytest.raises(ValueError, match="metric"):
        plot_xva_sensitivity_heatmap(result, metric=metric)


@pytest.mark.parametrize(
    "title",
    [123, False, [], {}],
)
def test_invalid_title_rejected(result, title):
    with pytest.raises(TypeError, match="title"):
        plot_xva_sensitivity_heatmap(result, title=title)


@pytest.mark.parametrize(
    "invalid",
    [None, {}, np.zeros((2, 3)), "not a result"],
)
def test_invalid_result_type_rejected(invalid):
    with pytest.raises(TypeError, match="result"):
        plot_xva_sensitivity_heatmap(invalid)


@pytest.mark.parametrize(
    "field_name",
    ["counterparty_spreads_bps", "own_spreads_bps"],
)
def test_empty_spread_axis_rejected(result, field_name):
    invalid = replace_field(
        result,
        field_name,
        np.array([]),
    )

    with pytest.raises(ValueError, match=field_name):
        plot_xva_sensitivity_heatmap(invalid)


@pytest.mark.parametrize(
    "field_name",
    ["counterparty_spreads_bps", "own_spreads_bps"],
)
def test_multidimensional_spread_axis_rejected(
    result,
    field_name,
):
    invalid = replace_field(
        result,
        field_name,
        np.array([[25.0, 120.0]]),
    )

    with pytest.raises(ValueError, match=field_name):
        plot_xva_sensitivity_heatmap(invalid)


@pytest.mark.parametrize(
    "bad_value",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_counterparty_spreads_rejected(
    result,
    bad_value,
):
    spreads = result.counterparty_spreads_bps.copy()
    spreads[0] = bad_value

    invalid = replace_field(
        result,
        "counterparty_spreads_bps",
        spreads,
    )

    with pytest.raises(ValueError, match="counterparty"):
        plot_xva_sensitivity_heatmap(invalid)


@pytest.mark.parametrize(
    "bad_value",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_own_spreads_rejected(result, bad_value):
    spreads = result.own_spreads_bps.copy()
    spreads[0] = bad_value

    invalid = replace_field(
        result,
        "own_spreads_bps",
        spreads,
    )

    with pytest.raises(ValueError, match="own_spreads"):
        plot_xva_sensitivity_heatmap(invalid)


@pytest.mark.parametrize("metric", XVA_SENSITIVITY_METRICS)
def test_wrong_matrix_shape_rejected(result, metric):
    invalid = replace_field(
        result,
        metric,
        np.zeros((3, 2)),
    )

    with pytest.raises(ValueError, match=metric):
        plot_xva_sensitivity_heatmap(invalid)


@pytest.mark.parametrize("metric", XVA_SENSITIVITY_METRICS)
def test_nonfinite_matrix_rejected(result, metric):
    values = getattr(result, metric).copy()
    values[0, 0] = np.nan

    invalid = replace_field(result, metric, values)

    with pytest.raises(ValueError, match=metric):
        plot_xva_sensitivity_heatmap(invalid)


def test_all_matrices_are_validated_not_only_selected(result):
    invalid = replace_field(
        result,
        "naive_cva",
        np.full((2, 3), np.nan),
    )

    with pytest.raises(ValueError, match="naive_cva"):
        plot_xva_sensitivity_heatmap(
            invalid,
            metric="first_to_default_net_xva",
        )


def test_plotting_does_not_mutate_result(result):
    before = {
        field.name: np.array(
            getattr(result, field.name),
            copy=True,
        )
        for field in fields(BilateralXVASensitivityResult)
    }

    for metric in XVA_SENSITIVITY_METRICS:
        plot_xva_sensitivity_heatmap(result, metric=metric)

    for field in fields(BilateralXVASensitivityResult):
        np.testing.assert_array_equal(
            getattr(result, field.name),
            before[field.name],
        )


def test_single_cell_grid_supported():
    result = calculate_bilateral_xva_sensitivity(
        times=[0.0, 1.0],
        expected_exposure=[0.0, 100.0],
        expected_negative_exposure=[0.0, 50.0],
        discount_factors=[1.0, 1.0],
        counterparty_spreads_bps=[120.0],
        own_spreads_bps=[180.0],
    )

    fig = plot_xva_sensitivity_heatmap(result)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2

    np.testing.assert_allclose(
        np.asarray(fig.axes[0].collections[0].get_array()).reshape(1, 1),
        result.first_to_default_net_xva,
    )


def test_zero_exposure_grid_supported():
    result = calculate_bilateral_xva_sensitivity(
        times=[0.0, 1.0],
        expected_exposure=[0.0, 0.0],
        expected_negative_exposure=[0.0, 0.0],
        discount_factors=[1.0, 1.0],
        counterparty_spreads_bps=[0.0, 120.0],
        own_spreads_bps=[0.0, 180.0],
    )

    fig = plot_xva_sensitivity_heatmap(result)

    assert isinstance(fig, Figure)
    np.testing.assert_allclose(
        np.asarray(fig.axes[0].collections[0].get_array()),
        0.0,
    )
