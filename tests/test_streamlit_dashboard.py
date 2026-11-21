"""Focused tests for Streamlit dashboard Plotly figures.

Extract chart functions without importing or running the Streamlit app.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

CHART_FUNCTIONS = {
    "exposure_chart",
    "trade_chart",
    "xva_chart",
    "sensitivity_chart",
}


@pytest.fixture(scope="module")
def charts():
    """Compile chart functions directly from the dashboard source."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in CHART_FUNCTIONS
    ]

    assert {node.name for node in functions} == CHART_FUNCTIONS

    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)

    from app.dashboard_theme import CHART_COLORS, style_figure
    namespace = {
        "go": go,
        "CHART_COLORS": CHART_COLORS,
        "style_figure": style_figure,
    }
    # Only four allowlisted chart functions are compiled; the Streamlit UI is not run.
    exec(compile(module, str(APP_PATH), "exec"), namespace)  # noqa: S102

    return namespace


@pytest.fixture
def sample_result():
    """Small synthetic result with known dimensions and signed values."""
    times = np.array([0.0, 0.5, 1.0])

    trade_values = np.array(
        [
            [[10.0, 20.0, -5.0], [10.0, 30.0, -15.0]],
            [[-4.0, 8.0, 5.0], [-6.0, 12.0, 15.0]],
        ]
    )

    portfolio_values = trade_values.sum(axis=0)

    exposures = SimpleNamespace(
        ee=np.array([5.0, 35.0, 10.0]),
        ene=np.array([0.0, 0.0, 0.0]),
        pfe=np.array([6.0, 40.0, 15.0]),
    )

    standalone = SimpleNamespace(
        cva=120.0,
        dva=40.0,
        net_xva=-80.0,
    )

    first_to_default = SimpleNamespace(
        cva=110.0,
        dva=35.0,
        net_xva=-75.0,
    )

    sensitivity = SimpleNamespace(
        counterparty_spreads_bps=np.array([50.0, 100.0, 200.0]),
        own_spreads_bps=np.array([25.0, 75.0]),
        first_to_default_net_xva=np.array([[-10.0, -20.0, -40.0], [5.0, -5.0, -25.0]]),
        first_to_default_cva=np.array([[10.0, 20.0, 40.0], [10.0, 20.0, 40.0]]),
        first_to_default_dva=np.array([[0.0, 0.0, 0.0], [15.0, 15.0, 15.0]]),
        naive_net_xva=np.array([[-12.0, -24.0, -48.0], [6.0, -6.0, -30.0]]),
        cva_correction=np.array([[-2.0, -4.0, -8.0], [-2.0, -4.0, -8.0]]),
        dva_correction=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        net_xva_correction=np.array([[2.0, 4.0, 8.0], [-1.0, 1.0, 5.0]]),
    )

    return SimpleNamespace(
        times=times,
        trade_values=trade_values,
        portfolio_values=portfolio_values,
        exposures=exposures,
        standalone_xva=standalone,
        first_to_default_xva=first_to_default,
        sensitivity=sensitivity,
    )


def test_dashboard_source_contains_expected_chart_functions(charts):
    assert CHART_FUNCTIONS.issubset(charts)


def test_exposure_chart_has_three_correct_series(charts, sample_result):
    fig = charts["exposure_chart"](sample_result)

    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["EE", "ENE", "PFE"]

    for trace, expected in zip(
        fig.data,
        (
            sample_result.exposures.ee,
            sample_result.exposures.ene,
            sample_result.exposures.pfe,
        ),
        strict=True,
    ):
        np.testing.assert_allclose(trace.x, sample_result.times)
        np.testing.assert_allclose(trace.y, expected)


def test_exposure_chart_axes(charts, sample_result):
    fig = charts["exposure_chart"](sample_result)

    assert fig.layout.xaxis.title.text == "Time (years)"
    assert fig.layout.yaxis.title.text == "Exposure (EUR)"


def test_trade_chart_has_expected_names(charts, sample_result):
    fig = charts["trade_chart"](sample_result)

    assert [trace.name for trace in fig.data] == [
        "Receiver IRS",
        "Payer swaption",
        "Combined portfolio",
    ]


def test_trade_chart_preserves_signed_mtm(charts, sample_result):
    fig = charts["trade_chart"](sample_result)

    np.testing.assert_allclose(
        fig.data[0].y,
        sample_result.trade_values[0].mean(axis=0),
    )
    np.testing.assert_allclose(
        fig.data[1].y,
        sample_result.trade_values[1].mean(axis=0),
    )
    np.testing.assert_allclose(
        fig.data[2].y,
        sample_result.portfolio_values.mean(axis=0),
    )

    assert fig.data[0].y[-1] < 0.0


def test_trade_chart_uses_same_time_grid(charts, sample_result):
    fig = charts["trade_chart"](sample_result)

    for trace in fig.data:
        np.testing.assert_allclose(trace.x, sample_result.times)


def test_xva_chart_has_two_scenarios(charts, sample_result):
    fig = charts["xva_chart"](sample_result)

    assert [trace.name for trace in fig.data] == [
        "Standalone",
        "First-to-default",
    ]
    assert fig.layout.barmode == "group"


def test_xva_chart_preserves_signed_net_adjustment(charts, sample_result):
    fig = charts["xva_chart"](sample_result)

    assert list(fig.data[0].x) == ["CVA", "DVA", "Net XVA"]
    np.testing.assert_allclose(fig.data[0].y, [120.0, 40.0, -80.0])
    np.testing.assert_allclose(fig.data[1].y, [110.0, 35.0, -75.0])


@pytest.mark.parametrize(
    "metric",
    [
        "first_to_default_net_xva",
        "first_to_default_cva",
        "first_to_default_dva",
        "naive_net_xva",
        "cva_correction",
        "dva_correction",
        "net_xva_correction",
    ],
)
def test_sensitivity_chart_preserves_matrix(charts, sample_result, metric):
    fig = charts["sensitivity_chart"](sample_result, metric)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    trace = fig.data[0]
    expected = getattr(sample_result.sensitivity, metric)

    np.testing.assert_allclose(trace.z, expected)
    np.testing.assert_allclose(
        trace.x,
        sample_result.sensitivity.counterparty_spreads_bps,
    )
    np.testing.assert_allclose(
        trace.y,
        sample_result.sensitivity.own_spreads_bps,
    )


@pytest.mark.parametrize(
    "metric",
    [
        "first_to_default_net_xva",
        "naive_net_xva",
        "cva_correction",
        "dva_correction",
        "net_xva_correction",
    ],
)
def test_signed_sensitivity_uses_symmetric_scale(charts, sample_result, metric):
    fig = charts["sensitivity_chart"](sample_result, metric)
    trace = fig.data[0]

    assert trace.zmin == pytest.approx(-trace.zmax)
    assert trace.zmid == pytest.approx(0.0)


@pytest.mark.parametrize(
    "metric",
    [
        "first_to_default_cva",
        "first_to_default_dva",
    ],
)
def test_nonnegative_sensitivity_starts_at_zero(charts, sample_result, metric):
    fig = charts["sensitivity_chart"](sample_result, metric)

    assert fig.data[0].zmin == pytest.approx(0.0)


def test_sensitivity_axis_labels(charts, sample_result):
    fig = charts["sensitivity_chart"](
        sample_result,
        "first_to_default_net_xva",
    )

    assert fig.layout.xaxis.title.text == "Counterparty CDS spread (bps)"
    assert fig.layout.yaxis.title.text == "Own CDS spread (bps)"


def test_dashboard_imports_are_optional():
    """App must not be imported by core engine modules."""
    source = APP_PATH.read_text(encoding="utf-8")

    assert "import streamlit as st" in source
    assert "import plotly.graph_objects as go" in source

    core_root = APP_PATH.parents[1] / "src" / "pyccr"

    for path in core_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue

            assert not any(
                name in {"streamlit", "plotly"}
                or name.startswith(("streamlit.", "plotly."))
                for name in names
            ), f"UI dependency imported by core module: {path}"
