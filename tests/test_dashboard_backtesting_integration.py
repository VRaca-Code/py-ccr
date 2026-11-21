"""Integration checks for the synthetic backtesting dashboard tab."""

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_backtesting_tab_and_controls_present():
    source = DASHBOARD.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "Backtesting & Model Validation" in source
    assert "SYNTHETIC DEMONSTRATION ONLY" in source
    assert "bt_windows" in source
    assert "bt_quantile" in source
    assert "bt_tail" in source
    assert "bt_alternative" in source

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "test_coverage",
        "breach_timeline_chart",
        "rolling_coverage_chart",
        "binomial_null_chart",
    }.issubset(called)


def test_backtesting_tab_runs_in_streamlit():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 9
    assert app.tabs[8].label == "Backtesting & Model Validation"

    chart_elements = app.get("plotly_chart")
    assert len(chart_elements) >= 12


def test_backtesting_controls_are_interactive():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)
    assert not app.exception

    factor = next(item for item in app.selectbox if item.key == "bt_factor")
    factor.set_value("EUR/USD FX")
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 9


def test_bootstrap_dashboard_controls_and_chart_present():
    source = DASHBOARD.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for key in (
        "bt_block_length",
        "bt_resamples",
        "bt_ci_level",
    ):
        assert key in source

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "bootstrap_coverage_interval" in called
    assert "bootstrap_coverage_chart" in called
    assert "not null-calibrated bootstrap hypothesis tests" in source


def test_bootstrap_controls_are_interactive():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 9
    assert len(app.get("plotly_chart")) >= 13

    block_control = next(
        item for item in app.selectbox if item.key == "bt_block_length"
    )
    block_control.set_value(5)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.get("plotly_chart")) >= 13


def test_dependence_dashboard_controls_and_chart_present():
    source = DASHBOARD.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "bt_max_lag" in source
    assert "Consecutive breach pairs" in source
    assert "Breach clustering and serial dependence" in source

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "breach_dependence_diagnostics" in called
    assert "breach_autocorrelation_chart" in called


def test_dependence_controls_are_interactive():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 9
    assert len(app.get("plotly_chart")) >= 14

    lag_control = next(item for item in app.selectbox if item.key == "bt_max_lag")
    lag_control.set_value(5)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.get("plotly_chart")) >= 14


def test_christoffersen_dashboard_integration_present():
    source = DASHBOARD.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "Christoffersen conditional-coverage tests" in source
    assert "LRuc p-value (asymptotic)" in source
    assert "LRind p-value (asymptotic)" in source
    assert "LRcc p-value (asymptotic)" in source
    assert "sparse_transitions" in source
    assert "independence_identifiable" in source

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "christoffersen_coverage_test" in called


def test_christoffersen_dashboard_runs_with_default_controls():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 9
    assert len(app.get("plotly_chart")) >= 14

    metric_labels = {item.label for item in app.metric}
    assert "LRuc p-value (asymptotic)" in metric_labels
    assert "LRind p-value (asymptotic)" in metric_labels
    assert "LRcc p-value (asymptotic)" in metric_labels


def test_christoffersen_dashboard_handles_rare_tail():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)
    assert not app.exception

    quantile = next(item for item in app.selectbox if item.key == "bt_quantile")
    quantile.set_value(0.995)
    app.run(timeout=90)

    assert not app.exception
    metric_labels = {item.label for item in app.metric}
    assert "LRuc p-value (asymptotic)" in metric_labels
    assert "LRind p-value (asymptotic)" in metric_labels
    assert "LRcc p-value (asymptotic)" in metric_labels
