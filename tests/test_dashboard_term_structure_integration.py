"""Structural integration checks for term-structure dashboard tabs."""

import ast
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


@pytest.fixture(scope="module")
def source():
    return DASHBOARD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source):
    return ast.parse(source)


def function_calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_dashboard_parses(tree):
    assert isinstance(tree, ast.Module)


def test_existing_exposure_chart_preserved(source):
    assert "exposure_chart(" in source


def test_existing_trade_chart_preserved(source):
    assert "trade_chart(" in source


def test_existing_xva_chart_preserved(source):
    assert "xva_chart(" in source


def test_existing_sensitivity_chart_preserved(source):
    assert "sensitivity_chart(" in source


def test_seven_tabs_present(source, tree):
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "st"
        and node.value.func.attr == "tabs"
    ]

    assert len(assignments) == 1

    target = assignments[0].targets[0]
    assert isinstance(target, (ast.Tuple, ast.List))

    names = [element.id for element in target.elts if isinstance(element, ast.Name)]

    assert names == [
        "exposure_tab",
        "portfolio_tab",
        "xva_tab",
        "sensitivity_tab",
        "curve_tab",
        "hull_white_tab",
        "backtesting_tab",
    ]

    assert "Yield curve & forwards" in source
    assert "Hull–White diagnostics" in source


def test_discount_chart_connected(tree):
    assert function_calls(tree, "discount_factor_chart")


def test_forward_chart_connected(tree):
    assert function_calls(tree, "forward_curve_chart")


def test_forward_slope_chart_connected(tree):
    assert function_calls(tree, "forward_slope_chart")


def test_drift_chart_connected(tree):
    assert function_calls(tree, "hull_white_drift_chart")


def test_parameter_comparison_connected(tree):
    assert function_calls(tree, "parameter_comparison_chart")


def test_scenario_parameters_used_for_monitoring(source):
    assert "mean_reversion=config.mean_reversion" in source
    assert "volatility=config.volatility" in source
    assert "zero_rates=(0.03,) * 6" in source


def test_alternative_parameters_are_diagnostic_only(source):
    assert "alternative_mean_reversion" in source
    assert "alternative_volatility" in source
    assert "Alternative parameters are diagnostic only" in source
    assert "not calibration residuals" in source
