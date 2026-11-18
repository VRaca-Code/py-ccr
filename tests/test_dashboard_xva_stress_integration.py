"""Dashboard integration tests for additive credit/funding XVA stress."""

import ast
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.xva_stress_panel import calculate_dashboard_xva_stress
from pyccr.dashboard.scenario import DashboardScenarioConfig, run_dashboard_scenario

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "app/streamlit_app.py"


def test_dashboard_structure_preserved():
    source = DASHBOARD.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    assert "render_xva_stress_panel(config, result)" in source
    assert "Funding & Credit Stress" in (ROOT / "app/xva_stress_panel.py").read_text(
        encoding="utf-8"
    )
    assert source.count("st.tabs(") == 1
    assert '"Bilateral XVA"' in source


@pytest.fixture(scope="module")
def scenario():
    config = DashboardScenarioConfig(n_paths=200, n_steps=12)
    return config, run_dashboard_scenario(config)


def test_zero_shocks_preserve_additive_base(scenario):
    config, result = scenario
    comparison = calculate_dashboard_xva_stress(
        config,
        result,
        borrowing_spread_bps=100.0,
        lending_spread_bps=50.0,
        counterparty_hazard_shock_bps=0.0,
        own_hazard_shock_bps=0.0,
        borrowing_shock_bps=0.0,
        lending_shock_bps=0.0,
    )
    assert comparison.delta_net_xva == pytest.approx(0.0)
    assert comparison.base.cva == pytest.approx(result.standalone_xva.cva)
    assert comparison.base.dva == pytest.approx(result.standalone_xva.dva)
    np.testing.assert_allclose(
        comparison.base.times,
        result.times,
    )


def test_borrowing_shock_increases_fca(scenario):
    config, result = scenario
    comparison = calculate_dashboard_xva_stress(
        config,
        result,
        borrowing_spread_bps=100.0,
        lending_spread_bps=50.0,
        counterparty_hazard_shock_bps=0.0,
        own_hazard_shock_bps=0.0,
        borrowing_shock_bps=100.0,
        lending_shock_bps=0.0,
    )
    assert comparison.delta_fca >= 0.0
    assert comparison.delta_cva == 0.0
    assert comparison.delta_dva == 0.0
    assert comparison.delta_net_xva == pytest.approx(-comparison.delta_fca)


def test_dashboard_renders_stress_panel():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 7

    labels = {metric.label for metric in app.metric}
    assert "Base additive net XVA" in labels
    assert "Stressed additive net XVA" in labels
    assert "Change in additive net XVA" in labels


def test_dashboard_stress_controls_are_interactive():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=90)
    app.run(timeout=90)

    assert not app.exception

    control = next(
        item for item in app.number_input if item.key == "xva_borrowing_shock"
    )
    control.set_value(100.0)
    app.run(timeout=90)

    assert not app.exception
    assert len(app.tabs) == 7
