"""Regression tests for the Bachelier pricing REST endpoint."""

import pytest
from fastapi.testclient import TestClient

from pyccr.api.main import app
from pyccr.pricing.bachelier_swaption import bachelier_swaption_price

ENDPOINT = "/api/v1/swaptions/bachelier/price"

PAYLOAD = {
    "forward": 0.035,
    "strike": 0.035,
    "expiry": 2.0,
    "normal_volatility": 0.0075,
    "annuity": 4.5,
    "option_type": "payer",
    "notional": 1_000_000,
}


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_api_price_matches_quantitative_engine(option_type: str) -> None:
    """REST pricing agrees with the existing Bachelier implementation."""
    payload = {**PAYLOAD, "option_type": option_type}

    response = TestClient(app).post(ENDPOINT, json=payload)

    assert response.status_code == 200

    result = response.json()
    expected = bachelier_swaption_price(**payload)

    assert result["price"] == pytest.approx(expected, rel=1e-12)
    assert result["model"] == "Bachelier"
    assert result["currency"] == "unspecified"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("normal_volatility", -0.01),
        ("expiry", -1.0),
        ("annuity", 0.0),
        ("option_type", "call"),
        ("notional", -100.0),
    ],
)
def test_api_rejects_invalid_inputs(field: str, invalid_value: object) -> None:
    """Pydantic rejects invalid pricing requests."""
    payload = {**PAYLOAD, field: invalid_value}

    response = TestClient(app).post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_api_rejects_unknown_fields() -> None:
    """Unexpected JSON fields must not be silently ignored."""
    payload = {**PAYLOAD, "unexpected_parameter": 123}

    response = TestClient(app).post(ENDPOINT, json=payload)

    assert response.status_code == 422


def test_api_rejects_nonfinite_values() -> None:
    """Pydantic rejects a non-finite value in the request body."""
    import json

    payload = {**PAYLOAD, "forward": "NaN"}

    response = TestClient(app).post(
        ENDPOINT,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_bachelier_endpoint_in_openapi_schema() -> None:
    """The pricing endpoint appears in generated API documentation."""
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert ENDPOINT in paths
    assert "post" in paths[ENDPOINT]
