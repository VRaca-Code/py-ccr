"""Regression tests for the PyCCR REST API."""

from fastapi.testclient import TestClient

from pyccr.api.main import app


def test_health_endpoint() -> None:
    """Health endpoint returns a successful JSON response."""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "pyccr",
    }


def test_openapi_schema() -> None:
    """FastAPI exposes the documented health endpoint."""
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert schema["info"]["title"] == "PyCCR Risk Analytics API"
    assert schema["info"]["version"] == "0.1.0"
    assert "/health" in schema["paths"]
    assert "get" in schema["paths"]["/health"]
