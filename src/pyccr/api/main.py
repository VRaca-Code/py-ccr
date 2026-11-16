"""FastAPI application for PyCCR."""

from fastapi import FastAPI

from pyccr.api.pricing import router as pricing_router

app = FastAPI(
    title="PyCCR Risk Analytics API",
    description="Pricing, calibration, simulation and risk monitoring.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return application health status."""
    return {"status": "ok", "service": "pyccr"}


app.include_router(pricing_router)
