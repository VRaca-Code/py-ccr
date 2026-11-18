# PyCCR

PyCCR is a Python research project for multi-asset counterparty credit risk, exposure simulation, XVA, model validation, and interactive risk analytics.

**Status:** Research-oriented MVP. Not a certified regulatory capital engine or production risk-management system.

## Requirements

- Python 3.12 or newer
- Git
- Docker and Docker Compose for container deployment

## Installation

Clone https://github.com/VRaca-Code/py-ccr.git and run the following commands from the repository root:

    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,dashboard,api]"

## Quality checks

    ruff check .
    ruff format --check .
    python -m mypy --no-incremental --cache-dir=/dev/null
    python -m pytest -q

Mypy is configured for the XVA and stress modules. GitHub Actions targets Python 3.12 and 3.13.

## API

    python -m pip install -e ".[api]"
    uvicorn pyccr.api.main:app --host 127.0.0.1 --port 8000

Health: http://127.0.0.1:8000/health

Documentation: http://127.0.0.1:8000/docs

## Streamlit dashboard

    python -m pip install -e ".[dashboard]"
    streamlit run app/streamlit_app.py

The dashboard includes exposure, XVA, term-structure, backtesting, and stress-analysis views.

## Docker Compose

Build and start the API and dashboard:

    docker compose up --build -d

Services are published on localhost:

- API: http://127.0.0.1:8000
- Dashboard: http://127.0.0.1:8501

Inspect or stop the services:

    docker compose ps
    docker compose logs --tail=100
    docker compose down

The API health endpoint has returned HTTP 200 in a local test, dashboard support imports have passed, and the Python wheel has been built and inspected. Docker images and Compose services have not yet been built or started in the development environment.

## Capabilities

- Counterparty exposure simulation and exposure profiles
- FX and interest-rate simulation components
- CVA, DVA, and simplified funding valuation adjustments
- Additive standalone XVA aggregation
- Deterministic and simulated stress scenarios
- Monte Carlo convergence diagnostics
- Coverage, dependence, and backtesting diagnostics
- FastAPI service and Streamlit dashboard

## Important model limitations

### Statistical interpretation

- Synthetic simulated observations are not historical data.
- Risk-neutral paths are not real-world market forecasts.
- Exact binomial coverage tests assume IID observations under the stated null hypothesis.
- Bootstrap confidence intervals are not automatically null-calibrated p-values.
- Sparse exceedances can invalidate asymptotic chi-square approximations.
- Antithetic pairs and common-random-number designs require dependence-aware uncertainty interpretation.
- Extreme-tail empirical quantiles cannot extrapolate beyond the observed sample maximum.

### Stress and valuation interpretation

- Correlation-only shocks may change finite-sample Monte Carlo estimates without changing true marginal FX-forward exposure or standalone XVA in the current constant-rate setup.
- Joint Hull-White/FX short-rate paths do not currently drive FX drift, discounting, or FX-forward repricing.
- Funding adjustments use simplified deterministic spreads.
- CVA, DVA, and FVA are aggregated additively, not through a first-to-default-coupled bilateral valuation.
- Full collateral, wrong-way risk, and funding/default feedback are not modeled.

## Repository layout

- `src/pyccr/`: Python library and API
- `app/`: Streamlit dashboard
- `tests/`: automated tests
- `.github/`: CI workflow
- `Dockerfile`: API and dashboard image targets
- `compose.yaml`: local deployment
- `pyproject.toml`: package and tool configuration

A successful wheel build or local health check does not establish successful container deployment, remote CI execution, or production readiness.
