# syntax=docker/dockerfile:1

# PyCCR API and dashboard images.
# Build with --target api or --target dashboard.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/pyccr

# Install project dependencies separately from source for layer reuse.
COPY pyproject.toml ./
COPY src ./src
COPY app ./app

RUN python -m pip install --upgrade pip

# API target.
FROM base AS api

RUN python -m pip install ".[api]"

RUN useradd --create-home --uid 10001 pyccr
USER pyccr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "pyccr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit dashboard target.
FROM base AS dashboard

RUN python -m pip install ".[dashboard]"

RUN useradd --create-home --uid 10001 pyccr
USER pyccr

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
