# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
COPY packages/mra_lib/pyproject.toml packages/mra_lib/pyproject.toml
COPY packages/mra_cli/pyproject.toml packages/mra_cli/pyproject.toml
COPY packages/mra_web/pyproject.toml packages/mra_web/pyproject.toml

# Create minimal package stubs so uv can resolve workspace members
RUN mkdir -p packages/mra_lib/src/mra_lib && \
    mkdir -p packages/mra_cli/src/mra_cli && \
    mkdir -p packages/mra_web/src/mra_web && \
    touch packages/mra_lib/src/mra_lib/__init__.py && \
    touch packages/mra_cli/src/mra_cli/__init__.py && \
    touch packages/mra_web/src/mra_web/__init__.py

# Install dependencies into .venv
RUN uv sync --frozen --no-dev

# Copy actual source code
COPY packages/ packages/
COPY examples/ examples/

# Reinstall workspace packages with actual source
RUN uv sync --frozen --no-dev

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Copy only the virtual environment from builder (no uv, no source needed)
COPY --from=builder /app/.venv /app/.venv

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default configuration via env vars
ENV API_HOST="0.0.0.0" \
    API_PORT="8000" \
    ENVIRONMENT="production" \
    LOG_LEVEL="INFO"

EXPOSE 8000

USER app

# Headless by default — start the API server
CMD ["python", "-m", "uvicorn", "mra_web.app:app", "--host", "0.0.0.0", "--port", "8000"]
