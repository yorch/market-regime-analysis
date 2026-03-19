# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv and build tools needed for sdist-only packages (e.g. peewee)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

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
    touch packages/mra_web/src/mra_web/__init__.py && \
    touch packages/mra_lib/README.md

# Install dependencies into .venv (cached layer — only rebuilds when manifests change)
RUN uv sync --frozen --no-dev

# Copy actual source code
COPY packages/ packages/

# Reinstall workspace packages as non-editable wheels baked into the venv
# (only .venv is copied to runtime — no source tree)
RUN uv sync --frozen --no-dev --no-editable

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 scanner

# Copy only the virtual environment from builder (no uv, no source, no build artifacts)
COPY --from=builder /app/.venv /app/.venv

# Put venv on PATH so installed entry points are directly available
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MRA_NO_DASHBOARD=1

USER scanner

ENTRYPOINT ["mra-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
