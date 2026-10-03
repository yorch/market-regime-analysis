# syntax=docker/dockerfile:1
# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Pinned uv version (bump deliberately)
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

# Build tools needed for sdist-only packages (e.g. peewee)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Byte-compile for faster startup; copy (not hardlink) out of the cache mount;
# always use the image's Python instead of downloading one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
COPY packages/mra_lib/pyproject.toml packages/mra_lib/pyproject.toml
COPY packages/mra_cli/pyproject.toml packages/mra_cli/pyproject.toml
COPY packages/mra_web/pyproject.toml packages/mra_web/pyproject.toml

# Minimal stubs so uv can read the workspace members' metadata
RUN mkdir -p packages/mra_lib/src/mra_lib \
             packages/mra_cli/src/mra_cli \
             packages/mra_web/src/mra_web && \
    touch packages/mra_lib/src/mra_lib/__init__.py \
          packages/mra_cli/src/mra_cli/__init__.py \
          packages/mra_web/src/mra_web/__init__.py \
          packages/mra_lib/README.md

# Third-party dependencies only (cached layer — rebuilds only when manifests change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-workspace

# Copy actual source code
COPY packages/ packages/

# Install workspace packages as non-editable wheels baked into the venv
# (only .venv is copied to runtime — no source tree)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Non-root user. Pre-create the cache dir so a named volume mounted there is
# initialised with uid 1000 ownership instead of root.
RUN useradd -m -u 1000 mra \
    && mkdir -p /home/mra/.cache \
    && chown -R mra:mra /home/mra

# Copy only the virtual environment from builder (no uv, no source, no build artifacts)
COPY --from=builder /app/.venv /app/.venv

# Put venv on PATH so installed entry points are directly available.
# Inside the container the API binds 0.0.0.0; docker-compose.yml publishes it
# on the host's loopback only.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

USER mra

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

ENTRYPOINT ["mra-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
