# ── Market Regime Analysis ───────────────────────────────────────────────────
# Task runner for the multi-package workspace.
# Install just: https://github.com/casey/just

set dotenv-load

default:
    @just --list --unsorted

# ── Run ──────────────────────────────────────────────────────────────────────

# Run the CLI with arguments (e.g. just run current-analysis --symbol SPY)
run *ARGS:
    uv run mra {{ARGS}}

# Run the optimization script
optimize *ARGS:
    uv run mra-optimize {{ARGS}}

# Start the API server
api *ARGS:
    uv run mra-api {{ARGS}}

# Start the API in development mode
api-dev:
    uv run mra-api --dev

# ── Test ─────────────────────────────────────────────────────────────────────

# Run all tests across all packages
test *ARGS:
    uv run pytest {{ARGS}}

# Run only unit tests (exclude integration and slow markers)
test-unit:
    uv run pytest -m "not integration and not slow"

# Run tests for a specific package
test-lib *ARGS:
    uv run pytest packages/mra_lib/tests/ {{ARGS}}

test-cli *ARGS:
    uv run pytest packages/mra_cli/tests/ {{ARGS}}

test-web *ARGS:
    uv run pytest packages/mra_web/tests/ {{ARGS}}

# Run tests with verbose output
test-v:
    uv run pytest -v

# Run tests with coverage (fails if below 65%)
test-cov:
    uv run pytest --cov=mra_lib --cov=mra_cli --cov=mra_web --cov-report=term-missing --cov-fail-under=65

# ── Quality ──────────────────────────────────────────────────────────────────

# Format code across all packages
fmt:
    uv run ruff format packages/ examples/

# Check formatting without modifying files
fmt-check:
    uv run ruff format --check packages/ examples/

# Lint code across all packages
lint:
    uv run ruff check packages/ examples/

# Fix auto-fixable lint issues
fix:
    uv run ruff check --fix packages/ examples/

# Run type checking
types:
    uv run mypy packages/

# Run all quality checks (gate before commit)
qa: fmt lint types

# ── Install ──────────────────────────────────────────────────────────────────

# Install all dependencies (including dev)
install:
    uv sync

# Install pre-commit hooks
pre-commit-install:
    uv run pre-commit install

# ── Docker ───────────────────────────────────────────────────────────────────

# Build the Docker image
docker-build:
    docker compose build

# Start detached
docker-up:
    docker compose up -d

# Stop
docker-down:
    docker compose down

# Restart
docker-restart:
    docker compose restart

# Tail logs
docker-logs:
    docker compose logs -f

# One-shot run (smoke test)
docker-once:
    docker compose run --rm scanner --once

# Debug shell
docker-shell:
    docker compose run --rm --entrypoint /bin/bash scanner

# Full reset (removes containers + volumes)
docker-clean:
    docker compose down -v --rmi local

# Start with PostgreSQL
docker-pg-up:
    docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d

# Stop with PostgreSQL
docker-pg-down:
    docker compose -f docker-compose.yml -f docker-compose.postgres.yml down

# Tail logs (both services)
docker-pg-logs:
    docker compose -f docker-compose.yml -f docker-compose.postgres.yml logs -f

# Full reset with PostgreSQL (removes volumes + images)
docker-pg-clean:
    docker compose -f docker-compose.yml -f docker-compose.postgres.yml down -v --rmi local

# ── Utilities ────────────────────────────────────────────────────────────────

# Remove all cache files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Show available data providers
providers:
    uv run mra list-providers

# Show CLI help
help:
    uv run mra --help
