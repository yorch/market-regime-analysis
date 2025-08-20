SHELL := bash
.DEFAULT_GOAL := help

# Defaults (override with e.g. `make cli-current SYMBOL=QQQ PROVIDER=yfinance`)
SYMBOL ?= SPY
PROVIDER ?= yfinance
HOST ?= 0.0.0.0
PORT ?= 8000
FILE ?=

.PHONY: help install install-dev lint format test check cli-help cli-current providers api api-dev pre-commit-install clean

help: ## Show this help
	@echo "Common tasks:" && \
	awk 'BEGIN {FS":.*##"}; /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime dependencies with uv
	uv sync

install-dev: ## Install all dependencies including dev tools
	uv sync --all-groups

lint: ## Lint codebase with Ruff
	uv run ruff check .

format: ## Format codebase with Ruff
	uv run ruff format .

test: ## Run tests (set FILE= to run a specific file)
	uv run pytest $(FILE)

check: ## Run lint and tests
	$(MAKE) lint && $(MAKE) test

cli-help: ## Show CLI help
	uv run main.py --help

cli-current: ## Run current analysis (SYMBOL, PROVIDER configurable)
	uv run main.py current-analysis --symbol $(SYMBOL) --provider $(PROVIDER)

providers: ## List available data providers
	uv run main.py list-providers

api: ## Start API server
	uv run start_api.py --host $(HOST) --port $(PORT)

api-dev: ## Start API server in dev mode (reload, debug)
	uv run start_api.py --dev --host $(HOST) --port $(PORT)

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

clean: ## Remove caches and temporary files
	rm -rf .ruff_cache .pytest_cache **/__pycache__
