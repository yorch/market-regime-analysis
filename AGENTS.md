# Repository Guidelines

## Project Overview

A market regime analysis system using Hidden Markov Models to classify market states (Bull/Bear Trending, Mean Reverting, High/Low Volatility, Breakout) and generate trading signals. Organized as a multi-package uv workspace: `mra_lib` (core library), `mra_cli` (CLI via Click), and `mra_web` (FastAPI web API).

## Project Structure

- `packages/mra_lib/src/mra_lib/`: Core library (HMM analyzer, risk, providers, backtesting).
  - `data_providers/`: Pluggable data sources (`yfinance`, `alphavantage`, `polygon`).
  - Key modules: `analyzer.py`, `indicators/hmm_detector.py`, `indicators/true_hmm_detector.py`, `portfolio/portfolio.py`, `risk/risk_calculator.py`, `backtesting/`.
- `packages/mra_cli/src/mra_cli/main.py`: Click-based CLI commands (entry point: `mra`).
- `packages/mra_web/src/mra_web/`: FastAPI app (entry point: `mra-api`).
- Tests: `packages/<pkg>/tests/`.
- Docs: `docs/`, planning docs archived in `docs/archive/`.

## Build, Test, and Development Commands

- Install deps: `uv sync`
- Run CLI help: `uv run mra --help`
- Example: `uv run mra current-analysis --symbol SPY --provider yfinance`
- Run API (dev): `uv run mra-api --dev` (docs at `http://localhost:8000/docs`)
- Tests: `uv run pytest` or `just test`
- Lint: `uv run ruff check packages/ examples/`
- Format: `uv run ruff format packages/ examples/`
- Full QA gate: `just qa` (fmt + lint + types)

## Coding Style & Naming Conventions

- Python 3.13+, 4-space indentation, type hints required for public APIs.
- Ruff configured in `pyproject.toml`: line length 100, double quotes, import sorting.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep docstrings concise; prefer explicit errors and defensive checks.

## Testing Guidelines

- Framework: `pytest`. Markers: `integration`, `slow`.
- Location: `packages/<pkg>/tests/`.
- Conventions: name tests `test_<unit>_<behavior>()`; use fixtures and deterministic inputs.
- Run locally with `uv run pytest`; ensure tests pass before PRs.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits: `feat:`, `fix:`, `docs:`, etc.
- Write imperative, scoped messages: `feat(cli): add multi-symbol analysis`.
- PRs must include: clear description, test evidence, and impact notes.
- Pass CI (lint + tests) and keep diffs focused; update docs for user-facing changes.

## Security & Configuration Tips

- Use environment variables for secrets: `ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, `JWT_SECRET`.
- Avoid `--api-key` in shell history; use `export VAR=...` or `.env`.
- CORS/rate limits/JWT configured via `config.py`/env; review before exposing the API.
