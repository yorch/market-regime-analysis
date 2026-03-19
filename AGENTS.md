# Repository Guidelines

## Project Structure & Module Organization

- `packages/mra_lib/src/mra_lib/`: Core library package (HMM analyzer, risk, providers).
  - `data_providers/`: Pluggable data sources (`yfinance`, `alphavantage`, `polygon`).
  - Key modules: `analyzer.py`, `indicators/hmm_detector.py`, `portfolio/portfolio.py`, `risk/risk_calculator.py`.
- `packages/mra_cli/src/mra_cli/main.py`: Click-based CLI commands (entry point: `mra`).
- `packages/mra_web/src/mra_web/`: FastAPI app (entry point: `mra-api`).
- `tests`: Test files live in `packages/mra_lib/tests/`, `packages/mra_cli/tests/`, `packages/mra_web/tests/`.
- `docs/`, `examples/`, and planning docs: `README.md`, `ARCHITECTURE.md`, `API_README.md`.

## Build, Test, and Development Commands

- Install deps: `uv sync` (runtime) or `uv sync --all-groups` (dev tooling)
- Run CLI help: `uv run mra --help`
- Common CLI example: `uv run mra current-analysis --symbol SPY --provider yfinance`
- Run API (dev): `uv run mra-api --dev` (docs at `http://localhost:8000/docs`)
- Tests (all): `uv run pytest`
- Pytest (all): `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Pre-commit (install): `uv run pre-commit install`

## Coding Style & Naming Conventions

- Python 3.13+, 4-space indentation, type hints required for public APIs.
- Ruff configured in `pyproject.toml`:
  - Line length 100, double quotes, import sorting enabled.
  - Selected rules: `E,F,I,UP,B,C4,T20,SIM,PL,RUF`; targeted per-file ignores for complex modules.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep docstrings concise; prefer explicit errors and defensive checks.

## Testing Guidelines

- Framework: `pytest`.
- Location: `packages/mra_lib/tests/` (e.g., `test_mock.py`, `test_system.py`).
- Conventions: name tests `test_<unit>_<behavior>()`; use fixtures and deterministic inputs.
- Aim to cover new branches/edge cases; add provider mocks where needed (`data_providers/mock_provider.py`).
- Run locally with `uv run pytest`; ensure tests pass before PRs.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits seen in history: `feat:`, `fix:`, `docs:`, etc.
- Write imperative, scoped messages: `feat(cli): add multi-symbol analysis`.
- PRs must include: clear description, linked issue, test evidence (logs or screenshots), and impact notes for CLI/API.
- Pass CI (lint + tests) and keep diffs focused; update docs for user-facing changes.

## Security & Configuration Tips

- Prefer environment variables for secrets: `ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, `JWT_SECRET`.
- Avoid `--api-key` in shell history; use `export VAR=...` or `.env` injection.
- CORS/rate limits/JWT are configured via `config.py`/env; review before exposing the API.
