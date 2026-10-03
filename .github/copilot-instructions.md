# Market Regime Analysis — AI Coding Agent Instructions

The canonical guide for AI coding assistants is [`AGENTS.md`](../AGENTS.md) at the repository
root (`CLAUDE.md` is a symlink to it). Read it before making changes: it covers the workspace
layout, commands, conventions, CI, and security configuration.

Quick reference:

- uv workspace with three packages: `mra_lib` (core library, no CLI/web framework deps), `mra_cli` (Click CLI,
  `uv run mra`), `mra_web` (FastAPI API, `uv run mra-api`).
- Install: `uv sync`. Offline demo data: `--provider mock`.
- Before committing: `just qa` (ruff format check + ruff lint + mypy, which is blocking in CI)
  and `just test-unit`.
- Docstrings use Google style; type hints throughout; Conventional Commits.
- User docs: [`README.md`](../README.md); API reference: [`docs/api.md`](../docs/api.md);
  project state: [`docs/status.md`](../docs/status.md).
