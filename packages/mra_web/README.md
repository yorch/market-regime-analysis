# mra-web

FastAPI web API for market regime analysis.

## Entry Point

- `mra-api` — Starts the Uvicorn server

## Usage

```bash
uv run mra-api --dev        # Development mode with auto-reload
uv run mra-api --host 0.0.0.0 --port 8000 --workers 4  # Production
```

API docs available at `http://localhost:8000/docs` (Swagger UI).

See [docs/api.md](../../docs/api.md) for full API reference.

Part of the [market-regime-analysis](../../README.md) workspace.
