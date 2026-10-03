# mra-web

FastAPI web API for market regime analysis.

## Entry Points

- `mra-api` — Validates the configuration and starts the Uvicorn server
  (`mra start-api` from `mra-cli` runs the same server)
- `mra-token` — Mints a JWT signed with `JWT_SECRET` (there is no login endpoint)

## Usage

```bash
# Development: ENVIRONMENT=development, auto-reload, unauthenticated requests allowed
uv run mra-api --dev

# Production (default): JWT_SECRET of 32+ characters required
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uv run mra-api --host 0.0.0.0 --port 8000 --workers 4
uv run mra-token --sub alice            # Authorization: Bearer <token>
```

Clients authenticate with `Authorization: Bearer <JWT>` or `X-API-Key` (keys from `API_KEYS`).
API docs are served at `/docs` in development or with `ENABLE_DOCS=true`.

See [docs/api.md](../../docs/api.md) for the full API reference.

Part of the [market-regime-analysis](../../README.md) workspace.
