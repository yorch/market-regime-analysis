# Market Regime Analysis API

REST + WebSocket API for market regime analysis using Hidden Markov Models
(package `mra_web`, FastAPI).

## Quick Start

### Installation

```bash
uv sync

# Optional: data provider credentials (Yahoo Finance and mock need none)
export ALPHA_VANTAGE_API_KEY=your_key_here
export POLYGON_API_KEY=your_key_here
export APCA_API_KEY_ID=your_key_id APCA_API_SECRET_KEY=your_secret
export TIINGO_API_KEY=your_key_here
```

### Start the Server

```bash
# Development: ENVIRONMENT=development, auto-reload, DEBUG logging, docs at /docs,
# unauthenticated requests allowed. No JWT_SECRET needed.
uv run mra-api --dev            # or: uv run mra start-api --dev

# Production (the default environment): a JWT secret of 32+ characters is required
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uv run mra-api --host 0.0.0.0 --port 8000 --workers 4
```

`mra-api` options: `--host` (`API_HOST`, default `127.0.0.1`), `--port` (`API_PORT`, default
`8000`), `--workers` (`API_WORKERS`, default 1), `--reload` (`API_RELOAD`), `--log-level`
(`LOG_LEVEL`), `--dev`. `mra start-api [--host] [--port] [--dev]` runs the same server
(it honours `API_HOST`, `API_PORT` and `LOG_LEVEL`, but always uses one worker).

Outside `ENVIRONMENT=development` the server refuses to start (exit code 2) when `JWT_SECRET`
is unset, empty, a well-known placeholder, or shorter than 32 characters, or when `API_KEYS`
contains a key shorter than 16 characters. In development a random per-process secret is
generated instead (with a warning); tokens signed with it do not survive a restart, and it
cannot be combined with `--workers` > 1.

### Routes

| Method | Path | Auth | Response |
|--------|------|------|----------|
| GET | `/` | public | API info |
| GET | `/health`, `/api/v1/health` | public | `{"status": "healthy", "timestamp", "version"}` |
| GET | `/ready` | public | readiness checks |
| GET | `/metrics` | required | API metrics + WebSocket connection stats |
| GET | `/api/v1/metrics` | required | same as `/metrics` |
| POST | `/api/v1/analysis/detailed` | required | `AnalysisResponse` (one timeframe) |
| POST | `/api/v1/analysis/current` | required | `MultiAnalysisResponse` (1D, 1H, 15m) |
| POST | `/api/v1/analysis/confirmation` | required | `ConfirmationResponse` (multi-timeframe confirmation) |
| POST | `/api/v1/analysis/multi-symbol` | required | `PortfolioAnalysisResponse` |
| POST | `/api/v1/position-sizing` | required | `PositionSizingResponse` |
| GET | `/api/v1/providers` | required | `ProvidersResponse` |
| GET | `/api/v1/regimes/{symbol}/history` | required | `RegimeHistoryResponse` (recorded regimes) |
| POST | `/api/v1/charts/generate` | required | `image/png` |
| POST | `/api/v1/export/csv` | required | `text/csv` |
| GET | `/ws/monitoring/status` | required | active WebSocket connections |
| WS | `/ws/monitoring/{symbol}` | required | regime update stream |
| GET | `/docs`, `/redoc`, `/openapi.json` | public | only in development or with `ENABLE_DOCS=true` |
| GET | `/debug/config` | public | development only; configuration without secrets |

"Required" means a JWT or API key (see [Authentication](#authentication)); in development,
requests without credentials are accepted as `dev_user`.

The `curl` examples below use `$AUTH`, e.g. `AUTH="Authorization: Bearer $TOKEN"` or
`AUTH="X-API-Key: $KEY"`. Against a development server you can drop the `-H "$AUTH"` part.

## Authentication

There is no login or token-issuing endpoint. Clients authenticate with one of:

1. **JWT Bearer token** — `Authorization: Bearer <token>`. Tokens are HS256-signed with
   `JWT_SECRET` and must carry `sub` and `exp` claims; other algorithms (including `none`)
   are rejected. Mint them out of band, where `JWT_SECRET` is available:

   ```bash
   # Lifetime defaults to JWT_EXPIRATION_HOURS (24)
   TOKEN=$(JWT_SECRET=... uv run mra-token --sub alice --hours 12)
   ```

   `mra-token` exits with code 2 if `JWT_SECRET` is missing or weak.

2. **API key** — `X-API-Key: <key>`, one of the comma-separated keys in `API_KEYS` (16+
   characters each). Keys are compared in constant time. Query-string keys are not accepted.

   ```bash
   export API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
   ```

If both headers are sent, only `X-API-Key` is checked. Missing or invalid credentials return
`401` with `WWW-Authenticate: Bearer`. In development, requests **without** credentials are
accepted as `dev_user`, but credentials that are sent are still validated.

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/providers
curl -H "X-API-Key: $KEY" http://localhost:8000/api/v1/providers
```

The `api_key` field in request bodies (and the WebSocket `api_key` query parameter) is the
**data provider** key, not an API credential. If it is omitted, the server uses its own
provider environment variables.

## Endpoints

All request bodies are JSON. Common fields: `provider` (default **`alphavantage`**; one of
`yfinance`, `alphavantage`, `polygon`, `alpaca`, `tiingo`, `mock`) and `api_key` (provider
key, optional). For `alpaca`, pass `api_key` as `"KEY_ID:SECRET_KEY"` or set both `APCA_*`
variables on the server.

Symbols are upper-cased and must match `^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$` (e.g. `BRK.B`,
`^GSPC`, `ES=F`). Timeframes are `1D`, `1H`, `15m`.

### POST `/api/v1/analysis/detailed`

HMM analysis for one timeframe; only that timeframe is loaded.

```bash
curl -X POST http://localhost:8000/api/v1/analysis/detailed \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"}'
```

Response (`AnalysisResponse`; values from the `mock` provider, rounded):

```json
{
  "symbol": "SPY",
  "timeframe": "1D",
  "current_regime": "High Volatility",
  "regime_confidence": 1.0,
  "regime_persistence": 0.45,
  "transition_probability": 0.345,
  "hmm_state": 2,
  "risk_level": "High",
  "position_sizing_multiplier": 0.154,
  "recommended_strategy": "Volatility Trading",
  "analysis_timestamp": "2026-10-03T17:21:27.281848Z",
  "metrics": {
    "arbitrage_opportunities": [],
    "statistical_signals": ["MACD: Bearish signal"],
    "key_levels": {
      "resistance": 264.79, "support": 239.24, "sma_50": 250.24, "sma_200": 237.06,
      "bb_upper": 253.70, "bb_lower": 238.07, "atr_resistance": 247.15, "atr_support": 238.71
    },
    "hmm_state": 2,
    "transition_probability": 0.345
  }
}
```

`metrics` always has the keys `arbitrage_opportunities` (list of strings),
`statistical_signals` (list of strings), `key_levels` (name → price), `hmm_state` and
`transition_probability`.

### POST `/api/v1/analysis/current`

Analysis for all timeframes (1D, 1H, 15m). Each timeframe is loaded and analyzed
independently; failed timeframes are omitted from `analyses`. If every timeframe fails, the
error of the last failure is returned (see [Errors](#errors)).

```bash
curl -X POST http://localhost:8000/api/v1/analysis/current \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "provider": "yfinance"}'
```

Response: `{"symbol": "SPY", "analyses": [AnalysisResponse, ...], "analysis_timestamp": ...}`.

### POST `/api/v1/analysis/confirmation`

Multi-timeframe confirmation: do the timeframes agree on a direction? Same request shape as
`/analysis/current`, plus an optional `timeframes` list (at least two distinct values from
`1D`, `1H`, `15m`; duplicates are dropped and the list is ordered coarsest first; default:
all three). Each timeframe is loaded and analyzed independently; a timeframe that fails is
left out of `analyses` and counts as unavailable. The signal always uses the full default
weights (as the CLI does), so timeframes left out of `timeframes` also count as
unavailable: a subset saves provider calls but cannot loosen the rules (e.g. `["1H",
"15m"]` can never be confirmed). If every timeframe fails, the error of the last failure
is returned (see [Errors](#errors)).

```bash
curl -X POST http://localhost:8000/api/v1/analysis/confirmation \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "provider": "yfinance", "timeframes": ["1D", "1H", "15m"]}'
```

Response (`analyses` holds one `AnalysisResponse` per timeframe that succeeded):

```json
{
  "symbol": "SPY",
  "timeframes": ["1D", "1H", "15m"],
  "analyses": [],
  "confirmation": {
    "direction": "bullish",
    "agreement": 0.8,
    "confirmed": true,
    "primary_timeframe": "1D",
    "aligned_timeframes": ["1D", "1H"],
    "conflicting_timeframes": ["15m"],
    "unavailable_timeframes": [],
    "risk_timeframes": [],
    "confidence": 1.0,
    "threshold": 0.6,
    "reason": "confirmed"
  },
  "analysis_timestamp": "2026-10-04T12:00:00Z"
}
```

How the signal is computed (`mra_lib.signals.confirmation.confirm_timeframes`):

- **Bias** per regime: Bull Trending and Breakout (the detector only labels up-trending
  volatility expansion a breakout) are `bullish`; Bear Trending is `bearish`; Mean
  Reverting, Low Volatility and High Volatility are `neutral`. High Volatility is also
  listed in `risk_timeframes`. `Unknown` (the detector's "no rule matched" fallback, which
  can come with high confidence) gives no directional read and counts as unavailable, like
  a failed timeframe.
- **`direction`** is the bias of the primary timeframe, the highest available one
  (1D > 1H > 15m).
- **`agreement`** = sum of `weight x confidence` over the timeframes aligned with
  `direction`, divided by the total weight (1D 0.5 + 1H 0.3 + 15m 0.2). Unavailable
  timeframes keep their weight in the denominator, so they lower agreement. With these
  defaults 1D alone never reaches the threshold, nor do 1H + 15m without 1D. When
  `direction` is `neutral`, `agreement` measures neutral consensus; use `confirmed` as the
  signal.
- **`confirmed`** requires at least two available timeframes, a directional primary, at
  least one lower timeframe aligned with it, and `agreement >= threshold` (0.6).
- **`reason`**: `confirmed`, `no_timeframes`, `insufficient_timeframes`,
  `primary_neutral`, `no_alignment` (no lower timeframe agrees) or `below_threshold`.
- **`confidence`**: weight-averaged regime confidence of the aligned timeframes.

### POST `/api/v1/analysis/multi-symbol`

Portfolio analysis across up to 20 symbols (blank entries and duplicates are dropped). Only
the requested timeframe is loaded and each symbol is analyzed once. Symbols that fail are
left out of `analyses`.

```bash
curl -X POST http://localhost:8000/api/v1/analysis/multi-symbol \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"symbols": ["SPY", "QQQ", "IWM"], "timeframe": "1D", "provider": "yfinance"}'
```

Response fields: `symbols`, `timeframe`, `analyses` (list of `AnalysisResponse`),
`correlations` (symbol → symbol → correlation of **returns**, `null` where undefined), and
`portfolio_metrics` with `total_symbols`, `analyzed_symbols`, `dominant_regime`,
`average_confidence`, `regime_consensus`, `risk_level`, `correlation_risk`,
`diversification_benefit` (both `null` with fewer than two analyzed symbols) and
`regime_distribution`.

If **no** symbol could be analyzed, the response reports the most severe per-symbol cause,
classified like the single-symbol routes (see [Errors](#errors)): provider auth failure → `502`,
provider rate limit or outage → `503`, every symbol unknown → `400`. Server-side causes win over
unknown symbols, so the status does not depend on symbol order. Messages are generic; per-symbol
details are only in the server log. If some symbols succeed, the response is `200` with only
those symbols in `analyses`.

### POST `/api/v1/position-sizing`

Regime-adjusted position size: the base size is scaled by the regime multiplier, confidence
and persistence, then adjusted for correlation. No market data is fetched.

```bash
curl -X POST http://localhost:8000/api/v1/position-sizing \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"base_size": 0.02, "regime": "Bull Trending", "confidence": 0.8,
       "persistence": 0.75, "correlation": 0.1}'
```

```json
{
  "base_size": 0.02,
  "regime": "Bull Trending",
  "regime_adjusted_size": 0.020683,
  "correlation_adjusted_size": 0.020683,
  "final_recommendation": 0.020683,
  "calculations": {
    "base_position_size": 0.02,
    "regime_multiplier": 1.3,
    "confidence_factor": 0.86,
    "persistence_factor": 0.925,
    "combined_multiplier": 1.03415,
    "correlation_adjustment": 1.0
  },
  "timestamp": "2026-10-03T17:21:48.120717Z"
}
```

`regime` must be one of `Bull Trending`, `Bear Trending`, `Mean Reverting`,
`High Volatility`, `Low Volatility`, `Breakout`, `Unknown`; `base_size`, `confidence` and
`persistence` are in `[0, 1]`, `correlation` in `[-1, 1]`. Positive sizes are bounded to
1–50% at each step (a zero size stays zero). The `calculations` object reports the factors
actually applied:

- `regime_multiplier` — the regime's own multiplier (0 for `Unknown`).
- `confidence_factor` — `0.3 + 0.7 × confidence`.
- `persistence_factor` — `0.7 + 0.3 × persistence`.
- `combined_multiplier` — `regime_adjusted_size / base_size`, i.e. the product of the three
  factors above after the 1–50% bounding (`null` when `base_size` is 0).
- `correlation_adjustment` — `correlation_adjusted_size / regime_adjusted_size` (`null` when
  `regime_adjusted_size` is 0).

### GET `/api/v1/providers`

Registered data providers with `description`, `requires_api_key`, `rate_limit_per_minute`,
`supported_intervals` and `supported_periods`.

```bash
curl -H "$AUTH" http://localhost:8000/api/v1/providers
```

### POST `/api/v1/charts/generate`

Renders the 5-panel regime chart in memory (headless backend) and returns it as
**`image/png`** with `Content-Disposition: inline; filename="<SYMBOL>_<TF>_<days>_regime_chart.png"`.
`days` defaults to 60 (1–365). Nothing is written on the server.

```bash
curl -X POST http://localhost:8000/api/v1/charts/generate \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "timeframe": "1D", "days": 60, "provider": "yfinance"}' \
  -o spy_chart.png
```

### POST `/api/v1/export/csv`

Returns the analysis (one row per timeframe that could be analyzed) as **`text/csv`**,
built in memory. `filename` (optional; letters, digits, `.`, `_`, `-`, max 100 chars, `.csv`
appended if missing) only sets the download name in `Content-Disposition: attachment`;
nothing is written on the server. The `X-Record-Count` header holds the number of rows. If
every timeframe fails, the error of the last failure is returned (e.g. `400` for an unknown
symbol); if they all succeed but produce no rows, the response is `503`.

```bash
curl -X POST http://localhost:8000/api/v1/export/csv \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "provider": "yfinance", "filename": "spy_analysis.csv"}' \
  -o spy_analysis.csv
```

Columns: `timestamp, symbol, timeframe, close_price, regime, hmm_state, regime_confidence,
regime_persistence, transition_probability, strategy, position_multiplier, risk_level,
arbitrage_count, signal_count, rsi, macd, volatility, atr_percent, price_zscore, autocorr_1`,
then one `level_<name>` column per key level.

### GET `/api/v1/regimes/{symbol}/history`

Recorded regime history for a symbol from the SQLite database at `MRA_DB_PATH`, newest bar
first. This endpoint only reads: records are written by `mra current-analysis --record` (and,
in a follow-up, the scheduled scanner). The analysis endpoints do **not** write to the store.

| Query parameter | Default | Description |
|-----------------|---------|-------------|
| `timeframe` | all | `1D`, `1H` or `15m` |
| `since` / `until` | none | Inclusive bounds on `bar_time`, ISO 8601; values without an offset are UTC |
| `limit` | `100` | Maximum records (`>= 1`); larger values are capped at `1000` (the response's `limit` shows the effective value) |

The symbol follows the same rules as the analysis endpoints (normalized to upper case). An
invalid symbol, timeframe, limit or date, or `since` after `until`, returns `422`
(`VALIDATION_ERROR`). An unknown symbol returns `200` with an empty `records` list. If the
database cannot be opened or read, the response is `503` with the generic message
`Regime history unavailable`.

```bash
curl -H "$AUTH" "http://localhost:8000/api/v1/regimes/SPY/history?timeframe=1D&limit=2"
```

```json
{
  "symbol": "SPY",
  "timeframe": "1D",
  "limit": 2,
  "count": 2,
  "records": [
    {
      "symbol": "SPY",
      "timeframe": "1D",
      "bar_time": "2026-01-06T00:00:00",
      "recorded_at": "2026-01-06T21:05:12.345678Z",
      "regime": "Bull Trending",
      "confidence": 0.91,
      "persistence": 0.8,
      "transition_probability": 0.95,
      "recommended_strategy": "Trend Following",
      "close": 598.12,
      "provider": "yfinance"
    }
  ]
}
```

`bar_time` is the timestamp of the last bar analyzed, without an offset (the provider
contract): UTC for intraday bars, and the exchange session date at `00:00` for daily bars, so
a daily bar's `bar_time` is a date, not a UTC instant. `since`/`until` are compared against
these values after converting any offset to UTC (use plain dates such as `2026-01-02` to filter
daily bars). `recorded_at` is when the record was written (UTC). This route reads the database
in the thread pool without taking an analysis slot; a query slower than 30 s returns `504`. Records are unique per
(`symbol`, `timeframe`, `bar_time`): re-analyzing the same bar replaces the earlier record.

### Metrics

`GET /metrics` and `GET /api/v1/metrics` are the same handler and return:

```json
{
  "uptime_seconds": 22.6,
  "request_counts": {"/analysis/detailed": 2, "/export/csv": 1},
  "error_counts": {"/analysis/detailed": 1},
  "average_response_times": {"/analysis/detailed": 0.91, "/export/csv": 3.59},
  "total_requests": 3,
  "total_errors": 1,
  "websocket_connections": {"total_connections": 0, "active_symbols": [], "connections_by_symbol": {}}
}
```

`websocket_connections` reports `total_connections`, `active_symbols` and
`connections_by_symbol`. Counters are per worker process and reset on
restart.

## Errors

Every error response — authentication, validation, unknown route, rate limit, provider
failure, or unexpected exception — uses one envelope:

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": {
    "errors": [
      {"loc": ["body", "timeframe"], "msg": "Value error, Timeframe must be one of: 1D, 1H, 15m",
       "type": "value_error"}
    ]
  },
  "timestamp": "2026-10-03T17:21:34.618320Z"
}
```

| Status | `error_code` | When |
|--------|--------------|------|
| 400 | `BAD_REQUEST` | Unknown or invalid symbol, or invalid input / no data for the symbol |
| 400 | `API_KEY_REQUIRED` | The provider needs a key and none was sent or configured; `details` has `provider` and `required_env_vars` |
| 401 | `UNAUTHORIZED` | Missing or invalid credentials (`WWW-Authenticate: Bearer`) |
| 404 / 405 | `NOT_FOUND` / `METHOD_NOT_ALLOWED` | Unknown route or method |
| 422 | `VALIDATION_ERROR` | Invalid body or query; `details.errors` lists `loc`, `msg`, `type` (input values are not echoed) |
| 429 | `RATE_LIMITED` | Rate limit exceeded; `details.retry_after` and a `Retry-After` header |
| 500 | `INTERNAL_SERVER_ERROR` | Unexpected failure |
| 502 | `HTTP_502` | The data provider rejected the server's provider credentials |
| 503 | `SERVICE_UNAVAILABLE` | Provider unreachable, provider rate limit (`Retry-After: 60`), server busy (`Retry-After: 5`, see `API_MAX_CONCURRENT_ANALYSES`), nothing could be analyzed/exported, or the regime history database is unavailable |
| 504 | `TIMEOUT` | Analysis exceeded `API_TIMEOUT` seconds |

Provider failures are classified by the root cause, wherever it is in the exception chain
(the analyzer re-wraps provider errors): unknown symbol → 400, provider auth failure → 502,
provider rate limit or connection/timeout error → 503.

Messages are generic on purpose: provider errors can embed request URLs that carry API keys,
so exception text is only written to the server log, with `apikey=`, `token=`, bearer tokens
and configured secrets redacted.

Successful responses are strict JSON: `NaN`/`Infinity` values (e.g. an undefined
correlation) are returned as `null`.

## Rate Limits

Every HTTP route except `/health`, `/ready` and `/api/v1/health` is limited to
`RATE_LIMIT_PER_MINUTE` requests (default 60) per client IP in a fixed one-minute window.
IPv6 clients are keyed by their /64 prefix. The limit is applied by request path in a small
ASGI middleware (`mra_web/ratelimit.py`), so it covers every route.

Counters live in process memory: with `--workers N`, each worker enforces its own window
(a client can get up to N × the limit), and they reset on restart. Behind a reverse proxy,
set uvicorn's `FORWARDED_ALLOW_IPS` environment variable to the proxy's address (uvicorn
trusts `X-Forwarded-For` only from `127.0.0.1` by default); otherwise every client is keyed by
the proxy's address, for rate limits and WebSocket per-IP caps alike. In Docker the proxy is
usually not `127.0.0.1`.

Independently, `API_MAX_CONCURRENT_ANALYSES` (default 4) bounds analyses running at once per
process (HTTP and WebSocket); excess HTTP requests get `503` with `Retry-After: 5`. A
timed-out analysis keeps its slot until its thread finishes.

## WebSocket Monitoring

`/ws/monitoring/{symbol}` streams regime updates for the daily (`1D`) timeframe.

Query parameters: `provider` (default `alphavantage`), `api_key` (data provider key),
`interval` (seconds between updates, 60–3600, default 300), and optionally `token`.

**Authentication** is checked before the handshake is accepted, from (in order) an
`X-API-Key` header, an `Authorization: Bearer` header, or the `token` query parameter (a JWT
or an API key). Clients that cannot set headers and do not want the token in the URL can
connect without credentials and send `{"token": "<JWT or API key>"}` as the first message
within 5 seconds. In development, connections without credentials are accepted.

**Origin**: browser `Origin` headers must be listed in `CORS_ORIGINS` (`*` allows any);
clients that send no `Origin` (non-browser) are allowed but still need credentials.

**Caps**: `WS_MAX_CONNECTIONS` (default 100) in total and `WS_MAX_CONNECTIONS_PER_IP`
(default 5), per worker process.

**Close codes**:

| Code | Reason |
|------|--------|
| 1008 | Disallowed origin, invalid credentials, invalid symbol/provider/interval, no first-message token within 5 s, or a provider key is required but missing |
| 1013 | Connection cap reached |
| 1011 | 5 consecutive failed analyses, or an internal error |

Each analysis runs in the thread pool (bounded by `API_TIMEOUT` and the shared analysis
slots), so the event loop is never blocked. Monitoring stops as soon as the client
disconnects.

**Messages** are JSON objects `{"message_type", "symbol", "data", "timestamp"}`:

- `connection` — `data`: `status`, `provider`, `interval`, `message`
- `update` — `data`: `symbol`, `current_regime`, `regime_confidence`, `regime_change`,
  `previous_regime`, `alert_level` (`low`; `medium` when confidence < 0.6; `high` on a regime
  change), `timestamp`
- `alert` — sent after an `update` that changes the regime; `data`: `alert_type`
  (`regime_change`), `previous_regime`, `new_regime`, `confidence`, `message`
- `error` — `data`: `error` (`"Analysis failed"` or `"Server busy"`), `error_count`,
  `max_errors`

`GET /ws/monitoring/status` (authenticated) returns `active_connections`,
`monitored_symbols` and `connections_by_symbol`.

### JavaScript

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/monitoring/SPY?provider=yfinance&interval=300');
ws.onopen = () => ws.send(JSON.stringify({ token: TOKEN }));  // JWT or API key
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.message_type, msg.data);
};
```

### Python

```python
import asyncio
import json

import websockets

TOKEN = "..."  # JWT from `uv run mra-token --sub <name>`, or one of API_KEYS


async def monitor_symbol():
    uri = "ws://localhost:8000/ws/monitoring/SPY?provider=yfinance&interval=60"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"token": TOKEN}))
        while True:
            msg = json.loads(await websocket.recv())
            if msg["message_type"] == "update":
                update = msg["data"]
                print(f"Regime: {update['current_regime']} ({update['regime_confidence']:.3f})")
                if update["regime_change"]:
                    print(f"REGIME CHANGE: {update['previous_regime']} -> {update['current_regime']}")


asyncio.run(monitor_symbol())
```

## Python Client

[`examples/api_client.py`](../examples/api_client.py) contains `MarketRegimeAPIClient` and a
demo that exercises the endpoints against `http://localhost:8000`. It reads credentials from
`MRA_TOKEN` (a JWT) or `MRA_API_KEY` (one of `API_KEYS`); neither is needed against a
development server.

```bash
uv run mra-api --dev &                       # or a production server plus credentials:
# export MRA_TOKEN=$(uv run mra-token --sub demo)   (with the server's JWT_SECRET)
uv run examples/api_client.py                # REST demo (uses yfinance)
uv run examples/api_client.py websocket      # 30-second WebSocket demo
```

```python
from examples.api_client import MarketRegimeAPIClient  # run from the repository root

client = MarketRegimeAPIClient("http://localhost:8000", token=TOKEN)  # or service_key=KEY
analysis = client.detailed_analysis("SPY", "1D")  # provider defaults to yfinance here
print(analysis["current_regime"], analysis["regime_confidence"])
portfolio = client.multi_symbol_analysis(["SPY", "QQQ", "IWM"], "1D")
print(portfolio["portfolio_metrics"]["dominant_regime"])
```

## Configuration

Read by `mra_web/config.py` (and `mra_web/server.py` for the bind options):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENVIRONMENT` | `production` | `development` allows unauthenticated requests, an ephemeral JWT secret, docs and `/debug/config` |
| `JWT_SECRET` | none | HS256 secret, 32+ characters; required outside development |
| `JWT_EXPIRATION_HOURS` | `24` | Default lifetime of tokens minted by `mra-token` |
| `API_KEYS` | empty | Comma-separated keys accepted in `X-API-Key` (16+ characters each) |
| `CORS_ORIGINS` | empty | Comma-separated browser origins allowed for CORS and WebSockets |
| `CORS_METHODS` / `CORS_HEADERS` | `GET,POST,OPTIONS` / `Authorization,Content-Type,X-API-Key` | CORS policy (only used when `CORS_ORIGINS` is set) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-client-IP limit (see [Rate Limits](#rate-limits)) |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind address for `mra-api` |
| `API_WORKERS` / `API_RELOAD` | `1` / `false` | `mra-api` worker processes / auto-reload |
| `ENABLE_DOCS` | development only | Serve `/docs`, `/redoc`, `/openapi.json` |
| `API_TIMEOUT` | `300` | Seconds an analysis may run before the request gets `504` |
| `API_MAX_CONCURRENT_ANALYSES` | `4` | Analyses running at once per process (HTTP + WebSocket) |
| `WS_MAX_CONNECTIONS` / `WS_MAX_CONNECTIONS_PER_IP` | `100` / `5` | WebSocket caps per process |
| `MRA_DB_PATH` | `~/.mra/regimes.db` | SQLite regime history database (parent dir created `0700`); docker compose uses `/home/mra/.mra/regimes.db` on the `app-data` volume |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `DEBUG` | `false` | Only reported by `/debug/config` (`--dev` sets it) |

Data provider credentials: `ALPHA_VANTAGE_API_KEY` (or `ALPHAVANTAGE_API_KEY`),
`ALPHA_VANTAGE_PREMIUM`, `POLYGON_API_KEY`, `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY`,
`ALPACA_DATA_FEED` (`iex` or `sip`), `TIINGO_API_KEY`. See [`.env.example`](../.env.example).

The scanner (`mra scan`, compose service `scanner`) writes to the same `MRA_DB_PATH`
database, so `GET /api/v1/regimes/{symbol}/history` serves its records. Its alert
channels are read by the scanner only (the API never sends alerts); each is enabled
only when its variables are set:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALERT_WEBHOOK_URL` | unset | Generic webhook; receives each alert event as a JSON `POST` |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | unset | Telegram Bot API `sendMessage` (both required) |
| `DISCORD_WEBHOOK_URL` | unset | Discord channel webhook (`content` + embed) |
| `SCAN_SYMBOLS` | `SPY,QQQ,IWM` | compose only: `--symbols` of the scanner service |
| `SCAN_TIMEFRAMES` | `1D,1H,15m` | compose only: `--timeframes` |
| `SCAN_INTERVAL` | `3600` | compose only: seconds between scans |
| `SCAN_WATCH` | `1D` | compose only: timeframes whose changes alert |
| `SCAN_MIN_CONFIDENCE` | `0.6` | compose only: minimum regime confidence for an alert |
| `SCAN_PROVIDER` | `yfinance` | compose only: data provider (sets `DEFAULT_PROVIDER`) |

Alert URLs must be `https://` (`http://` only for localhost). The webhook URLs and the bot
token are secrets: the scanner never logs them or puts them in error messages, and the
API's log scrubber also redacts their values if they are set in its environment.

| Provider | API key | Client-side limit (`rate_limit_per_minute`) |
|----------|---------|---------------------------------------------|
| `yfinance` | no | 60 req/min |
| `alphavantage` | yes | 5 req/min (free tier: 25 req/day) |
| `polygon` | yes | 5 req/min (free plan) |
| `alpaca` | key ID + secret | 200 req/min |
| `tiingo` | yes | 1 req/min (free tier: 50 req/hour) |
| `mock` | no | none (offline synthetic data) |

## Docker

The repository [`Dockerfile`](../Dockerfile) builds a two-stage image that runs
`mra-api --host 0.0.0.0 --port 8000` as a non-root user (`ENVIRONMENT=production`), exposes
port 8000 and has a `/health` `HEALTHCHECK`. `JWT_SECRET` (32+ characters) is required.

```bash
# docker compose: reads .env and publishes on 127.0.0.1:${API_PORT:-8000}. Only the
# variables listed in docker-compose.yml reach the container; add others there.
cp .env.example .env   # set JWT_SECRET and any provider keys
docker compose up -d --build

# plain docker
docker build -t market-regime-analysis .
docker run -p 127.0.0.1:8000:8000 \
  -e JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  market-regime-analysis

# compose keeps the regime history database on the app-data volume
# (MRA_DB_PATH=/home/mra/.mra/regimes.db); with plain docker add
#   -v mra-data:/home/mra/.mra

# optional scheduled scanner (shares app-data with api; see "Scanner & alerts" in README)
docker compose --profile scanner up -d

# the image also contains the CLI and the token minter
docker run --rm --entrypoint mra market-regime-analysis --provider mock current-analysis
docker run --rm -e JWT_SECRET=... --entrypoint mra-token market-regime-analysis --sub alice
```

## Production Notes

```bash
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export API_KEYS="<long-random-key>"
export CORS_ORIGINS=https://yourdomain.com
uv run mra-api --host 0.0.0.0 --port 8000 --workers 4
```

Put a TLS-terminating reverse proxy in front. WebSockets need the upgrade headers:

```nginx
server {
    listen 80;
    server_name your-api-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Known limitations: rate-limit counters, metrics and WebSocket caps are per process (there is
no shared store), and fitted models are not cached, so every request refits the HMM.

## Troubleshooting

- **Server exits with "Refusing to start"** — set a `JWT_SECRET` of 32+ characters, or use
  `--dev` locally.
- **`401` everywhere** — send `Authorization: Bearer $TOKEN` or `X-API-Key`; a token minted
  with a different `JWT_SECRET` (or an ephemeral development secret) is rejected.
- **`400 API_KEY_REQUIRED`** — the default provider is `alphavantage`; pass
  `"provider": "yfinance"` (or `mock`), or set the provider's key.
- **`503` with `Retry-After: 5`** — all analysis slots are busy; retry or raise
  `API_MAX_CONCURRENT_ANALYSES`.
- **WebSocket closes with 1008** — check the credentials, `Origin` vs `CORS_ORIGINS`, and the
  `interval` range.
- **Debug configuration** — in development, `curl http://localhost:8000/debug/config`.
