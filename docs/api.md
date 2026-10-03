# Market Regime Analysis API

REST API for market regime analysis using Hidden Markov Models.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Set up environment variables (optional)
export ALPHA_VANTAGE_API_KEY=your_key_here
export POLYGON_API_KEY=your_key_here
export APCA_API_KEY_ID=your_key_id APCA_API_SECRET_KEY=your_secret
export TIINGO_API_KEY=your_key_here
```

### Start the Server

```bash
# Development mode: auto-reload, docs enabled, unauthenticated requests allowed
uv run mra-api --dev

# Production (the default environment): a JWT secret of 32+ characters is required
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uv run mra-api --host 0.0.0.0 --port 8000 --workers 4
```

Outside `ENVIRONMENT=development` the server refuses to start when `JWT_SECRET` is unset,
empty, a placeholder, or shorter than 32 characters. In development a random per-process
secret is generated (with a warning) instead.

### Access the API

- **API Documentation**: <http://localhost:8000/docs> (Swagger UI; development or `ENABLE_DOCS=true`)
- **Alternative Docs**: <http://localhost:8000/redoc> (ReDoc; same condition)
- **Health Check**: <http://localhost:8000/health> (public)
- **Metrics**: <http://localhost:8000/metrics> (authenticated)

All `/api/v1/*` examples below omit credentials for brevity. Outside development, add
`-H "Authorization: Bearer $TOKEN"` or `-H "X-API-Key: $KEY"` (see [Authentication](#-authentication)).

## 📚 API Endpoints

### Analysis Endpoints

#### POST `/api/v1/analysis/detailed`

Single timeframe HMM analysis with comprehensive metrics.

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/detailed" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "timeframe": "1D",
    "provider": "yfinance"
  }'
```

#### POST `/api/v1/analysis/current`

Multi-timeframe analysis across 1D, 1H, and 15m intervals. Each timeframe is loaded and
analyzed independently; timeframes that fail are omitted (503 only if all fail).

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/current" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "provider": "yfinance"
  }'
```

#### POST `/api/v1/analysis/multi-symbol`

Portfolio analysis across multiple symbols (max 20). Each symbol is analyzed once;
`correlations` are correlations of returns over the symbols that loaded (`null` where
undefined), and `correlation_risk` is `null` with fewer than two analyzed symbols.

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/multi-symbol" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["SPY", "QQQ", "IWM"],
    "timeframe": "1D",
    "provider": "yfinance"
  }'
```

### Utility Endpoints

#### POST `/api/v1/position-sizing`

Kelly Criterion-based position sizing with regime adjustments.

```bash
curl -X POST "http://localhost:8000/api/v1/position-sizing" \
  -H "Content-Type: application/json" \
  -d '{
    "base_size": 0.02,
    "regime": "Bull Trending",
    "confidence": 0.8,
    "persistence": 0.75,
    "correlation": 0.1
  }'
```

#### GET `/api/v1/providers`

List available data providers and their capabilities.

```bash
curl "http://localhost:8000/api/v1/providers"
```

#### POST `/api/v1/charts/generate`

Render the 5-panel HMM regime chart. Returns `image/png`.

```bash
curl -X POST "http://localhost:8000/api/v1/charts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "timeframe": "1D",
    "days": 60,
    "provider": "yfinance"
  }' -o spy_chart.png
```

#### POST `/api/v1/export/csv`

Export the analysis (one row per timeframe) as `text/csv`. The CSV is built in memory and
returned in the response; nothing is written on the server. `filename` (optional, letters,
digits, `.`, `_`, `-`) only sets the download name in `Content-Disposition`. The
`X-Record-Count` header holds the number of rows.

```bash
curl -X POST "http://localhost:8000/api/v1/export/csv" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "provider": "yfinance",
    "filename": "spy_analysis.csv"
  }' -o spy_analysis.csv
```

Symbols must match `^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$` after upper-casing (e.g. `BRK.B`,
`^GSPC`, `ES=F`). Multi-symbol requests accept at most 20 symbols; duplicates are dropped.

## 🌐 WebSocket Monitoring

Real-time regime monitoring via WebSocket connections.

WebSocket connections must authenticate (except in development). The server checks, in
order, an `X-API-Key` header, an `Authorization: Bearer` header, or a `token` query
parameter (a JWT or API key) before accepting the handshake. Clients that cannot set headers
can instead send `{"token": "<JWT or API key>"}` as the first message within 5 seconds.
Invalid or missing credentials, an unknown `provider`, or an `interval` outside 60-3600
seconds close the socket with code `1008`. Analysis runs in a worker thread (bounded by
`API_TIMEOUT`); the server stops monitoring as soon as the client disconnects, and closes
with `1011` after 5 consecutive failed analyses (each reported as a generic `error` message). Browser `Origin` headers
must be listed in `CORS_ORIGINS`. Connections are capped (`WS_MAX_CONNECTIONS`, default 100;
`WS_MAX_CONNECTIONS_PER_IP`, default 5); over the cap the handshake is refused (code `1013`).
`api_key` in the query string is the **data provider** key, not an API credential.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/monitoring/SPY?provider=yfinance&interval=300');
ws.onopen = () => ws.send(JSON.stringify({ token: TOKEN }));

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Message type:', data.message_type);
    console.log('Data:', data.data);
};
```

### Python Example

```python
import asyncio
import websockets
import json

async def monitor_symbol():
    uri = "ws://localhost:8000/ws/monitoring/SPY?provider=yfinance&interval=60"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"token": TOKEN}))
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["message_type"] == "update":
                update = data["data"]
                print(f"Regime: {update['current_regime']}")
                print(f"Confidence: {update['regime_confidence']:.3f}")

                if update["regime_change"]:
                    print(f"🚨 REGIME CHANGE: {update['previous_regime']} → {update['current_regime']}")

asyncio.run(monitor_symbol())
```

## 🔐 Authentication

Every `/api/v1/*` route (except `/api/v1/health`), `/metrics`, `/ws/monitoring/status` and
the WebSocket endpoint require credentials unless `ENVIRONMENT=development`. In
development, requests **without** credentials are accepted as `dev_user`; credentials
that are sent are still validated. Missing or invalid credentials return `401`.

There is no token-issuing endpoint. Tokens are minted out of band with the server's secret.

### 1. JWT Bearer Tokens

Tokens are HS256-signed with `JWT_SECRET` and must carry `sub` and `exp` claims; other
algorithms (including `none`) are rejected.

```bash
# Mint a token where JWT_SECRET is available (lifetime defaults to JWT_EXPIRATION_HOURS)
TOKEN=$(JWT_SECRET=... uv run mra-token --sub alice --hours 12)

curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/analysis/detailed" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"}'
```

### 2. API Keys

Set `API_KEYS` to a comma-separated list of long random keys (16+ characters) and send
one in the `X-API-Key` header. Keys are compared in constant time. Query-string keys are
not accepted.

```bash
export API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
curl -H "X-API-Key: $KEY" "http://localhost:8000/api/v1/providers"
```

## 🐍 Python Client

`examples/api_client.py` contains a small client:

```python
from examples.api_client import MarketRegimeAPIClient

# JWT from `uv run mra-token --sub demo`, or service_key=<one of API_KEYS>
client = MarketRegimeAPIClient("http://localhost:8000", token=TOKEN)

analysis = client.detailed_analysis("SPY", "1D")
print(f"Current regime: {analysis['current_regime']}")
print(f"Confidence: {analysis['regime_confidence']:.3f}")

portfolio = client.multi_symbol_analysis(["SPY", "QQQ", "IWM"], "1D")
print(f"Dominant regime: {portfolio['portfolio_metrics']['dominant_regime']}")
```

## 🏃 Running Examples

```bash
# Run all API examples (set MRA_TOKEN or MRA_API_KEY unless the server is in development)
uv run examples/api_client.py

# Test WebSocket monitoring
uv run examples/api_client.py websocket
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENVIRONMENT` | `production` | `development` enables the unauthenticated dev user, docs and `/debug/config` |
| `JWT_SECRET` | none | HS256 secret, 32+ characters; required outside development |
| `JWT_EXPIRATION_HOURS` | `24` | Default lifetime of tokens minted by `mra-token` |
| `API_KEYS` | empty | Comma-separated keys accepted in `X-API-Key` (16+ characters each) |
| `CORS_ORIGINS` | empty | Comma-separated browser origins allowed for CORS and WebSockets |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-client-IP limit for all HTTP routes except health probes |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind address for `mra-api` |
| `ENABLE_DOCS` | dev only | Serve `/docs`, `/redoc`, `/openapi.json` |
| `WS_MAX_CONNECTIONS` / `WS_MAX_CONNECTIONS_PER_IP` | `100` / `5` | WebSocket caps |
| `API_TIMEOUT` | `300` | Seconds an analysis may run before the request gets `504` |
| `API_MAX_CONCURRENT_ANALYSES` | `4` | Analyses running at once per process (HTTP + WebSocket); excess requests get `503` with `Retry-After`. A timed-out analysis keeps its slot until its thread finishes |
| `API_WORKERS`, `API_RELOAD`, `LOG_LEVEL`, `DEBUG` | | Server tuning |

```bash
# Data providers
export ALPHA_VANTAGE_API_KEY=your_key_here
export POLYGON_API_KEY=your_key_here
export APCA_API_KEY_ID=your_key_id
export APCA_API_SECRET_KEY=your_secret
export ALPACA_DATA_FEED=iex   # or sip
export TIINGO_API_KEY=your_key_here
```

Rate limits are kept in process memory, so each worker enforces its own window. Behind a
reverse proxy, run uvicorn with `--proxy-headers` and trusted `--forwarded-allow-ips` so the
limit applies per real client rather than per proxy.

### Data Providers

| Provider | API Key Required | Rate Limit | Data Quality |
|----------|------------------|------------|--------------|
| **Yahoo Finance** | No | 60 req/min | Community |
| **Alpha Vantage** | Yes | 5 req/min | Professional |
| **Polygon.io** | Yes | 60+ req/min | Institutional |
| **Alpaca** | Yes (key ID + secret) | 200 req/min (free) | IEX (free) / SIP |
| **Tiingo** | Yes | 50 req/hour (free) | Adjusted EOD + IEX intraday |

For `provider: "alpaca"`, set `api_key` in the request to `"KEY_ID:SECRET_KEY"`, or set both `APCA_*` variables on the server.

## 📊 Response Format

All API responses follow a consistent format:

### Success Response

```json
{
  "symbol": "SPY",
  "timeframe": "1D",
  "current_regime": "Bull Trending",
  "regime_confidence": 0.847,
  "regime_persistence": 0.723,
  "transition_probability": 0.156,
  "hmm_state": 2,
  "risk_level": "Medium",
  "position_sizing_multiplier": 1.25,
  "recommended_strategy": "Momentum Following",
  "analysis_timestamp": "2024-01-15T10:30:00.000Z",
  "metrics": {
    "raw_features": [...],
    "state_probabilities": [...],
    "regime_description": "Strong upward momentum with high persistence",
    "statistical_features": {...}
  }
}
```

### Error Response

Every error (auth, validation, routing, rate limit, provider, unexpected) uses one envelope:

```json
{
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "Data provider unavailable",
  "details": {},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

| Status | `error_code` | When |
|--------|--------------|------|
| 400 | `BAD_REQUEST`, `API_KEY_REQUIRED` | Invalid input / no data for the symbol; provider key missing (`details.required_env_vars`) |
| 401 | `UNAUTHORIZED` | Missing or invalid credentials |
| 404 / 405 | `NOT_FOUND` / `METHOD_NOT_ALLOWED` | Unknown route or method |
| 422 | `VALIDATION_ERROR` | Request body/query invalid; `details.errors` lists `loc`, `msg`, `type` (input values are not echoed) |
| 429 | `RATE_LIMITED` | Rate limit exceeded (`Retry-After` header) |
| 500 | `INTERNAL_SERVER_ERROR` | Unexpected failure |
| 502 | `HTTP_502` | The data provider rejected the server's provider credentials |
| 503 | `SERVICE_UNAVAILABLE` | Provider unreachable or rate-limited (`Retry-After`), or nothing could be analyzed |
| 504 | `TIMEOUT` | Analysis exceeded `API_TIMEOUT` seconds |

Responses are strict JSON: `NaN`/`Infinity` values (e.g. an undefined correlation or
confidence) are returned as `null`.

Error messages are generic on purpose: provider errors can embed request URLs that carry
API keys, so exception text is only written to the server log (with `apikey=`, `token=`,
bearer tokens and configured secrets redacted). Rate-limited requests return `429` with a
`Retry-After` header.

## 🚀 Performance

- **Async Support**: Analysis runs in a thread pool, so the event loop stays responsive
- **Per-timeframe loading**: Single-timeframe requests load only that timeframe
- **Rate Limiting**: Per-client limit on all HTTP routes (`RATE_LIMIT_PER_MINUTE`)
- **WebSocket Streaming**: Real-time updates with minimal latency

## 🔧 Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Readiness check (for K8s)
curl http://localhost:8000/ready
```

### Metrics

```bash
# Get API metrics (authenticated)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/metrics
```

Returns:

- Request counts by endpoint
- Error rates and types
- Average response times
- WebSocket connection statistics
- System uptime

### Logging

All API requests and errors are logged with structured format:

```text
2024-01-15 10:30:00 - INFO - API Request - Endpoint: /analysis/detailed, Client: 192.168.1.100
2024-01-15 10:30:01 - INFO - API Response - Endpoint: /analysis/detailed, Status: 200, Time: 0.856s
```

## 🐳 Docker Deployment

The repository `Dockerfile` builds a multi-stage image that runs `mra-api` as a non-root user on
port 8000 with a `/health` healthcheck. `JWT_SECRET` (>=32 characters) is required.

```bash
# docker compose (reads .env; publishes on 127.0.0.1:${API_PORT:-8000})
cp .env.example .env   # then set JWT_SECRET and any provider keys
docker compose up -d --build

# or plain docker
docker build -t market-regime-analysis .
docker run -p 127.0.0.1:8000:8000 -e JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  -e ALPHA_VANTAGE_API_KEY=your_key market-regime-analysis
```

## 🎯 Production Deployment

### Environment Setup

```bash
# Production configuration (ENVIRONMENT defaults to production)
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export API_KEYS="<long-random-key>"
export CORS_ORIGINS=https://yourdomain.com
export RATE_LIMIT_PER_MINUTE=100
```

### Run with Gunicorn

```bash
pip install gunicorn
gunicorn mra_web.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-api-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
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

## 📈 Scaling

### Horizontal Scaling

- Deploy multiple API server instances behind a load balancer
- Use Redis for shared rate limiting and session storage
- Configure WebSocket sticky sessions for real-time monitoring

### Performance Optimization

- Enable response compression (gzip)
- Implement caching layer (Redis/Memcached)
- Use connection pooling for data providers
- Monitor and tune worker process counts

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**

   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 PID
   ```

2. **Missing API Keys**

   ```bash
   # Check environment variables
   echo $ALPHA_VANTAGE_API_KEY
   echo $POLYGON_API_KEY
   ```

3. **Rate Limiting**

   ```bash
   # Check current limits
   curl http://localhost:8000/metrics
   ```

4. **WebSocket Connection Issues**
   - Verify WebSocket URL format
   - Check proxy configurations
   - Monitor server logs for connection errors

### Debug Mode

```bash
# Start in debug mode
uv run mra-api --dev

# Check debug configuration (development only, secrets omitted)
curl http://localhost:8000/debug/config
```

## Support

- Interactive docs at `/docs` (Swagger UI; development or `ENABLE_DOCS=true`)
- Server logs carry error details; responses stay generic
- Health check endpoints for system status
