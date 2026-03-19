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
```

### Start the Server

```bash
# Development mode with auto-reload
uv run mra-api --dev

# Production mode
uv run mra-api --host 0.0.0.0 --port 8000 --workers 4

# Or directly with uvicorn
uvicorn mra_web.app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the API

- **API Documentation**: <http://localhost:8000/docs> (Swagger UI)
- **Alternative Docs**: <http://localhost:8000/redoc> (ReDoc)
- **Health Check**: <http://localhost:8000/health>
- **Metrics**: <http://localhost:8000/metrics>

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

Multi-timeframe analysis across 1D, 1H, and 15m intervals.

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/current" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "provider": "yfinance"
  }'
```

#### POST `/api/v1/analysis/multi-symbol`

Portfolio analysis across multiple symbols.

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

Generate HMM visualization charts.

```bash
curl -X POST "http://localhost:8000/api/v1/charts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "timeframe": "1D",
    "days": 60,
    "provider": "yfinance"
  }'
```

#### POST `/api/v1/export/csv`

Export analysis data to CSV format.

```bash
curl -X POST "http://localhost:8000/api/v1/export/csv" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "provider": "yfinance",
    "filename": "spy_analysis.csv"
  }'
```

## 🌐 WebSocket Monitoring

Real-time regime monitoring via WebSocket connections.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/monitoring/SPY?provider=yfinance&interval=300');

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

The API supports two authentication methods:

### 1. JWT Bearer Tokens

```bash
# Get token
curl -X POST "http://localhost:8000/auth/token?username=demo"

# Use token
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/analysis/detailed" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"}'
```

### 2. API Keys (Future Enhancement)

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/api/v1/providers"
```

## 🐍 Python Client

Use the provided Python client for easy integration:

```python
from examples_api import MarketRegimeAPIClient

# Initialize client
client = MarketRegimeAPIClient("http://localhost:8000")

# Authenticate (optional in development)
client.authenticate("demo_user")

# Run analysis
analysis = client.detailed_analysis("SPY", "1D")
print(f"Current regime: {analysis['current_regime']}")
print(f"Confidence: {analysis['regime_confidence']:.3f}")

# Portfolio analysis
portfolio = client.multi_symbol_analysis(["SPY", "QQQ", "IWM"], "1D")
print(f"Dominant regime: {portfolio['portfolio_metrics']['dominant_regime']}")

# Position sizing
sizing = client.position_sizing(0.02, "Bull Trending", 0.8, 0.75)
print(f"Recommended size: {sizing['final_recommendation']:.1%}")
```

## 🏃 Running Examples

```bash
# Run all API examples
uv run examples/api_client.py

# Test WebSocket monitoring
uv run examples/api_client.py websocket
```

## ⚙️ Configuration

### Environment Variables

```bash
# Server configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
export ENVIRONMENT=production
export DEBUG=false

# Authentication
export JWT_SECRET=your-secret-key-change-in-production
export JWT_EXPIRATION_HOURS=24

# Rate limiting
export RATE_LIMIT_PER_MINUTE=60
export RATE_LIMIT_BURST=10

# CORS
export CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"

# Data providers
export ALPHA_VANTAGE_API_KEY=your_key_here
export POLYGON_API_KEY=your_key_here
```

### Data Providers

| Provider | API Key Required | Rate Limit | Data Quality |
|----------|------------------|------------|--------------|
| **Yahoo Finance** | No | 60 req/min | Community |
| **Alpha Vantage** | Yes | 5 req/min | Professional |
| **Polygon.io** | Yes | 60+ req/min | Institutional |

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

```json
{
  "error_code": "PROVIDER_API_ERROR",
  "message": "Failed to fetch data from Alpha Vantage",
  "details": {
    "provider": "alphavantage",
    "symbol": "SPY",
    "timeframe": "1D",
    "retry_after": 60
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## 🚀 Performance

- **Async Support**: Non-blocking I/O for concurrent requests
- **Rate Limiting**: Configurable per-endpoint rate limits
- **Caching**: Response caching for frequently accessed data
- **Parallel Processing**: Multi-timeframe analysis runs in parallel
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
# Get API metrics
curl http://localhost:8000/metrics
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

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "mra-api", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
# Build and run
docker build -t market-regime-api .
docker run -p 8000:8000 -e ALPHA_VANTAGE_API_KEY=your_key market-regime-api
```

## 🎯 Production Deployment

### Environment Setup

```bash
# Production configuration
export ENVIRONMENT=production
export DEBUG=false
export JWT_SECRET=your-super-secure-secret-key
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

# Check debug configuration
curl http://localhost:8000/debug/config
```

## Support

- Interactive docs at `/docs` (Swagger UI)
- Error responses include detailed error information
- Health check endpoints for system status
