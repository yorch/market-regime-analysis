"""
FastAPI server for Market Regime Analysis API.

``create_app`` builds the application from an :class:`APIConfig`; the module-level
``app`` uses the configuration loaded from the environment.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mra_web.auth import User, authenticate_request
from mra_web.config import APIConfig, config
from mra_web.endpoints import router as api_router
from mra_web.ratelimit import RateLimitMiddleware
from mra_web.security import install_log_scrubber
from mra_web.utils import NumpyJSONResponse, api_metrics, create_error_response
from mra_web.websocket import manager, ws_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
install_log_scrubber()
logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Market regime analysis API based on Hidden Markov Models.

## Authentication

Every `/api/v1/*` route, `/metrics` and the WebSocket endpoints require one of:
- **JWT Bearer token** (`Authorization: Bearer <token>`), HS256-signed with `JWT_SECRET`
  and carrying `sub` and `exp` claims. Mint one with `uv run mra-token --sub <name>`.
- **API key** (`X-API-Key: <key>`), one of the keys in `API_KEYS`.

## Rate Limiting

All API routes are limited per client IP (`RATE_LIMIT_PER_MINUTE`, default 60/min).

## WebSocket Monitoring

- **Endpoint**: `/ws/monitoring/{symbol}`
- **Parameters**: `provider`, `api_key` (data provider key), `interval`, `token`
"""


# Health probes are never rate limited.
HEALTH_PATHS = ("/health", "/ready", "/api/v1/health")


def _health_payload() -> dict:
    return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}


def create_app(cfg: APIConfig | None = None) -> FastAPI:
    """Build the FastAPI application for a configuration."""
    cfg = cfg or config

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan context manager."""
        install_log_scrubber()  # uvicorn may have (re)configured its handlers
        app.state.start_time = time.time()
        logger.info("Starting Market Regime Analysis API Server")
        logger.info("Environment: %s", cfg.environment)
        logger.info("Rate limit: %s req/min", cfg.rate_limit_per_minute)
        if cfg.jwt_secret_ephemeral:
            logger.warning("Using an ephemeral development JWT secret")
        yield
        logger.info("Shutting down Market Regime Analysis API Server")

    docs = cfg.docs_enabled
    app = FastAPI(
        title="Market Regime Analysis API",
        description=API_DESCRIPTION,
        version="1.0.0",
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
        lifespan=lifespan,
        default_response_class=NumpyJSONResponse,
    )
    app.state.config = cfg

    # Rate limiting: every HTTP route except health probes, per client IP.
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=cfg.rate_limit_per_minute,
        exempt_paths=HEALTH_PATHS,
    )

    # CORS: no origins unless configured; never credentials with a wildcard.
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials="*" not in cfg.cors_origins,
            allow_methods=cfg.cors_methods,
            allow_headers=cfg.cors_headers,
        )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header to responses."""
        start_time = time.time()
        response = await call_next(request)
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        return response

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        """Handle HTTP exceptions with standardized error format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
            if isinstance(exc.detail, dict)
            else {
                "error_code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": time.time(),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(_request: Request, exc: Exception):
        """Handle unexpected exceptions without exposing internals."""
        logger.error("Unhandled exception", exc_info=exc)
        error_response = create_error_response(
            "INTERNAL_SERVER_ERROR", "An unexpected error occurred"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode="json"),
        )

    # Include routers
    app.include_router(api_router, prefix="")
    app.include_router(ws_router, prefix="/ws")

    @app.get("/", tags=["root"])
    async def root(request: Request):
        """Root endpoint with API information."""
        return {
            "name": "Market Regime Analysis API",
            "version": "1.0.0",
            "description": "Market regime analysis using HMM methodology",
            "documentation": "/docs" if docs else None,
            "health_check": "/health",
            "providers": "/api/v1/providers",
            "websocket_monitoring": "/ws/monitoring/{symbol}",
        }

    @app.get("/health", tags=["health"])
    async def health_check():
        """Basic health check endpoint."""
        return _health_payload()

    @app.get("/ready", tags=["health"])
    async def readiness_check():
        """Readiness check for container orchestration."""
        checks = {
            "api": "healthy",
            "websocket_manager": "healthy" if manager else "unhealthy",
            "metrics": "healthy" if api_metrics else "unhealthy",
        }
        all_healthy = all(state == "healthy" for state in checks.values())
        return {
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": time.time(),
        }

    @app.get("/metrics", tags=["monitoring"])
    async def get_api_metrics(
        request: Request,
        current_user: User = Depends(authenticate_request),  # noqa: B008
    ):
        """Get API metrics and usage statistics (authenticated)."""
        metrics = api_metrics.get_metrics()
        metrics["websocket_connections"] = {
            "total_connections": manager.get_connection_count(),
            "active_symbols": manager.get_active_symbols(),
            "connections_by_symbol": {
                symbol: manager.get_connection_count(symbol)
                for symbol in manager.get_active_symbols()
            },
        }
        return metrics

    # Debug endpoint (development only)
    if cfg.is_development:

        @app.get("/debug/config", tags=["debug"])
        async def debug_config():
            """Get current configuration (development only, secrets omitted)."""
            return {
                "host": cfg.host,
                "port": cfg.port,
                "environment": cfg.environment,
                "debug": cfg.debug,
                "rate_limit_per_minute": cfg.rate_limit_per_minute,
                "cors_origins": cfg.cors_origins,
                "log_level": cfg.log_level,
                "jwt_expiration_hours": cfg.jwt_expiration_hours,
                "api_keys_configured": len(cfg.api_keys),
                "docs_enabled": docs,
            }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mra_web.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level.lower(),
        workers=config.workers if not config.reload else 1,
    )
