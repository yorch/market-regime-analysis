"""
FastAPI server for Market Regime Analysis API.

This is the main FastAPI application that serves all API endpoints
following the implementation plan.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config import config
from market_regime_analysis.api.auth import create_user_token
from market_regime_analysis.api.endpoints import router as api_router
from market_regime_analysis.api.utils import NumpyJSONResponse, api_metrics, create_error_response
from market_regime_analysis.api.websocket import manager, ws_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    app.state.start_time = time.time()
    logger.info("Starting Market Regime Analysis API Server")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"Rate limit: {config.rate_limit_per_minute} req/min")
    logger.info("Market Regime Analysis API Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Market Regime Analysis API Server")


# Create FastAPI app
app = FastAPI(
    title="Market Regime Analysis API",
    description="""
    Professional-grade market regime analysis API implementing Jim Simons' Hidden Markov Model methodology.
    
    ## Features
    
    - **Comprehensive Analysis**: Multi-timeframe regime detection using HMM
    - **Portfolio Analysis**: Cross-asset correlation and regime analysis
    - **Real-time Monitoring**: WebSocket streaming for continuous updates
    - **Risk Management**: Kelly Criterion-based position sizing
    - **Multiple Providers**: Support for Alpha Vantage, Polygon.io, and Yahoo Finance
    
    ## Authentication
    
    This API supports two authentication methods:
    - **JWT Bearer Tokens**: For user-based authentication
    - **API Keys**: For service-to-service authentication
    
    ## Rate Limiting
    
    API requests are rate limited to prevent abuse:
    - **Default**: 60 requests per minute per IP
    - **Burst**: Up to 10 additional requests
    
    ## WebSocket Monitoring
    
    Real-time regime monitoring is available via WebSocket:
    - **Endpoint**: `/ws/monitoring/{symbol}`
    - **Parameters**: `provider`, `api_key`, `interval`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    default_response_class=NumpyJSONResponse,
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=config.cors_methods,
    allow_headers=config.cors_headers,
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
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
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception):
    """Handle general exceptions with standardized error format."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Don't expose internal details in production
    error_details = {"type": type(exc).__name__} if config.debug else {}

    error_response = create_error_response(
        "INTERNAL_SERVER_ERROR",
        "An unexpected error occurred" if not config.debug else str(exc),
        error_details,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response.model_dump()
    )


# Include routers
app.include_router(api_router, prefix="")
app.include_router(ws_router, prefix="/ws")


# Root endpoint
@app.get("/", tags=["root"])
@limiter.limit(f"{config.rate_limit_per_minute}/minute")
async def root(request: Request):
    """Root endpoint with API information."""
    return {
        "name": "Market Regime Analysis API",
        "version": "1.0.0",
        "description": "Professional market regime analysis using HMM methodology",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "providers": "/api/v1/providers",
        "websocket_monitoring": "/ws/monitoring/{symbol}",
        "environment": config.environment,
        "uptime_seconds": (time.time() - app.state.start_time)
        if hasattr(app.state, "start_time")
        else 0,
    }


# Authentication endpoints
@app.post("/auth/token", tags=["authentication"])
@limiter.limit("10/minute")
async def login_for_access_token(request: Request, username: str = "demo"):
    """
    Get an access token for testing purposes.

    In production, this would validate credentials against a database.
    For now, it creates tokens for any username for testing.
    """
    if not username or username.strip() == "":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required")

    access_token = create_user_token(username.strip())

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": config.jwt_expiration_hours * 3600,
        "username": username.strip(),
    }


# Health check endpoints
@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "environment": config.environment,
    }


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Readiness check for container orchestration."""
    try:
        # Basic checks
        checks = {
            "api": "healthy",
            "websocket_manager": "healthy" if manager else "unhealthy",
            "metrics": "healthy" if api_metrics else "unhealthy",
        }

        # Check if any critical component is unhealthy
        all_healthy = all(status == "healthy" for status in checks.values())

        return {
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready"
        ) from e


# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
@limiter.limit("30/minute")
async def get_api_metrics(request: Request):
    """Get API metrics and usage statistics."""
    metrics = api_metrics.get_metrics()

    # Add WebSocket connection info
    metrics["websocket_connections"] = {
        "total_connections": manager.get_connection_count(),
        "active_symbols": manager.get_active_symbols(),
        "connections_by_symbol": {
            symbol: manager.get_connection_count(symbol) for symbol in manager.get_active_symbols()
        },
    }

    return metrics


# Startup logic moved to lifespan context manager above


# Debug endpoint (development only)
if config.environment == "development":

    @app.get("/debug/config", tags=["debug"])
    async def debug_config():
        """Get current configuration (development only)."""
        return {
            "host": config.host,
            "port": config.port,
            "environment": config.environment,
            "debug": config.debug,
            "rate_limit_per_minute": config.rate_limit_per_minute,
            "cors_origins": config.cors_origins,
            "log_level": config.log_level,
            "jwt_expiration_hours": config.jwt_expiration_hours,
        }


if __name__ == "__main__":
    import uvicorn

    # Note: start_time is now set in the lifespan context manager
    # This ensures it's properly set even when running with reload=True

    # Run server
    uvicorn.run(
        "api_server:app",
        host=config.host,
        port=config.port,
        reload=config.reload if hasattr(config, "reload") else False,
        log_level=config.log_level.lower(),
        workers=getattr(config, "workers", 1) if not getattr(config, "reload", False) else 1,
    )
