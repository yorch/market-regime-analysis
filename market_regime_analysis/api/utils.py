"""
Utility functions for the Market Regime Analysis API.

This module provides helper functions and error handling utilities.
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ..data_classes import RegimeAnalysis
from ..enums import MarketRegime, TradingStrategy
from .models import AnalysisResponse, ErrorResponse

# Setup logging
logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and datetime objects."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class NumpyJSONResponse(JSONResponse):
    """Custom JSONResponse that handles numpy types."""

    def render(self, content) -> bytes:
        return json.dumps(content, cls=NumpyJSONEncoder, ensure_ascii=False).encode("utf-8")


def create_error_response(
    error_code: str, message: str, details: dict[str, Any] | None = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details or {},
        timestamp=datetime.now(UTC),
    )


def handle_api_exception(error_code: str, message: str, status_code: int = 500) -> HTTPException:
    """Create an HTTPException with standardized error format."""
    error_response = create_error_response(error_code, message)
    raise HTTPException(status_code=status_code, detail=error_response.model_dump())


def convert_numpy_types(obj: Any) -> Any:  # noqa: PLR0911
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def validate_api_key(provider: str, api_key: str | None) -> str:
    """Validate and retrieve API key for providers that require it."""

    if provider not in ["alphavantage", "polygon"]:
        return api_key or ""

    if api_key:
        return api_key

    # Try environment variables
    env_keys = {
        "alphavantage": ["ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"],
        "polygon": ["POLYGON_API_KEY"],
    }

    for env_var in env_keys.get(provider, []):
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

    provider_name = provider.replace("_", " ").title()
    raise HTTPException(
        status_code=400,
        detail=create_error_response(
            "API_KEY_REQUIRED",
            f"{provider_name} API key is required when using {provider} provider.",
            {
                "provider": provider,
                "required_env_vars": env_keys[provider],
            },
        ).model_dump(),
    )


def convert_regime_analysis_to_response(
    analysis: RegimeAnalysis, symbol: str, timeframe: str
) -> AnalysisResponse:
    """Convert RegimeAnalysis dataclass to API response model."""
    return AnalysisResponse(
        symbol=symbol,
        timeframe=timeframe,
        current_regime=analysis.current_regime.value,
        regime_confidence=convert_numpy_types(analysis.regime_confidence),
        regime_persistence=convert_numpy_types(analysis.regime_persistence),
        transition_probability=convert_numpy_types(analysis.transition_probability),
        hmm_state=convert_numpy_types(analysis.hmm_state),
        risk_level=analysis.risk_level,
        position_sizing_multiplier=convert_numpy_types(analysis.position_sizing_multiplier),
        recommended_strategy=analysis.recommended_strategy.value,
        analysis_timestamp=datetime.now(UTC),
        metrics=convert_numpy_types(
            {
                "arbitrage_opportunities": analysis.arbitrage_opportunities,
                "statistical_signals": analysis.statistical_signals,
                "key_levels": analysis.key_levels,
                "hmm_state": analysis.hmm_state,
                "transition_probability": analysis.transition_probability,
            }
        ),
    )


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


def log_api_request(
    endpoint: str, request_data: dict[str, Any], client_ip: str | None = None
) -> None:
    """Log API request for monitoring and debugging."""
    logger.info(
        f"API Request - Endpoint: {endpoint}, Client: {client_ip or 'unknown'}, "
        f"Data: {sanitize_log_data(request_data)}"
    )


def log_api_response(
    endpoint: str, response_status: int, response_time: float, client_ip: str | None = None
) -> None:
    """Log API response for monitoring and debugging."""
    logger.info(
        f"API Response - Endpoint: {endpoint}, Status: {response_status}, "
        f"Time: {response_time:.3f}s, Client: {client_ip or 'unknown'}"
    )


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive information from log data."""
    sanitized = data.copy()
    sensitive_keys = ["api_key", "password", "token", "secret"]

    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "***REDACTED***"

    return sanitized


def get_regime_from_string(regime_str: str) -> MarketRegime:
    """Convert string regime to MarketRegime enum."""
    for regime in MarketRegime:
        if regime.value == regime_str:
            return regime
    raise ValueError(f"Invalid regime: {regime_str}")


def get_strategy_from_string(strategy_str: str) -> TradingStrategy:
    """Convert string strategy to TradingStrategy enum."""
    for strategy in TradingStrategy:
        if strategy.value == strategy_str:
            return strategy
    raise ValueError(f"Invalid strategy: {strategy_str}")


class APIMetrics:
    """Simple in-memory metrics collector for API monitoring."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.request_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self.response_times: dict[str, list[float]] = {}
        self.start_time = datetime.now(UTC)

    def record_request(self, endpoint: str) -> None:
        """Record a request to an endpoint."""
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1

    def record_error(self, endpoint: str) -> None:
        """Record an error for an endpoint."""
        self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1

    def record_response_time(self, endpoint: str, response_time: float) -> None:
        """Record response time for an endpoint."""
        if endpoint not in self.response_times:
            self.response_times[endpoint] = []
        self.response_times[endpoint].append(response_time)

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics summary."""
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)

        return {
            "uptime_seconds": uptime,
            "request_counts": self.request_counts,
            "error_counts": self.error_counts,
            "average_response_times": avg_response_times,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
        }


# Global metrics instance
api_metrics = APIMetrics()
