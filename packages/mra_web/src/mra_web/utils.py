"""
Utility functions for the Market Regime Analysis API.

This module provides helper functions and error handling utilities.
"""

import asyncio
import json
import logging
import math
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.data_providers import required_env_vars, resolve_api_key

from .config import get_config
from .models import AnalysisResponse, ErrorResponse

# Setup logging
logger = logging.getLogger(__name__)

T = TypeVar("T")

# Timeframes the API analyzes and the history loaded for each. Endpoints pass only
# the timeframe they need, so a single-timeframe request loads a single dataset.
TIMEFRAMES: tuple[str, ...] = ("1D", "1H", "15m")
DEFAULT_PERIODS: dict[str, str] = {"1D": "2y", "1H": "6mo", "15m": "1mo"}


def periods_for(*timeframes: str) -> dict[str, str]:
    """Return the analyzer ``periods`` mapping for the given timeframes."""
    return {tf: DEFAULT_PERIODS[tf] for tf in timeframes}


def to_jsonable(obj: Any) -> Any:  # noqa: PLR0911
    """Convert numpy/pandas/datetime values into strict-JSON-safe Python values.

    Non-finite floats (NaN, +/-Inf) and ``NaT`` become ``None``.
    """
    if obj is None or isinstance(obj, str | bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp | datetime | date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray | pd.Series | pd.Index):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for numpy, pandas and datetime objects."""

    def default(self, o):
        converted = to_jsonable(o)
        if converted is o:
            return super().default(o)
        return converted


class NumpyJSONResponse(JSONResponse):
    """JSONResponse that emits strict JSON (NaN/Inf -> null) and handles numpy types."""

    def render(self, content) -> bytes:
        return json.dumps(
            to_jsonable(content),
            cls=NumpyJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


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
    raise HTTPException(status_code=status_code, detail=error_response.model_dump(mode="json"))


def convert_numpy_types(obj: Any) -> Any:  # noqa: PLR0911
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
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
    resolved = resolve_api_key(provider, api_key)
    if resolved is not None:
        return resolved

    provider_name = provider.replace("_", " ").title()
    raise HTTPException(
        status_code=400,
        detail=create_error_response(
            "API_KEY_REQUIRED",
            f"{provider_name} API key is required when using {provider} provider.",
            {
                "provider": provider,
                "required_env_vars": required_env_vars(provider),
            },
        ).model_dump(mode="json"),
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
        metrics=to_jsonable(
            {
                "arbitrage_opportunities": analysis.arbitrage_opportunities,
                "statistical_signals": analysis.statistical_signals,
                "key_levels": analysis.key_levels,
                "hmm_state": analysis.hmm_state,
                "transition_probability": analysis.transition_probability,
            }
        ),
    )


async def run_in_thread(
    func: Callable[..., T], *args: Any, _timeout: float | None = None, **kwargs: Any
) -> T:
    """Run a blocking function in Starlette's thread pool with a timeout.

    Args:
        func: Blocking callable
        *args, **kwargs: Passed to ``func``
        _timeout: Seconds to wait (default ``API_TIMEOUT``)

    Raises:
        HTTPException: 504 if the call does not finish in time. The worker thread
            cannot be cancelled and finishes in the background.
    """
    timeout = get_config().timeout if _timeout is None else _timeout
    try:
        async with asyncio.timeout(timeout):
            return await run_in_threadpool(func, *args, **kwargs)
    except TimeoutError as e:
        logger.warning(
            "Blocking call %s timed out after %ss", getattr(func, "__name__", func), timeout
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Analysis timed out"
        ) from e


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

    # Response-time samples kept per endpoint (bounded memory).
    MAX_SAMPLES = 1000

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.request_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self.response_times: dict[str, deque[float]] = {}
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
            self.response_times[endpoint] = deque(maxlen=self.MAX_SAMPLES)
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
