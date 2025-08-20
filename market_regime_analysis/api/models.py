"""
Pydantic request/response models for the Market Regime Analysis API.

This module defines all request and response models following the plan specifications.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..enums import MarketRegime


# Base models
class BaseRequest(BaseModel):
    """Base request model with common fields."""

    provider: str = Field(default="alphavantage", description="Data provider")
    api_key: str | None = Field(default=None, description="Provider API key")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider choice."""
        allowed_providers = ["yfinance", "alphavantage", "polygon"]
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {', '.join(allowed_providers)}")
        return v


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    error_code: str = Field(description="Error code identifier")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error timestamp"
    )


# Request models
class DetailedAnalysisRequest(BaseRequest):
    """Request model for detailed analysis endpoint."""

    symbol: str = Field(description="Trading symbol (e.g., SPY, QQQ)")
    timeframe: str = Field(description="Timeframe for analysis")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe choice."""
        allowed_timeframes = ["1D", "1H", "15m"]
        if v not in allowed_timeframes:
            raise ValueError(f"Timeframe must be one of: {', '.join(allowed_timeframes)}")
        return v


class CurrentAnalysisRequest(BaseRequest):
    """Request model for current analysis endpoint."""

    symbol: str = Field(description="Trading symbol (e.g., SPY, QQQ)")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()


class MultiSymbolAnalysisRequest(BaseRequest):
    """Request model for multi-symbol analysis endpoint."""

    symbols: list[str] = Field(description="List of trading symbols")
    timeframe: str = Field(description="Timeframe for analysis")

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate symbols list."""
        if not v or len(v) == 0:
            raise ValueError("At least one symbol must be provided")
        return [symbol.strip().upper() for symbol in v if symbol.strip()]

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe choice."""
        allowed_timeframes = ["1D", "1H", "15m"]
        if v not in allowed_timeframes:
            raise ValueError(f"Timeframe must be one of: {', '.join(allowed_timeframes)}")
        return v


class GenerateChartsRequest(BaseRequest):
    """Request model for chart generation endpoint."""

    symbol: str = Field(description="Trading symbol (e.g., SPY, QQQ)")
    timeframe: str = Field(description="Timeframe for analysis")
    days: int = Field(default=60, description="Number of days to plot")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe choice."""
        allowed_timeframes = ["1D", "1H", "15m"]
        if v not in allowed_timeframes:
            raise ValueError(f"Timeframe must be one of: {', '.join(allowed_timeframes)}")
        return v

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: int) -> int:
        """Validate days parameter."""
        if v <= 0:
            raise ValueError("Days must be a positive integer")
        if v > 365:
            raise ValueError("Days cannot exceed 365")
        return v


class ExportCSVRequest(BaseRequest):
    """Request model for CSV export endpoint."""

    symbol: str = Field(description="Trading symbol (e.g., SPY, QQQ)")
    filename: str | None = Field(default=None, description="Optional filename for export")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()


class PositionSizingRequest(BaseModel):
    """Request model for position sizing endpoint."""

    base_size: float = Field(description="Base position size (0.0-1.0)")
    regime: str = Field(description="Market regime")
    confidence: float = Field(description="Regime confidence (0.0-1.0)")
    persistence: float = Field(description="Regime persistence (0.0-1.0)")
    correlation: float = Field(default=0.0, description="Portfolio correlation (-1.0-1.0)")

    @field_validator("base_size")
    @classmethod
    def validate_base_size(cls, v: float) -> float:
        """Validate base size percentage."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Base size must be between 0.0 and 1.0")
        return v

    @field_validator("regime")
    @classmethod
    def validate_regime(cls, v: str) -> str:
        """Validate regime choice."""
        allowed_regimes = [r.value for r in MarketRegime]
        if v not in allowed_regimes:
            raise ValueError(f"Regime must be one of: {', '.join(allowed_regimes)}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence percentage."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @field_validator("persistence")
    @classmethod
    def validate_persistence(cls, v: float) -> float:
        """Validate persistence percentage."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Persistence must be between 0.0 and 1.0")
        return v

    @field_validator("correlation")
    @classmethod
    def validate_correlation(cls, v: float) -> float:
        """Validate correlation value."""
        if not -1.0 <= v <= 1.0:
            raise ValueError("Correlation must be between -1.0 and 1.0")
        return v


# Response models
class AnalysisResponse(BaseModel):
    """Response model for analysis endpoints."""

    symbol: str = Field(description="Trading symbol")
    timeframe: str = Field(description="Analysis timeframe")
    current_regime: str = Field(description="Detected market regime")
    regime_confidence: float = Field(description="Confidence in regime detection")
    regime_persistence: float = Field(description="Regime persistence score")
    transition_probability: float = Field(description="Regime transition probability")
    hmm_state: int = Field(description="HMM state (0-5)")
    risk_level: str = Field(description="Risk assessment level")
    position_sizing_multiplier: float = Field(description="Position sizing adjustment")
    recommended_strategy: str = Field(description="Recommended trading strategy")
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Analysis timestamp"
    )
    metrics: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")


class MultiAnalysisResponse(BaseModel):
    """Response model for multi-timeframe analysis."""

    symbol: str = Field(description="Trading symbol")
    analyses: list[AnalysisResponse] = Field(description="Analysis results by timeframe")
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Analysis timestamp"
    )


class PortfolioAnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""

    symbols: list[str] = Field(description="Analyzed symbols")
    timeframe: str = Field(description="Analysis timeframe")
    analyses: list[AnalysisResponse] = Field(description="Individual symbol analyses")
    portfolio_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Portfolio-wide metrics"
    )
    correlations: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Cross-asset correlations"
    )
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Analysis timestamp"
    )


class PositionSizingResponse(BaseModel):
    """Response model for position sizing endpoint."""

    base_size: float = Field(description="Input base position size")
    regime: str = Field(description="Market regime")
    regime_adjusted_size: float = Field(description="Regime-adjusted position size")
    correlation_adjusted_size: float = Field(description="Correlation-adjusted position size")
    final_recommendation: float = Field(description="Final position size recommendation")
    calculations: dict[str, Any] = Field(default_factory=dict, description="Detailed calculations")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Calculation timestamp"
    )


class ProviderInfo(BaseModel):
    """Provider information model."""

    name: str = Field(description="Provider name")
    description: str = Field(description="Provider description")
    requires_api_key: bool = Field(description="Whether API key is required")
    rate_limit_per_minute: int = Field(description="Rate limit per minute")
    supported_intervals: list[str] = Field(description="Supported timeframe intervals")
    supported_periods: list[str] = Field(description="Supported data periods")


class ProvidersResponse(BaseModel):
    """Response model for providers endpoint."""

    providers: dict[str, ProviderInfo] = Field(description="Available providers")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response timestamp"
    )


class ChartResponse(BaseModel):
    """Response model for chart generation."""

    symbol: str = Field(description="Trading symbol")
    timeframe: str = Field(description="Chart timeframe")
    days: int = Field(description="Days of data plotted")
    chart_data: dict[str, Any] | None = Field(default=None, description="Chart data (if requested)")
    file_path: str | None = Field(default=None, description="Generated chart file path")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Generation timestamp"
    )


class ExportResponse(BaseModel):
    """Response model for CSV export."""

    symbol: str = Field(description="Trading symbol")
    file_path: str = Field(description="Exported file path")
    filename: str = Field(description="Export filename")
    records_count: int = Field(description="Number of exported records")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Export timestamp"
    )


# WebSocket models
class MonitoringMessage(BaseModel):
    """WebSocket message model for monitoring."""

    message_type: str = Field(description="Message type (update, alert, error)")
    symbol: str = Field(description="Trading symbol")
    data: dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Message timestamp"
    )


class MonitoringUpdate(BaseModel):
    """Monitoring update data model."""

    symbol: str = Field(description="Trading symbol")
    current_regime: str = Field(description="Current market regime")
    regime_confidence: float = Field(description="Regime confidence")
    regime_change: bool = Field(description="Whether regime changed since last update")
    previous_regime: str | None = Field(default=None, description="Previous regime if changed")
    alert_level: str = Field(description="Alert level (low, medium, high)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Update timestamp"
    )
