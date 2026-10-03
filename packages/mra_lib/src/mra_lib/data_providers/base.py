"""
Base Provider Interface

Core interfaces and utilities for market data providers in the plug-and-play architecture.

DataFrame contract
------------------
Every provider's ``fetch()`` returns a DataFrame that:

- has exactly the float64 columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
- has a sorted, de-duplicated, **tz-naive** ``DatetimeIndex``
- labels intraday bars (``1h``, ``15m``, ...) by their **UTC** open time
- labels daily-and-coarser bars by the exchange session date at midnight
- by default keeps the most recent bar even if it is still in progress; pass
  ``ProviderConfig(drop_incomplete_bar=True)`` to drop it

Errors
------
Providers raise the exception classes defined in :mod:`mra_lib.errors` and
re-exported here (``ProviderError`` and its subclasses ``InvalidSymbolError``,
``AuthError``, ``RateLimitError``). They subclass ``ValueError`` or
``ConnectionError`` so existing ``except ValueError`` / ``except ConnectionError``
handlers keep working, and :class:`~mra_lib.errors.MRAError` so ``except MRAError``
catches every library failure.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import pandas as pd

# Re-exported: providers and callers import the error classes from here
from mra_lib.errors import (  # noqa: F401
    AuthError,
    InvalidSymbolError,
    ProviderError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Lookback in days for the standard period strings shared across providers
PERIOD_DAYS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "2mo": 60,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "max": 7300,  # ~20 years
}

# Bar duration for intraday interval spellings used across providers
_INTRADAY_DURATIONS: dict[str, timedelta] = {
    **dict.fromkeys(("1m", "1min", "minute"), timedelta(minutes=1)),
    "2m": timedelta(minutes=2),
    **dict.fromkeys(("5m", "5min"), timedelta(minutes=5)),
    **dict.fromkeys(("15m", "15min"), timedelta(minutes=15)),
    **dict.fromkeys(("30m", "30min"), timedelta(minutes=30)),
    **dict.fromkeys(("1h", "1hour", "60m", "60min", "hour"), timedelta(hours=1)),
    "90m": timedelta(minutes=90),
}

_DAILY_INTERVALS = {"1d", "1day", "daily", "day"}

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

_MARKET_TZ = "America/New_York"
_MARKET_CLOSE_HOUR = 16


def period_to_start(period: str, end: datetime | None = None) -> datetime:
    """
    Convert a period string (e.g. '1y', '6mo', 'ytd') into a start datetime.

    Args:
        period: Period string; one of ``PERIOD_DAYS`` keys or ``'ytd'``
        end: End of the range (defaults to now, UTC)

    Returns:
        Start datetime with the same timezone as ``end``

    Raises:
        ValueError: If the period is not recognized
    """
    end = end or datetime.now(UTC)
    if period == "ytd":
        return end.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if period not in PERIOD_DAYS:
        raise ValueError(f"Unsupported period '{period}'. Supported: {sorted(PERIOD_DAYS)}")
    return end - timedelta(days=PERIOD_DAYS[period])


def intraday_duration(interval: str) -> timedelta | None:
    """Return the bar length for an intraday interval, or ``None`` for daily and coarser."""
    return _INTRADAY_DURATIONS.get(interval.lower())


class ProviderConfig:
    """Configuration container for data provider settings.

    Attributes:
        api_key: Provider API key (if required)
        timeout: Per-request timeout in seconds
        retries: Retries for transient failures (rate limits, 5xx, network errors)
        rate_limit: Requests per minute for the client-side limiter; ``0`` uses the
            provider's ``rate_limit_per_minute`` default, a negative value disables it
        drop_incomplete_bar: Drop the last bar if it is still in progress
    """

    def __init__(self, **kwargs: Any) -> None:
        self.api_key: str | None = kwargs.get("api_key")
        self.timeout: int = kwargs.get("timeout", 30)
        self.retries: int = kwargs.get("retries", 3)
        self.rate_limit: float = kwargs.get("rate_limit", 0.0)
        self.drop_incomplete_bar: bool = bool(kwargs.get("drop_incomplete_bar", False))

        # Store any additional provider-specific config
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class TokenBucket:
    """Thread-safe token bucket: ``capacity`` burst, refilled at ``rate_per_minute``."""

    def __init__(self, rate_per_minute: float, capacity: float | None = None) -> None:
        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = max(1.0, float(capacity if capacity is not None else rate_per_minute))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Take one token, sleeping until one is available. Returns seconds waited."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.capacity, self._tokens + (now - self._updated) * self.rate_per_second
            )
            self._updated = now
            self._tokens -= 1.0
            wait = -self._tokens / self.rate_per_second if self._tokens < 0 else 0.0
        if wait > 0:
            time.sleep(wait)
        return wait


class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    This class defines the interface that all market data providers must implement
    and provides common functionality for registration and validation.
    """

    # Registry for all available providers
    _providers: ClassVar[dict[str, type["MarketDataProvider"]]] = {}

    # Provider metadata
    provider_name: str = ""
    supported_intervals: ClassVar[set[str]] = set()
    supported_periods: ClassVar[set[str]] = set()
    requires_api_key: bool = False
    rate_limit_per_minute: int = 0
    # Requests allowed back-to-back before the limiter starts spacing them out
    # (defaults to ``rate_limit_per_minute``)
    rate_limit_burst: int | None = None
    description: str = ""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize provider with configuration."""
        self.config = config or ProviderConfig()
        self._validate_config()

        rate = self.config.rate_limit or self.rate_limit_per_minute
        self._limiter: TokenBucket | None = (
            self._shared_limiter(rate) if rate and rate > 0 else None
        )

    # Buckets shared by every instance using the same provider, key, and rate, so separate
    # analyzers (threads, portfolio symbols, web requests) draw from one quota
    _limiters: ClassVar[dict[tuple[str, str, float, int | None], TokenBucket]] = {}
    _limiters_lock: ClassVar[threading.Lock] = threading.Lock()

    def _shared_limiter(self, rate: float) -> TokenBucket:
        key = (self.provider_name, self.config.api_key or "", float(rate), self.rate_limit_burst)
        with MarketDataProvider._limiters_lock:
            bucket = MarketDataProvider._limiters.get(key)
            if bucket is None:
                bucket = TokenBucket(rate, self.rate_limit_burst)
                MarketDataProvider._limiters[key] = bucket
            return bucket

    @classmethod
    def reset_rate_limiters(cls) -> None:
        """Forget all shared rate-limiter state (mainly for tests)."""
        with MarketDataProvider._limiters_lock:
            MarketDataProvider._limiters.clear()

    @abstractmethod
    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch market data for a symbol, period, and interval.

        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            period: Time period for data (e.g., '1y', '2y', '6mo')
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame following the module-level contract (float OHLCV columns,
            sorted tz-naive DatetimeIndex)

        Raises:
            ValueError: If parameters are invalid (``InvalidSymbolError`` for unknown
                symbols / empty results)
            ConnectionError: If the API is unavailable (``AuthError`` for rejected
                credentials, ``RateLimitError`` when throttled)
        """

    @classmethod
    def register(cls, provider_class: type["MarketDataProvider"]) -> None:
        """Register a provider class in the global registry."""
        if not provider_class.provider_name:
            raise ValueError(f"Provider {provider_class.__name__} must define provider_name")

        cls._providers[provider_class.provider_name] = provider_class

    @classmethod
    def get_available_providers(cls) -> dict[str, dict[str, Any]]:
        """Get information about all available providers."""
        return {
            name: {
                "class": provider_class,
                "description": provider_class.description,
                "requires_api_key": provider_class.requires_api_key,
                "supported_intervals": list(provider_class.supported_intervals),
                "supported_periods": list(provider_class.supported_periods),
                "rate_limit_per_minute": provider_class.rate_limit_per_minute,
            }
            for name, provider_class in cls._providers.items()
        }

    @classmethod
    def create_provider(cls, provider_name: str, **config_kwargs: Any) -> "MarketDataProvider":
        """
        Factory method to create a provider instance.

        Args:
            provider_name: Name of the provider to create
            **config_kwargs: Configuration parameters for the provider

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider is not registered
        """
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Provider '{provider_name}' not found. Available: {available}")

        provider_class = cls._providers[provider_name]
        config = ProviderConfig(**config_kwargs)
        return provider_class(config)

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        if self.requires_api_key and not self.config.api_key:
            raise ValueError(f"Provider '{self.provider_name}' requires an API key")

    def throttle(self) -> None:
        """Block until the client-side rate limiter allows another request."""
        limiter = getattr(self, "_limiter", None)
        if limiter is not None:
            waited = limiter.acquire()
            if waited > 1:
                logger.info("%s: rate limiter waited %.1fs", self.provider_name, waited)

    def validate_parameters(self, symbol: str, period: str, interval: str) -> None:
        """
        Validate input parameters against provider capabilities.

        Args:
            symbol: Trading symbol
            period: Time period
            interval: Data interval

        Raises:
            ValueError: If parameters are not supported
        """
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string")

        if self.supported_intervals and interval not in self.supported_intervals:
            raise ValueError(
                f"Interval '{interval}' not supported by {self.provider_name}. "
                f"Supported: {sorted(self.supported_intervals)}"
            )

        if self.supported_periods and period not in self.supported_periods:
            raise ValueError(
                f"Period '{period}' not supported by {self.provider_name}. "
                f"Supported: {sorted(self.supported_periods)}"
            )

    def standardize_dataframe(self, df: pd.DataFrame, interval: str | None = None) -> pd.DataFrame:
        """
        Standardize a provider DataFrame to the module-level contract.

        The input is never modified. Tz-aware intraday indexes are converted to
        naive UTC; daily-and-coarser bars are normalized to midnight of their
        session date. Duplicate timestamps keep the last row.

        Args:
            df: Raw DataFrame from provider
            interval: Requested interval; enables daily-date normalization and the
                optional in-progress bar drop. When omitted, tz-aware indexes are
                converted to naive UTC.

        Returns:
            Standardized DataFrame with float64 OHLCV columns
        """
        missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required column: {missing[0]}")

        out = df[OHLCV_COLUMNS].copy()
        index = pd.DatetimeIndex(out.index)
        is_daily = interval is not None and intraday_duration(interval) is None

        if index.tz is not None:
            # Daily bars keep the exchange-local session date; intraday bars go to UTC
            index = index.tz_localize(None) if is_daily else index.tz_convert(UTC).tz_localize(None)
        if is_daily:
            index = index.normalize()  # type: ignore[attr-defined]
        out.index = index

        out = out.apply(pd.to_numeric, errors="coerce").astype("float64")
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.dropna(how="all")
        result: pd.DataFrame = out

        if interval is not None and self.config.drop_incomplete_bar:
            result = drop_in_progress_bar(result, interval)

        return result


def drop_in_progress_bar(
    df: pd.DataFrame, interval: str, now: datetime | None = None
) -> pd.DataFrame:
    """
    Drop the final bar if it has not closed yet.

    Intraday bars are complete once ``open + duration`` has passed (index is naive
    UTC). Daily bars are complete once 16:00 New York time has passed on their
    session date. Weekly and monthly bars are left untouched.

    Args:
        df: Standardized DataFrame
        interval: Bar interval
        now: Current time (defaults to now; naive values are treated as UTC)
    """
    if df.empty:
        return df
    now = now or datetime.now(UTC)
    now_utc = now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)
    last: datetime = pd.DatetimeIndex(df.index)[-1].to_pydatetime()
    duration = intraday_duration(interval)

    if duration is not None:
        in_progress = last + duration > now_utc.replace(tzinfo=None)
    elif interval.lower() in _DAILY_INTERVALS:
        local_now = now_utc.astimezone(ZoneInfo(_MARKET_TZ))
        today = local_now.date()
        in_progress = last.date() > today or (
            last.date() == today and local_now.hour < _MARKET_CLOSE_HOUR
        )
    else:
        in_progress = False

    return df.iloc[:-1] if in_progress else df
