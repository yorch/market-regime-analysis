"""
Alpha Vantage Data Provider

Calls the Alpha Vantage REST API (https://www.alphavantage.co/documentation/)
directly through the shared ``_http.get_json`` helper, so requests get timeouts,
retries, and client-side rate limiting.

Price adjustment
----------------
- Intraday bars are split/dividend adjusted (``adjusted=true``) and limited to the
  regular session (``extended_hours=false``). Alpha Vantage only returns the most
  recent ~30 days of intraday data per request, so longer periods are truncated.
- Weekly and monthly bars use the free ``*_ADJUSTED`` endpoints.
- Daily bars: ``TIME_SERIES_DAILY_ADJUSTED`` is a premium endpoint. Free keys get
  **unadjusted** daily prices (``TIME_SERIES_DAILY``), so splits appear as large
  one-day moves; a warning is logged. Premium users should pass
  ``ProviderConfig(premium=True)`` or set ``ALPHA_VANTAGE_PREMIUM=1`` to get
  adjusted daily prices. If the full daily history is refused on the free tier,
  the provider falls back to the latest ~100 bars.

Results are trimmed to the requested ``period``.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any, ClassVar

import pandas as pd

from ._http import get_json
from .base import (
    PERIOD_DAYS,
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    ProviderConfig,
    RateLimitError,
    period_to_start,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
PREMIUM_ENV = "ALPHA_VANTAGE_PREMIUM"

# Alpha Vantage "compact" responses hold the latest 100 data points
_COMPACT_POINTS = 100

# The unadjusted-prices warning is logged once per process
_warned_unadjusted = False


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage data provider with API key authentication."""

    provider_name = "alphavantage"

    # Accepted interval spellings -> canonical Alpha Vantage interval
    _INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "1m": "1min",
        "1min": "1min",
        "5m": "5min",
        "5min": "5min",
        "15m": "15min",
        "15min": "15min",
        "30m": "30min",
        "30min": "30min",
        "1h": "60min",
        "1hour": "60min",
        "60min": "60min",
        "1d": "daily",
        "1day": "daily",
        "daily": "daily",
        "1w": "weekly",
        "1wk": "weekly",
        "1week": "weekly",
        "weekly": "weekly",
        "1mo": "monthly",
        "1month": "monthly",
        "monthly": "monthly",
    }
    # Approximate bars per trading day, used to pick compact vs full output size
    _BARS_PER_DAY: ClassVar[dict[str, int]] = {
        "1min": 390,
        "5min": 78,
        "15min": 26,
        "30min": 13,
        "60min": 7,
        "daily": 1,
    }

    supported_intervals: ClassVar[set[str]] = set(_INTERVAL_MAP)
    supported_periods: ClassVar[set[str]] = {*PERIOD_DAYS, "ytd", "compact", "full"}
    requires_api_key = True
    rate_limit_per_minute = 5  # Free tier: 5 requests/minute, 25/day
    description = "Alpha Vantage API (free tier: unadjusted daily prices, 25 requests/day)"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize Alpha Vantage provider."""
        super().__init__(config)
        premium = getattr(self.config, "premium", None)
        if premium is None:
            premium = os.getenv(PREMIUM_ENV, "").lower() in {"1", "true", "yes"}
        self.premium: bool = bool(premium)

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage.

        Args:
            symbol: Trading symbol (e.g., 'SPY')
            period: Period string (e.g. '2y'), or Alpha Vantage's 'compact'/'full'
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame following the ``base`` contract

        Raises:
            InvalidSymbolError: Unknown symbol or empty result
            AuthError: Missing/invalid key or premium-only request
            RateLimitError: Per-minute or daily quota exhausted
            ConnectionError: Network or server failure
        """
        interval = interval.lower()
        self.validate_parameters(symbol, period, interval)
        av_interval = self._INTERVAL_MAP[interval]

        outputsize = self._outputsize(period, av_interval)
        params: dict[str, Any] = {"symbol": symbol.upper(), "datatype": "json"}

        if av_interval in self._BARS_PER_DAY and av_interval != "daily":
            params.update(
                function="TIME_SERIES_INTRADAY",
                interval=av_interval,
                adjusted="true",
                extended_hours="false",
                outputsize=outputsize,
            )
        elif av_interval == "daily":
            if self.premium:
                params["function"] = "TIME_SERIES_DAILY_ADJUSTED"
            else:
                params["function"] = "TIME_SERIES_DAILY"
                global _warned_unadjusted  # noqa: PLW0603
                if not _warned_unadjusted:
                    _warned_unadjusted = True
                    logger.warning(
                        "Alpha Vantage free tier returns unadjusted daily prices; splits and "
                        "dividends are not reflected. Set %s=1 with a premium key for "
                        "adjusted data.",
                        PREMIUM_ENV,
                    )
            params["outputsize"] = outputsize
        elif av_interval == "weekly":
            params["function"] = "TIME_SERIES_WEEKLY_ADJUSTED"
        else:
            params["function"] = "TIME_SERIES_MONTHLY_ADJUSTED"

        payload = self._request(params, symbol)
        df = self._payload_to_dataframe(payload, intraday=params["function"].endswith("INTRADAY"))
        df = self.standardize_dataframe(df, interval)

        if period in PERIOD_DAYS or period == "ytd":
            start = period_to_start(period, datetime.now(UTC)).replace(tzinfo=None)
            df = df[df.index >= start.replace(hour=0, minute=0, second=0, microsecond=0)]
        if df.empty:
            raise InvalidSymbolError(f"No data returned for {symbol} in period {period}")
        return df

    def _outputsize(self, period: str, av_interval: str) -> str:
        if period in {"compact", "full"}:
            return period
        bars_per_day = self._BARS_PER_DAY.get(av_interval)
        if bars_per_day is None:
            return "full"
        days = 366 if period == "ytd" else PERIOD_DAYS[period]
        expected_bars = days * 252 / 365 * bars_per_day
        return "compact" if expected_bars <= _COMPACT_POINTS else "full"

    def _request(self, params: dict[str, Any], symbol: str) -> dict[str, Any]:
        """Run a query, falling back to compact output if the full history is premium-only."""
        payload = self._get(params)
        message = self._message(payload)
        if (
            message
            and "premium" in message.lower()
            and not self._is_rate_limit(message)
            and params.get("outputsize") == "full"
        ):
            logger.warning(
                "Alpha Vantage refused full history on this plan; using the latest %d bars",
                _COMPACT_POINTS,
            )
            payload = self._get({**params, "outputsize": "compact"})
            message = self._message(payload)
        if message:
            raise self._classify(message, symbol)
        return payload

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        payload = get_json(
            BASE_URL,
            provider="Alpha Vantage",
            config=self.config,
            params={**params, "apikey": self.config.api_key},
            throttle=self.throttle,
        )
        if not isinstance(payload, dict):
            raise ConnectionError("Alpha Vantage returned an unexpected response")
        return payload

    @staticmethod
    def _message(payload: dict[str, Any]) -> str | None:
        """Return Alpha Vantage's error/notice text if the payload has no time series."""
        if any("Time Series" in key for key in payload):
            return None
        for key in ("Error Message", "Note", "Information"):
            if payload.get(key):
                return str(payload[key])
        return "Alpha Vantage returned no time series"

    @staticmethod
    def _is_rate_limit(message: str) -> bool:
        lowered = message.lower()
        return any(s in lowered for s in ("frequency", "rate limit", "requests per day"))

    @classmethod
    def _classify(cls, message: str, symbol: str) -> Exception:
        lowered = message.lower()
        # Rate-limit notices also mention premium plans, so check them first
        if cls._is_rate_limit(message):
            return RateLimitError(f"Alpha Vantage rate limit reached: {message}")
        if "apikey" in lowered or "api key" in lowered:
            return AuthError(f"Alpha Vantage rejected the API key: {message}")
        if "premium" in lowered:
            return AuthError(f"Alpha Vantage premium plan required: {message}")
        if "invalid api call" in lowered or "no time series" in lowered:
            return InvalidSymbolError(f"Alpha Vantage has no data for {symbol}: {message}")
        return ConnectionError(f"Alpha Vantage error for {symbol}: {message}")

    @staticmethod
    def _payload_to_dataframe(payload: dict[str, Any], *, intraday: bool) -> pd.DataFrame:
        """Convert an Alpha Vantage time-series payload into an OHLCV DataFrame."""
        series_key = next(key for key in payload if "Time Series" in key)
        series: dict[str, dict[str, str]] = payload[series_key]
        if not series:
            raise InvalidSymbolError("Alpha Vantage returned an empty time series")

        def field(row: dict[str, str], name: str) -> float:
            # Field names are numbered ("1. open"); the numbering differs per endpoint
            for key, value in row.items():
                if key.split(". ", 1)[-1] == name:
                    return float(value)
            raise KeyError(name)

        try:
            rows = []
            for row in series.values():
                close = field(row, "close")
                try:
                    adj_close = field(row, "adjusted close")
                except KeyError:
                    adj_close = close
                # Scale OHLC by the adjustment factor so the bar stays consistent
                factor = adj_close / close if close else 1.0
                rows.append(
                    {
                        "Open": field(row, "open") * factor,
                        "High": field(row, "high") * factor,
                        "Low": field(row, "low") * factor,
                        "Close": adj_close,
                        "Volume": field(row, "volume"),
                    }
                )
            index = pd.to_datetime(list(series.keys()))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Malformed price data from Alpha Vantage: {e}") from e

        if intraday:
            # Intraday timestamps are US/Eastern wall-clock times
            tz = payload.get("Meta Data", {}).get("6. Time Zone", "US/Eastern")
            index = index.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")

        df = pd.DataFrame(rows, index=index)
        valid: pd.DataFrame = df[df.index.notna()]
        return valid
