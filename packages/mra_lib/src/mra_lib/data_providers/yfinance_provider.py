"""
Yahoo Finance Data Provider

Free data provider using the yfinance library with comprehensive market coverage.

Yahoo only serves limited intraday history: sub-hourly bars for the last 60 days
and hourly bars for the last 730 days. Requests beyond those windows are rejected
up front with a ``ValueError`` instead of returning an empty frame.
"""

import warnings
from typing import ClassVar

import pandas as pd

from .base import (
    PERIOD_DAYS,
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    RateLimitError,
)

# Maximum lookback (days) Yahoo serves per intraday interval
_INTRADAY_MAX_DAYS: dict[str, int] = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "90m": 60,
    "60m": 730,
    "1h": 730,
}


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider using yfinance library."""

    provider_name = "yfinance"
    supported_intervals: ClassVar[set[str]] = {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
    supported_periods: ClassVar[set[str]] = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }
    requires_api_key = False
    rate_limit_per_minute = 60  # Conservative estimate
    description = "Free Yahoo Finance data provider with comprehensive market coverage"

    def validate_parameters(self, symbol: str, period: str, interval: str) -> None:
        """Validate parameters, including Yahoo's intraday history limits."""
        super().validate_parameters(symbol, period, interval)

        max_days = _INTRADAY_MAX_DAYS.get(interval)
        if max_days is None:
            return
        # 'ytd' can be up to 366 days
        days = 366 if period == "ytd" else PERIOD_DAYS.get(period, 0)
        if days > max_days:
            raise ValueError(
                f"Yahoo Finance only serves {interval} bars for the last {max_days} days; "
                f"period '{period}' is too long"
            )

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        self.validate_parameters(symbol, period, interval)

        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "yfinance library is required. Install with: pip install yfinance"
            ) from e

        self.throttle()
        try:
            ticker = yf.Ticker(symbol)
            # raise_errors: otherwise yfinance logs failures and returns an empty frame, which
            # would make network/rate-limit errors look like an unknown symbol
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = ticker.history(
                    period=period,
                    interval=interval,
                    timeout=self.config.timeout,
                    raise_errors=True,
                )
        except Exception as e:
            raise self._classify(symbol, e) from e

        if df is None or df.empty:
            raise InvalidSymbolError(
                f"No data returned for {symbol} (period={period}, interval={interval}); "
                "the symbol may be invalid or delisted"
            )

        return self.standardize_dataframe(df, interval)

    @staticmethod
    def _classify(symbol: str, error: Exception) -> Exception:
        """Map a yfinance exception onto the provider error hierarchy."""
        name = type(error).__name__
        if name == "YFRateLimitError":
            return RateLimitError(f"Yahoo Finance is rate limiting requests: {error}")
        if name in {"YFTickerMissingError", "YFPricesMissingError", "YFTzMissingError"}:
            return InvalidSymbolError(f"Yahoo Finance has no data for {symbol}: {error}")
        if name in {"YFInvalidPeriodError"} or isinstance(error, ValueError):
            return ValueError(f"Invalid request for {symbol}: {error}")
        if isinstance(error, PermissionError):
            return AuthError(f"Yahoo Finance rejected the request: {error}")
        return ConnectionError(f"Failed to fetch data from Yahoo Finance: {error}")
