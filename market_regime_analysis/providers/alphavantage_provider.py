"""
Alpha Vantage Data Provider

Professional data provider with API key authentication for real-time and historical data.
"""

from typing import ClassVar

import pandas as pd

from .base import MarketDataProvider, ProviderConfig


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage data provider with API key authentication."""

    provider_name = "alphavantage"
    supported_intervals: ClassVar[set[str]] = {
        # Alpha Vantage native formats
        "1min",
        "5min",
        "15min",
        "30min",
        "60min",
        "1d",
        "1wk",
        "1mo",
        # Common alternative formats that map to Alpha Vantage
        "1m",  # maps to 1min
        "5m",  # maps to 5min
        "15m",  # maps to 15min
        "30m",  # maps to 30min
        "1h",  # maps to 60min
        "1hour",  # maps to 60min
        "daily",  # maps to 1d
        "1day",  # maps to 1d
        "weekly",  # maps to 1wk
        "1week",  # maps to 1wk
        "monthly",  # maps to 1mo
        "1month",  # maps to 1mo
    }
    supported_periods: ClassVar[set[str]] = {
        # Alpha Vantage native formats
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "compact",
        "full",
        # Additional common period formats
        "2mo",  # 2 months (commonly used for 15m data)
        "4mo",  # 4 months
        "9mo",  # 9 months
        "10y",  # 10 years
        "max",  # Maximum available data
        "ytd",  # Year to date
    }
    requires_api_key = True
    rate_limit_per_minute = 5  # Free tier limitation
    description = "Professional Alpha Vantage API with real-time and historical data"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize Alpha Vantage provider with API key."""
        super().__init__(config)

        try:
            from alpha_vantage.timeseries import TimeSeries  # noqa: PLC0415

            self.ts = TimeSeries(key=self.config.api_key, output_format="pandas")
        except ImportError as e:
            raise ImportError(
                "alpha_vantage library is required. Install with: pip install alpha-vantage"
            ) from e

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        self.validate_parameters(symbol, period, interval)

        try:
            # Map common interval formats to Alpha Vantage format
            interval_mapping = {
                "1h": "60min",
                "1hour": "60min",
                "60min": "60min",
                "15m": "15min",
                "15min": "15min",
                "5m": "5min",
                "5min": "5min",
                "1m": "1min",
                "1min": "1min",
                "1d": "1d",
                "1day": "1d",
                "daily": "1d",
                "1w": "1wk",
                "1week": "1wk",
                "weekly": "1wk",
                "1mo": "1mo",
                "1month": "1mo",
                "monthly": "1mo",
            }

            # Normalize interval
            normalized_interval = interval_mapping.get(interval.lower(), interval)

            # Determine output size based on period
            # Use "compact" for short periods (less than 1 month), "full" for longer periods
            compact_periods = {"1d", "5d", "1mo"}
            outputsize = "compact" if period in compact_periods else "full"

            # Fetch data based on interval type
            if normalized_interval in ["1min", "5min", "15min", "30min", "60min"]:
                result = self.ts.get_intraday(
                    symbol=symbol, interval=normalized_interval, outputsize=outputsize
                )
                data = result[0]  # First element is the DataFrame
            elif normalized_interval == "1d":
                result = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
                data = result[0]  # First element is the DataFrame
            elif normalized_interval == "1wk":
                result = self.ts.get_weekly(symbol=symbol)
                data = result[0]  # First element is the DataFrame
            elif normalized_interval == "1mo":
                result = self.ts.get_monthly(symbol=symbol)
                data = result[0]  # First element is the DataFrame
            else:
                raise ValueError(f"Unsupported interval: {normalized_interval}")

            if data.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Normalize column names to match standard format
            column_mapping = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }

            # Rename columns if they exist (Alpha Vantage format)
            if any(col in data.columns for col in column_mapping):
                data = data.rename(columns=column_mapping)

            return self.standardize_dataframe(data)

        except Exception as e:
            raise ConnectionError(f"Failed to fetch data from Alpha Vantage: {e}") from e
