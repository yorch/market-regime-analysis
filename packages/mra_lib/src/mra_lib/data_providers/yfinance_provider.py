"""
Yahoo Finance Data Provider

Free data provider using the yfinance library with comprehensive market coverage.
"""

from typing import ClassVar

import pandas as pd

from .base import MarketDataProvider


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

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        self.validate_parameters(symbol, period, interval)

        try:
            import yfinance as yf  # noqa: PLC0415

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            return self.standardize_dataframe(df)

        except ImportError as e:
            raise ImportError(
                "yfinance library is required. Install with: pip install yfinance"
            ) from e
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data from Yahoo Finance: {e}") from e
