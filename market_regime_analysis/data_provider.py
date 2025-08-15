"""
Data Provider Interface and YFinance Implementation
"""

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch market data for a symbol, period, and interval.
        Returns a DataFrame with OHLCV columns.
        """
        pass


class YFinanceProvider(MarketDataProvider):
    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df


class AlphaVantageProvider(MarketDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        from alpha_vantage.timeseries import TimeSeries

        self.ts = TimeSeries(key=api_key, output_format="pandas")

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        # Map yfinance period/interval to Alpha Vantage parameters
        # Alpha Vantage supports '1min', '5min', '15min', '30min', '60min' for intraday
        # For daily, weekly, monthly use get_daily, get_weekly, get_monthly

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
        interval = interval_mapping.get(interval.lower(), interval)

        if interval in ["1min", "5min", "15min", "30min", "60min"]:
            # Intraday
            outputsize = "full" if period not in ["1d", "5d", "1mo"] else "compact"
            data, _ = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
        elif interval == "1d":
            outputsize = "full" if period not in ["1d", "5d", "1mo"] else "compact"
            data, _ = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
        elif interval == "1wk":
            data, _ = self.ts.get_weekly(symbol=symbol)
        elif interval == "1mo":
            data, _ = self.ts.get_monthly(symbol=symbol)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Normalize column names to match yfinance format
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

        return data
