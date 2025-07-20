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
        if interval in ["1min", "5min", "15min", "30min", "60min"]:
            # Intraday
            outputsize = "full" if period not in ["1d", "5d", "1mo"] else "compact"
            data, _ = self.ts.get_intraday(
                symbol=symbol, interval=interval, outputsize=outputsize
            )
        elif interval == "1d":
            outputsize = "full" if period not in ["1d", "5d", "1mo"] else "compact"
            data, _ = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
        elif interval == "1wk":
            data, _ = self.ts.get_weekly(symbol=symbol)
        elif interval == "1mo":
            data, _ = self.ts.get_monthly(symbol=symbol)
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        return data
