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
