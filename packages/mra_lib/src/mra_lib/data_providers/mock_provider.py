"""
Mock Data Provider

Example mock data provider for testing and demonstration purposes.
Shows how easy it is to add new providers to the plug-and-play system.
"""

from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import MarketDataProvider


class MockDataProvider(MarketDataProvider):
    """
    Example mock data provider for testing and demonstration.

    Shows how easy it is to add new providers to the system.
    To enable: register the provider in __init__.py
    """

    provider_name = "mock"
    supported_intervals: ClassVar[set[str]] = {"1d", "1h", "15m", "5m", "1m"}
    supported_periods: ClassVar[set[str]] = {"1d", "1mo", "1y", "2y"}
    requires_api_key = False
    rate_limit_per_minute = 0  # No limits for mock data
    description = "Mock data provider for testing and development"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Generate mock OHLCV data for testing."""
        self.validate_parameters(symbol, period, interval)

        # Create date range based on period and interval
        if period == "1d":
            periods = 96 if interval == "15m" else 24 if interval == "1h" else 1
        elif period == "1mo":
            periods = 30
        elif period == "1y":
            periods = 252
        elif period == "2y":
            periods = 504
        else:
            periods = 100

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=periods),
            periods=periods,
            freq="D" if interval == "1d" else "H" if interval == "1h" else "15min",
        )

        # Generate realistic-looking price data
        np.random.seed(42)  # For reproducible mock data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        # Create OHLCV data
        df = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        return self.standardize_dataframe(df)
