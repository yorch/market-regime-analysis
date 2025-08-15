"""
Example New Provider

This file demonstrates how easy it is to add a new data provider to the system.
Simply create a new file, inherit from MarketDataProvider, and register it.
"""

from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import MarketDataProvider


class ExampleNewProvider(MarketDataProvider):
    """
    Example showing how to add a new provider.

    To enable this provider:
    1. Add import to __init__.py: from .example_new_provider import ExampleNewProvider
    2. Add registration: MarketDataProvider.register(ExampleNewProvider)
    3. The provider will automatically appear in list-providers and be available via CLI
    """

    provider_name = "example"
    supported_intervals: ClassVar[set[str]] = {"1d", "1h", "15m"}
    supported_periods: ClassVar[set[str]] = {"1mo", "1y", "2y"}
    requires_api_key = False  # Set to True if API key needed
    rate_limit_per_minute = 100
    description = "Example data provider showing how easy it is to extend the system"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Implement your data fetching logic here.

        This is the only method you need to implement!
        The base class handles all validation, standardization, and registration.
        """
        self.validate_parameters(symbol, period, interval)

        # Your data fetching logic goes here
        # For example, you might:
        # 1. Make HTTP requests to your API
        # 2. Read from a database
        # 3. Load from files
        # 4. Transform data from another format

        # Mock implementation (replace with real data fetching):

        # Generate sample data (replace with real implementation)
        periods_count = 252 if period == "1y" else 30
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=periods_count), periods=periods_count, freq="D"
        )

        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.02,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(1000000, 5000000, len(dates)),
            },
            index=dates,
        )

        # The base class will handle standardization
        return self.standardize_dataframe(df)


# Note: This provider is not auto-registered.
# To enable it, uncomment the import and registration in __init__.py
