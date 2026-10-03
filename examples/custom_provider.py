#!/usr/bin/env python3
"""
Adding a custom market data provider.

Subclass ``MarketDataProvider``, implement ``fetch()``, and register the class.
It then works everywhere a provider name is accepted (``MarketRegimeAnalyzer``,
``create_provider``). The CLI's ``--provider`` choices are built from the registry
at import time, so for the CLI register your provider in
``mra_lib/data_providers/__init__.py``.

Run with::

    uv run examples/custom_provider.py
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from mra_lib import MarketRegimeAnalyzer
from mra_lib.data_providers import MarketDataProvider, register_provider
from mra_lib.data_providers.base import PERIOD_DAYS


class ExampleNewProvider(MarketDataProvider):
    """Toy provider returning a random walk; replace ``fetch`` with real I/O."""

    provider_name = "example"
    supported_intervals: ClassVar[set[str]] = {"1d"}
    supported_periods: ClassVar[set[str]] = {"1mo", "6mo", "1y", "2y"}
    requires_api_key = False  # Set to True if an API key is needed
    rate_limit_per_minute = 100  # Drives the built-in client-side rate limiter
    description = "Example data provider showing how to extend the system"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Return OHLCV bars; the base class validates inputs and standardizes output."""
        self.validate_parameters(symbol, period, interval)
        self.throttle()  # Call before every remote request

        # Replace this block with an HTTP call, database query, file read, ...
        n_bars = round(PERIOD_DAYS[period] * 252 / 365)
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_bars)
        rng = np.random.default_rng(7)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n_bars)))
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, n_bars),
            },
            index=dates,
        )

        # Float columns, sorted/de-duplicated tz-naive index (see base module docs)
        return self.standardize_dataframe(df, interval)


def main() -> None:
    register_provider(ExampleNewProvider)

    analyzer = MarketRegimeAnalyzer("DEMO", periods={"1D": "2y"}, provider_flag="example")
    analysis = analyzer.analyze_current_regime("1D")
    print(f"\nDEMO regime via custom provider: {analysis.current_regime.value}")


if __name__ == "__main__":
    main()
