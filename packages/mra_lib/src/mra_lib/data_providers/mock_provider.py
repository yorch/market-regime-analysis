"""
Mock Data Provider

Offline provider that generates deterministic synthetic OHLCV data. Registered
as ``mock`` so the CLI, API, and tests can run without network access or keys::

    uv run mra current-analysis --provider mock --symbol SPY

Each symbol gets its own reproducible series (the seed is derived from the
symbol and ``ProviderConfig(seed=...)``, default 42). Bars are spaced on a
continuous calendar (no weekends or session gaps); the bar count approximates a
real market: 252 trading days per year, 7 hourly / 26 fifteen-minute bars per day.
"""

import zlib
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import PERIOD_DAYS, MarketDataProvider

_TRADING_DAYS_PER_YEAR = 252

# interval -> (pandas frequency, bars per trading day)
_INTERVALS: dict[str, tuple[str, int]] = {
    "1d": ("D", 1),
    "1h": ("h", 7),
    "30m": ("30min", 13),
    "15m": ("15min", 26),
    "5m": ("5min", 78),
    "1m": ("min", 390),
}


class MockDataProvider(MarketDataProvider):
    """Deterministic synthetic data provider for offline use and testing."""

    provider_name = "mock"
    supported_intervals: ClassVar[set[str]] = set(_INTERVALS)
    supported_periods: ClassVar[set[str]] = {*PERIOD_DAYS, "ytd"}
    requires_api_key = False
    rate_limit_per_minute = 0  # No limits for mock data
    description = "Offline synthetic data (deterministic per symbol) for testing and demos"

    def _seed(self, symbol: str) -> int:
        base = int(getattr(self.config, "seed", 42))
        return (base * 1_000_003 + zlib.crc32(symbol.upper().encode())) % (2**32)

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data for ``symbol``."""
        self.validate_parameters(symbol, period, interval)

        freq, bars_per_day = _INTERVALS[interval]
        days = 366 if period == "ytd" else PERIOD_DAYS[period]
        trading_days = max(1, round(days * _TRADING_DAYS_PER_YEAR / 365))
        n_bars = trading_days * bars_per_day

        end = pd.Timestamp(datetime.now(UTC)).tz_localize(None).floor(freq)
        dates = pd.date_range(end=end, periods=n_bars, freq=freq)

        # Local generator: never touches numpy's global random state
        rng = np.random.default_rng(self._seed(symbol))
        scale = 1.0 / np.sqrt(bars_per_day)
        base_price = 50.0 + rng.uniform(0, 150)
        returns = rng.normal(0.0004 * scale**2, 0.015 * scale, n_bars)
        close = base_price * np.exp(np.cumsum(returns))
        open_ = np.concatenate(([base_price], close[:-1])) * (1 + rng.normal(0, 0.001, n_bars))
        spread = np.abs(rng.normal(0, 0.004 * scale, n_bars))

        df = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close) * (1 + spread),
                "Low": np.minimum(open_, close) * (1 - spread),
                "Close": close,
                "Volume": rng.integers(1_000_000, 10_000_000, n_bars).astype(float),
            },
            index=dates,
        )

        return self.standardize_dataframe(df, interval)
