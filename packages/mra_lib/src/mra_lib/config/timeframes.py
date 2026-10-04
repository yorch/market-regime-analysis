"""
Analysis timeframes and their default data windows.

Single source of truth for the timeframes the analyzer, CLI, and API work with.
Every default period is supported by all registered providers (Yahoo Finance
only serves 15-minute bars for the last 60 days, so 15m uses ``1mo``).
"""

import math
from types import MappingProxyType
from typing import Final

from mra_lib.config.regime_tables import TRADING_DAYS_PER_YEAR, periods_per_year

# Analysis timeframes, coarsest first
TIMEFRAMES: tuple[str, ...] = ("1D", "1H", "15m")

# Timeframe -> provider interval string
TIMEFRAME_INTERVALS: dict[str, str] = {"1D": "1d", "1H": "1h", "15m": "15m"}

# Timeframe -> default lookback period requested from the provider
DEFAULT_PERIODS: dict[str, str] = {"1D": "2y", "1H": "6mo", "15m": "1mo"}

# Bars per regular US trading session (rounded up: 09:30-16:00 has 7 hourly bars), for
# converting days to bars. Derived from regime_tables.PERIODS_PER_YEAR.
BARS_PER_DAY: dict[str, int] = {
    tf: math.ceil(periods_per_year(tf) / TRADING_DAYS_PER_YEAR) for tf in TIMEFRAMES
}

# Multi-timeframe confirmation defaults (see mra_lib.signals.confirmation).
CONFIRMATION_WEIGHTS: Final = MappingProxyType({"1D": 0.5, "1H": 0.3, "15m": 0.2})
"""Per-timeframe weight in the agreement score (read-only); higher timeframes dominate.

1D (0.5) weighs as much as 1H and 15m combined. With the default threshold, 1D alone
(at most 0.5) never confirms, so a 1D signal needs a lower timeframe to agree; and
1H + 15m without 1D (at most 0.5) never confirm either. Weights need not sum to 1:
agreement is normalized by their total."""

CONFIRMATION_THRESHOLD: Final[float] = 0.6
"""Minimum agreement score (0-1) for a directional signal to count as confirmed."""

MIN_CONFIRMATION_TIMEFRAMES: Final[int] = 2
"""Minimum number of classified, aligned timeframes for a confirmation."""
