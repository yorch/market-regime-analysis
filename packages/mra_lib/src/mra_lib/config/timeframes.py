"""
Analysis timeframes and their default data windows.

Single source of truth for the timeframes the analyzer, CLI, and API work with.
Every default period is supported by all registered providers (Yahoo Finance
only serves 15-minute bars for the last 60 days, so 15m uses ``1mo``).
"""

# Analysis timeframes, coarsest first
TIMEFRAMES: tuple[str, ...] = ("1D", "1H", "15m")

# Timeframe -> provider interval string
TIMEFRAME_INTERVALS: dict[str, str] = {"1D": "1d", "1H": "1h", "15m": "15m"}

# Timeframe -> default lookback period requested from the provider
DEFAULT_PERIODS: dict[str, str] = {"1D": "2y", "1H": "6mo", "15m": "1mo"}

# Approximate bars per regular US trading session, for converting days to bars
BARS_PER_DAY: dict[str, int] = {"1D": 1, "1H": 7, "15m": 26}
