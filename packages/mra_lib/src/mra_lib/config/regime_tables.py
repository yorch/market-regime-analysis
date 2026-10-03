"""
Canonical regime lookup tables.

Single source of truth for the per-regime position-size multipliers, the
regime -> recommended-strategy map, and bars-per-year by timeframe. The
analyzer and the risk calculator both read from here so the tables cannot
drift apart.

Notes
-----
``MarketRegime.UNKNOWN`` has a multiplier of ``0.0``: an unclassified regime
maps to ``TradingStrategy.AVOID`` and therefore carries no position. (This
matches the backtesting strategy and calibrator, which already used ``0.0``.)
"""

from types import MappingProxyType
from typing import Final

from .enums import MarketRegime, TradingStrategy

REGIME_MULTIPLIERS: Final = MappingProxyType(
    {
        MarketRegime.BULL_TRENDING: 1.3,
        MarketRegime.BEAR_TRENDING: 0.7,
        MarketRegime.MEAN_REVERTING: 1.2,
        MarketRegime.HIGH_VOLATILITY: 0.4,
        MarketRegime.LOW_VOLATILITY: 1.1,
        MarketRegime.BREAKOUT: 0.9,
        MarketRegime.UNKNOWN: 0.0,
    }
)
"""Relative position-size multiplier per regime (read-only)."""

REGIME_STRATEGIES: Final = MappingProxyType(
    {
        MarketRegime.BULL_TRENDING: TradingStrategy.TREND_FOLLOWING,
        MarketRegime.BEAR_TRENDING: TradingStrategy.DEFENSIVE,
        MarketRegime.MEAN_REVERTING: TradingStrategy.MEAN_REVERSION,
        MarketRegime.HIGH_VOLATILITY: TradingStrategy.VOLATILITY_TRADING,
        MarketRegime.LOW_VOLATILITY: TradingStrategy.MOMENTUM,
        MarketRegime.BREAKOUT: TradingStrategy.MOMENTUM,
        MarketRegime.UNKNOWN: TradingStrategy.AVOID,
    }
)
"""Recommended trading strategy per regime (read-only)."""

TRADING_DAYS_PER_YEAR: Final[int] = 252

PERIODS_PER_YEAR: Final = MappingProxyType(
    {
        "1wk": 52.0,
        "1w": 52.0,
        "1d": float(TRADING_DAYS_PER_YEAR),
        "1h": TRADING_DAYS_PER_YEAR * 6.5,  # 6.5 regular-session hours per day
        "30m": TRADING_DAYS_PER_YEAR * 13.0,
        "15m": TRADING_DAYS_PER_YEAR * 26.0,  # 26 fifteen-minute bars per session
        "5m": TRADING_DAYS_PER_YEAR * 78.0,
        "1m": TRADING_DAYS_PER_YEAR * 390.0,
    }
)
"""Bars per year by (lower-cased) timeframe, assuming US regular trading hours."""


def get_regime_multiplier(regime: MarketRegime) -> float:
    """Return the canonical position multiplier for ``regime`` (0.0 if unmapped)."""
    return REGIME_MULTIPLIERS.get(regime, 0.0)


def get_regime_strategy(regime: MarketRegime) -> TradingStrategy:
    """Return the recommended strategy for ``regime`` (AVOID if unmapped)."""
    return REGIME_STRATEGIES.get(regime, TradingStrategy.AVOID)


def periods_per_year(timeframe: str) -> float:
    """
    Return the number of bars per year for ``timeframe``.

    Parameters
    ----------
    timeframe : str
        e.g. ``1D``, ``1H``, ``15m`` (case-insensitive; see :data:`PERIODS_PER_YEAR`).

    Raises
    ------
    ValueError
        If the timeframe is not recognised.
    """
    try:
        return PERIODS_PER_YEAR[timeframe.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; expected one of {sorted(PERIODS_PER_YEAR)}"
        ) from None
