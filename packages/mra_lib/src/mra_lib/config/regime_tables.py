"""
Canonical regime lookup tables.

Single source of truth for the per-regime position-size multipliers, the
regime -> recommended-strategy map, the regime -> directional-bias map used by
multi-timeframe confirmation, and bars-per-year by timeframe. The
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

from .enums import DirectionalBias, MarketRegime, TradingStrategy

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

REGIME_BIAS: Final = MappingProxyType(
    {
        MarketRegime.BULL_TRENDING: DirectionalBias.BULLISH,
        MarketRegime.BEAR_TRENDING: DirectionalBias.BEARISH,
        MarketRegime.BREAKOUT: DirectionalBias.BULLISH,
        MarketRegime.MEAN_REVERTING: DirectionalBias.NEUTRAL,
        MarketRegime.LOW_VOLATILITY: DirectionalBias.NEUTRAL,
        MarketRegime.HIGH_VOLATILITY: DirectionalBias.NEUTRAL,
        MarketRegime.UNKNOWN: DirectionalBias.NEUTRAL,
    }
)
"""Directional bias per regime (read-only), used by multi-timeframe confirmation.

The map follows the detector's decision tree
(:mod:`mra_lib.indicators.regime_mapping`), which tests volatility first and
trend second:

- ``BULL_TRENDING`` / ``BEAR_TRENDING``: bullish / bearish by definition
  (volatility-normalized drift beyond the trend threshold).
- ``BREAKOUT``: bullish. The detector only labels a state a breakout when its
  volatility is expanding *and* its trend score is positive, so it is an upside
  breakout, not a directionless one.
- ``MEAN_REVERTING`` and ``LOW_VOLATILITY``: neutral. Both rules are reached
  only after the bull/bear trend rules failed, so the state has no significant
  drift.
- ``HIGH_VOLATILITY``: neutral. It is the first rule in the tree and is taken
  before the trend is looked at, so it carries no direction (it can be a
  crash or a squeeze). It is reported separately as a risk flag via
  :data:`RISK_REGIMES` instead of being forced into a direction.
- ``UNKNOWN``: neutral here for completeness, but the confirmation signal
  treats an UNKNOWN timeframe as *unavailable* (like a missing one), not as a
  neutral vote. UNKNOWN can come from a successful fit -- it is the decision
  tree's "no rule matched" fallback (and also non-finite state summaries) -- so
  it means "no interpretable read", which the signal handles like missing data:
  it keeps its weight in the agreement denominator and is skipped when picking
  the primary timeframe.
"""

RISK_REGIMES: Final[frozenset[MarketRegime]] = frozenset({MarketRegime.HIGH_VOLATILITY})
"""Regimes flagged as elevated risk by multi-timeframe confirmation (direction-free)."""

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


def get_regime_bias(regime: MarketRegime) -> DirectionalBias:
    """Return the directional bias for ``regime`` (NEUTRAL if unmapped)."""
    return REGIME_BIAS.get(regime, DirectionalBias.NEUTRAL)


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
