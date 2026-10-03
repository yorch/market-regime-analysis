"""
Map fitted HMM states to interpretable :class:`MarketRegime` labels.

Each state is summarized by its emission mean in *original feature units*
(the detector de-standardizes it with ``scaler.inverse_transform``) and run
through a fixed decision tree with absolute thresholds -- volatility first,
then trend:

1. ``rel_vol >= high_vol_ratio``                          -> HIGH_VOLATILITY
2. ``vol_expansion >= breakout_expansion`` and uptrend     -> BREAKOUT
3. ``trend_score >= trend_threshold``                      -> BULL_TRENDING
4. ``trend_score <= -trend_threshold``                     -> BEAR_TRENDING
5. ``rel_vol <= low_vol_ratio``                            -> LOW_VOLATILITY
6. ``autocorr <= mean_reversion_autocorr`` (negative)      -> MEAN_REVERTING
7. otherwise                                               -> UNKNOWN

where

- ``rel_vol`` is the state's volatility relative to the training sample's
  typical (geometric-mean) volatility, so a homogeneous market is not forced
  into high/low-volatility buckets (the previous 75th/25th-percentile rule
  always produced two of each);
- ``trend_score`` is ``trend_strength / volatility``: a volatility-normalized,
  scale-free drift measure (for constant drift ``mu`` and volatility ``sigma``
  it is roughly ``10 * mu / sigma`` with the default 10/30-bar SMAs). On a
  driftless random walk the per-bar score has a standard deviation of about
  2.2, so the default threshold of 1.0 calls a trend only when a whole state
  sits well away from zero;
- ``vol_expansion`` is ``log(vol_10 / vol_40)``;
- ``autocorr`` is the lag-1 return autocorrelation; MEAN_REVERTING requires it
  to be clearly negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mra_lib.config.enums import MarketRegime


@dataclass(frozen=True)
class RegimeThresholds:
    """Absolute thresholds of the state -> regime decision tree."""

    high_vol_ratio: float = 1.4
    """State volatility / typical volatility at or above which a state is HIGH_VOLATILITY."""

    low_vol_ratio: float = 1 / 1.4
    """State volatility / typical volatility at or below which a state is LOW_VOLATILITY."""

    trend_threshold: float = 1.0
    """``|trend_strength / volatility|`` needed to call a BULL/BEAR trend."""

    breakout_expansion: float = math.log(1.3)
    """``log(vol_10 / vol_40)`` at or above which an up-trending state is a BREAKOUT."""

    mean_reversion_autocorr: float = -0.1
    """Lag-1 autocorrelation at or below which a trendless state is MEAN_REVERTING."""


DEFAULT_THRESHOLDS = RegimeThresholds()


@dataclass(frozen=True)
class StateSummary:
    """A state's emission mean expressed in interpretable, scale-free units."""

    rel_vol: float
    trend_score: float
    vol_expansion: float
    autocorr: float


def classify_state(
    summary: StateSummary, thresholds: RegimeThresholds = DEFAULT_THRESHOLDS
) -> MarketRegime:
    """Apply the volatility-first, then-trend decision tree to one state."""
    values = (summary.rel_vol, summary.trend_score, summary.vol_expansion, summary.autocorr)
    if not all(math.isfinite(v) for v in values):
        return MarketRegime.UNKNOWN

    rules: tuple[tuple[bool, MarketRegime], ...] = (
        (summary.rel_vol >= thresholds.high_vol_ratio, MarketRegime.HIGH_VOLATILITY),
        (
            summary.vol_expansion >= thresholds.breakout_expansion and summary.trend_score > 0,
            MarketRegime.BREAKOUT,
        ),
        (summary.trend_score >= thresholds.trend_threshold, MarketRegime.BULL_TRENDING),
        (summary.trend_score <= -thresholds.trend_threshold, MarketRegime.BEAR_TRENDING),
        (summary.rel_vol <= thresholds.low_vol_ratio, MarketRegime.LOW_VOLATILITY),
        (
            summary.autocorr <= thresholds.mean_reversion_autocorr,
            MarketRegime.MEAN_REVERTING,
        ),
    )
    return next((regime for matched, regime in rules if matched), MarketRegime.UNKNOWN)
