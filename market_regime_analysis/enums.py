"""
Enumerations for market regime analysis system.

This module defines the core enumerations used throughout the market regime
analysis system, including market regimes and trading strategies.
"""

from enum import Enum


class MarketRegime(Enum):
    """
    Market regime classifications following Jim Simons' HMM methodology.

    These regimes represent the fundamental market states that drive
    different trading strategies and risk management approaches.
    """

    BULL_TRENDING = "Bull Trending"
    BEAR_TRENDING = "Bear Trending"
    MEAN_REVERTING = "Mean Reverting"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    BREAKOUT = "Breakout"
    UNKNOWN = "Unknown"


class TradingStrategy(Enum):
    """
    Trading strategies mapped to market regimes.

    Each strategy corresponds to optimal approaches for different
    market regimes as identified by the HMM analysis.
    """

    TREND_FOLLOWING = "Trend Following"
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "Mean Reversion"
    STATISTICAL_ARBITRAGE = "Statistical Arbitrage"
    VOLATILITY_TRADING = "Volatility Trading"
    DEFENSIVE = "Defensive"
    AVOID = "Avoid Trading"
