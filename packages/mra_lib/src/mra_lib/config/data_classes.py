"""
Data classes for market regime analysis system.

This module defines the core data structures used to represent
regime analysis results and related information.
"""

from dataclasses import dataclass

from .enums import MarketRegime, TradingStrategy


@dataclass
class RegimeAnalysis:
    """
    Comprehensive regime analysis results.

    This dataclass encapsulates all the key information from a market
    regime analysis, including the detected regime, confidence metrics,
    and trading recommendations.

    Attributes:
        current_regime: The detected market regime
        hmm_state: The underlying HMM state (0-5)
        transition_probability: Probability of transitioning to this state
        regime_persistence: How stable the current regime is (0-1)
        recommended_strategy: Optimal trading strategy for this regime
        position_sizing_multiplier: Risk-adjusted position size multiplier
        risk_level: Human-readable risk assessment
        arbitrage_opportunities: List of identified statistical arbitrage signals
        statistical_signals: List of regime-specific statistical signals
        key_levels: Dictionary of support/resistance levels
        regime_confidence: Confidence in the regime classification (0-1)
    """

    current_regime: MarketRegime
    hmm_state: int
    transition_probability: float
    regime_persistence: float
    recommended_strategy: TradingStrategy
    position_sizing_multiplier: float
    risk_level: str
    arbitrage_opportunities: list[str]
    statistical_signals: list[str]
    key_levels: dict[str, float]
    regime_confidence: float
