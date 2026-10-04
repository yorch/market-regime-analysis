"""Configuration module — enums, data classes, and regime lookup tables."""

from .data_classes import RegimeAnalysis
from .enums import DirectionalBias, MarketRegime, TradingStrategy
from .regime_tables import (
    PERIODS_PER_YEAR,
    REGIME_BIAS,
    REGIME_MULTIPLIERS,
    REGIME_STRATEGIES,
    RISK_REGIMES,
    get_regime_bias,
    get_regime_multiplier,
    get_regime_strategy,
    periods_per_year,
)

__all__ = [
    "PERIODS_PER_YEAR",
    "REGIME_BIAS",
    "REGIME_MULTIPLIERS",
    "REGIME_STRATEGIES",
    "RISK_REGIMES",
    "DirectionalBias",
    "MarketRegime",
    "RegimeAnalysis",
    "TradingStrategy",
    "get_regime_bias",
    "get_regime_multiplier",
    "get_regime_strategy",
    "periods_per_year",
]
