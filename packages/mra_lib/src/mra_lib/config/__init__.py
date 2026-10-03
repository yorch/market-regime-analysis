"""Configuration module — enums, data classes, and regime lookup tables."""

from .data_classes import RegimeAnalysis
from .enums import MarketRegime, TradingStrategy
from .regime_tables import (
    PERIODS_PER_YEAR,
    REGIME_MULTIPLIERS,
    REGIME_STRATEGIES,
    get_regime_multiplier,
    get_regime_strategy,
    periods_per_year,
)

__all__ = [
    "PERIODS_PER_YEAR",
    "REGIME_MULTIPLIERS",
    "REGIME_STRATEGIES",
    "MarketRegime",
    "RegimeAnalysis",
    "TradingStrategy",
    "get_regime_multiplier",
    "get_regime_strategy",
    "periods_per_year",
]
