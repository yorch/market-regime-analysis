"""
Backtesting framework for strategy validation.

This package provides comprehensive backtesting capabilities including:
- Historical simulation with realistic execution
- Transaction cost modeling
- Performance metrics calculation
- Walk-forward analysis
- Strategy optimization
"""

from .calibrator import CalibrationResult, RegimeMultiplierCalibrator, RegimeTradeStats
from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .optimizer import OptimizationResult, StrategyOptimizer
from .strategy import RegimeStrategy
from .trade_stats import compute_trade_stats, finite_profit_factor
from .transaction_costs import (
    EquityCostModel,
    FuturesCostModel,
    HighFrequencyCostModel,
    RetailCostModel,
    TransactionCostModel,
)
from .walk_forward import RegimeCache, WalkForwardValidator

__all__ = [
    "BacktestEngine",
    "CalibrationResult",
    "EquityCostModel",
    "FuturesCostModel",
    "HighFrequencyCostModel",
    "OptimizationResult",
    "PerformanceMetrics",
    "RegimeCache",
    "RegimeMultiplierCalibrator",
    "RegimeStrategy",
    "RegimeTradeStats",
    "RetailCostModel",
    "StrategyOptimizer",
    "TransactionCostModel",
    "WalkForwardValidator",
    "compute_trade_stats",
    "finite_profit_factor",
]
