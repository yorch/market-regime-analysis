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
from .transaction_costs import (
    EquityCostModel,
    FuturesCostModel,
    HighFrequencyCostModel,
    RetailCostModel,
    TransactionCostModel,
)
from .walk_forward import WalkForwardValidator

__all__ = [
    "BacktestEngine",
    "CalibrationResult",
    "EquityCostModel",
    "FuturesCostModel",
    "HighFrequencyCostModel",
    "OptimizationResult",
    "PerformanceMetrics",
    "RegimeMultiplierCalibrator",
    "RegimeStrategy",
    "RegimeTradeStats",
    "RetailCostModel",
    "StrategyOptimizer",
    "TransactionCostModel",
    "WalkForwardValidator",
]
