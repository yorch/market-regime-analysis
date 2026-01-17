"""
Backtesting framework for strategy validation.

This package provides comprehensive backtesting capabilities including:
- Historical simulation with realistic execution
- Transaction cost modeling
- Performance metrics calculation
- Walk-forward analysis
- Strategy comparison

Critical for validating regime-based trading strategies before deployment.
"""

from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .transaction_costs import (
    EquityCostModel,
    FuturesCostModel,
    HighFrequencyCostModel,
    RetailCostModel,
    TransactionCostModel,
)

__all__ = [
    "BacktestEngine",
    "PerformanceMetrics",
    "TransactionCostModel",
    "EquityCostModel",
    "FuturesCostModel",
    "HighFrequencyCostModel",
    "RetailCostModel",
]
