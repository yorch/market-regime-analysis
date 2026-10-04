"""
Backtesting framework for strategy validation.

This package provides comprehensive backtesting capabilities including:
- Historical simulation with realistic execution
- Transaction cost modeling
- Performance metrics calculation
- Walk-forward analysis
- Strategy optimization
- One-call backtest of a parameter set vs buy-and-hold (:func:`run_backtest`)
"""

from .calibrator import CalibrationResult, RegimeMultiplierCalibrator, RegimeTradeStats
from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .optimizer import OptimizationResult, StrategyOptimizer
from .runner import (
    BACKTEST_MODES,
    COST_MODELS,
    BacktestReport,
    SideMetrics,
    StrategyParams,
    WindowSummary,
    format_backtest_report,
    load_strategy_params,
    make_cost_model,
    parse_strategy_params,
    run_backtest,
    validate_strategy_params,
)
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
    "BACKTEST_MODES",
    "COST_MODELS",
    "BacktestEngine",
    "BacktestReport",
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
    "SideMetrics",
    "StrategyOptimizer",
    "StrategyParams",
    "TransactionCostModel",
    "WalkForwardValidator",
    "WindowSummary",
    "compute_trade_stats",
    "finite_profit_factor",
    "format_backtest_report",
    "load_strategy_params",
    "make_cost_model",
    "parse_strategy_params",
    "run_backtest",
    "validate_strategy_params",
]
