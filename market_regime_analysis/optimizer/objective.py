"""
Objective functions for parameter optimization.

Defines metrics and objectives for evaluating parameter combinations,
including Sharpe ratio, maximum drawdown, profit factor, and more.
"""

from dataclasses import dataclass
from enum import Enum


class OptimizationObjective(Enum):
    """Optimization objectives."""

    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"  # Return / Max Drawdown
    SORTINO_RATIO = "sortino_ratio"  # Return / Downside Deviation
    PROFIT_FACTOR = "profit_factor"  # Gross Profit / Gross Loss
    WIN_RATE = "win_rate"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"  # Custom multi-factor


@dataclass
class OptimizationMetrics:
    """
    Performance metrics for optimization evaluation.

    All metrics calculated from backtest results.
    """

    # Primary metrics
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Risk metrics
    volatility: float
    sortino_ratio: float
    downside_deviation: float

    # Validation metrics
    is_valid: bool  # Passes minimum requirements
    validation_errors: list[str]

    @classmethod
    def from_backtest_results(cls, backtest_results: dict) -> "OptimizationMetrics":
        """
        Create metrics from backtest results.

        Args:
            backtest_results: Dict from BacktestEngine.run_regime_strategy()

        Returns:
            OptimizationMetrics instance
        """
        performance = backtest_results["performance"]
        trades = backtest_results["trades"]

        # Get metrics from performance object (stored in .metrics dict)
        metrics = performance.metrics

        # Calculate validation
        validation_errors = []
        is_valid = True

        # Minimum 10 trades required for statistical significance
        if len(trades) < 10:
            validation_errors.append(f"Insufficient trades: {len(trades)} < 10")
            is_valid = False

        # Sharpe ratio must be finite
        sharpe = metrics["sharpe_ratio"]
        if not (-10 < sharpe < 10):
            validation_errors.append(f"Invalid Sharpe: {sharpe}")
            is_valid = False

        # Max drawdown must be reasonable
        max_dd = metrics["max_drawdown"]
        if max_dd < -0.50:  # <-50% drawdown rejected
            validation_errors.append(f"Excessive drawdown: {max_dd:.1%}")
            is_valid = False

        return cls(
            sharpe_ratio=metrics["sharpe_ratio"],
            total_return=metrics["total_return"],
            max_drawdown=metrics["max_drawdown"],
            calmar_ratio=metrics["calmar_ratio"],
            num_trades=len(trades),
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            avg_win=metrics["avg_win"],
            avg_loss=metrics["avg_loss"],
            volatility=metrics["annualized_volatility"],
            sortino_ratio=metrics["sortino_ratio"],
            downside_deviation=metrics["downside_deviation"],
            is_valid=is_valid,
            validation_errors=validation_errors,
        )

    def get_objective_value(self, objective: OptimizationObjective) -> float:
        """
        Get the value for a specific optimization objective.

        Args:
            objective: The objective to optimize

        Returns:
            Objective value (higher is better)
        """
        if not self.is_valid:
            return -999.0  # Invalid results get very low score

        if objective == OptimizationObjective.SHARPE_RATIO:
            return self.sharpe_ratio

        elif objective == OptimizationObjective.CALMAR_RATIO:
            return self.calmar_ratio

        elif objective == OptimizationObjective.SORTINO_RATIO:
            return self.sortino_ratio

        elif objective == OptimizationObjective.PROFIT_FACTOR:
            return self.profit_factor

        elif objective == OptimizationObjective.WIN_RATE:
            return self.win_rate

        elif objective == OptimizationObjective.TOTAL_RETURN:
            return self.total_return

        elif objective == OptimizationObjective.RISK_ADJUSTED_RETURN:
            # Custom multi-factor objective
            # Combines Sharpe, Calmar, and win rate
            return (
                0.5 * self.sharpe_ratio + 0.3 * self.calmar_ratio + 0.2 * (self.win_rate - 0.5)
            )

        else:
            raise ValueError(f"Unknown objective: {objective}")


class ObjectiveFunction:
    """
    Objective function for parameter optimization.

    Wraps backtest execution and metric calculation into a single
    callable that takes parameters and returns an objective value.
    """

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
        min_trades: int = 10,
        max_drawdown_limit: float = 0.50,
    ) -> None:
        """
        Initialize objective function.

        Args:
            objective: Which metric to optimize
            min_trades: Minimum trades required for valid result
            max_drawdown_limit: Maximum acceptable drawdown (as fraction)
        """
        self.objective = objective
        self.min_trades = min_trades
        self.max_drawdown_limit = max_drawdown_limit

    def evaluate(self, backtest_results: dict) -> tuple[float, OptimizationMetrics]:
        """
        Evaluate a parameter combination.

        Args:
            backtest_results: Results from BacktestEngine

        Returns:
            Tuple of (objective_value, metrics)
        """
        metrics = OptimizationMetrics.from_backtest_results(backtest_results)
        objective_value = metrics.get_objective_value(self.objective)

        return objective_value, metrics

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ObjectiveFunction(objective={self.objective.value}, "
            f"min_trades={self.min_trades}, "
            f"max_drawdown={self.max_drawdown_limit:.1%})"
        )
