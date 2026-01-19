"""
Grid search parameter optimization.

Systematic search through parameter combinations to find optimal settings
for regime-based trading strategies.
"""

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from ..backtester.engine import BacktestEngine
from ..enums import MarketRegime
from .objective import ObjectiveFunction, OptimizationMetrics, OptimizationObjective
from .parameter_space import ParameterSpace


@dataclass
class OptimizationResult:
    """Results from a single parameter combination."""

    parameters: dict[str, Any]
    objective_value: float
    metrics: OptimizationMetrics
    rank: int | None = None


class GridSearchOptimizer:
    """
    Grid search optimizer for strategy parameters.

    Systematically tests all combinations of parameters within defined
    ranges to find the optimal configuration.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace | None = None,
        objective_function: ObjectiveFunction | None = None,
        max_combinations: int = 10000,
        verbose: bool = True,
    ) -> None:
        """
        Initialize grid search optimizer.

        Args:
            parameter_space: Parameter ranges to search
            objective_function: Objective to optimize
            max_combinations: Maximum parameter combinations to test
            verbose: Print progress information
        """
        self.parameter_space = parameter_space or ParameterSpace()
        self.objective_function = objective_function or ObjectiveFunction(
            objective=OptimizationObjective.SHARPE_RATIO
        )
        self.max_combinations = max_combinations
        self.verbose = verbose

        # Results storage
        self.results: list[OptimizationResult] = []
        self.best_result: OptimizationResult | None = None

    def optimize(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,
        strategies: pd.Series,
        initial_capital: float = 100000.0,
        optimize_subset: bool = True,
    ) -> OptimizationResult:
        """
        Run grid search optimization.

        Args:
            df: Price data for backtesting
            regimes: Regime classifications
            strategies: Trading strategies
            initial_capital: Starting capital for backtest
            optimize_subset: If True, only optimize regime multipliers

        Returns:
            Best optimization result
        """
        # Get parameter space (possibly reduced)
        space = (
            self.parameter_space.get_subset_space(optimize_regime_only=True)
            if optimize_subset
            else self.parameter_space
        )

        # Generate parameter combinations
        param_combinations = self._generate_combinations(space)

        if len(param_combinations) > self.max_combinations:
            print(
                f"Warning: {len(param_combinations)} combinations exceed "
                f"max {self.max_combinations}. Using subset optimization."
            )
            space = self.parameter_space.get_subset_space(optimize_regime_only=True)
            param_combinations = self._generate_combinations(space)

        if self.verbose:
            print(f"\n=== Grid Search Optimization ===")
            print(f"Total combinations: {len(param_combinations)}")
            print(f"Objective: {self.objective_function.objective.value}")
            print(f"Parameter space:")
            for regime, param_range in space.regime_multipliers.items():
                print(
                    f"  {regime.value}: [{param_range.min_value:.1f}, {param_range.max_value:.1f}]"
                )
            print()

        # Test each combination
        self.results = []
        iterator = (
            tqdm(param_combinations, desc="Testing parameters")
            if self.verbose
            else param_combinations
        )

        for params in iterator:
            try:
                result = self._evaluate_parameters(
                    params, df, regimes, strategies, initial_capital
                )
                self.results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Error testing parameters {params}: {e}")
                # Create failed result
                self.results.append(
                    OptimizationResult(
                        parameters=params,
                        objective_value=-999.0,
                        metrics=OptimizationMetrics(
                            sharpe_ratio=-999,
                            total_return=0,
                            max_drawdown=1.0,
                            calmar_ratio=-999,
                            num_trades=0,
                            win_rate=0,
                            profit_factor=0,
                            avg_win=0,
                            avg_loss=0,
                            volatility=0,
                            sortino_ratio=-999,
                            downside_deviation=0,
                            is_valid=False,
                            validation_errors=[str(e)],
                        ),
                    )
                )

        # Rank results
        self._rank_results()

        # Get best result
        valid_results = [r for r in self.results if r.metrics.is_valid]
        if valid_results:
            self.best_result = max(valid_results, key=lambda r: r.objective_value)

            if self.verbose:
                print(f"\n=== Optimization Complete ===")
                print(f"Valid results: {len(valid_results)}/{len(self.results)}")
                print(f"Best {self.objective_function.objective.value}: {self.best_result.objective_value:.3f}")
                print(f"Best parameters:")
                for key, value in self.best_result.parameters.items():
                    print(f"  {key}: {value:.3f}")
        else:
            if self.verbose:
                print(f"\n=== Optimization Failed ===")
                print(f"No valid parameter combinations found!")
            self.best_result = self.results[0] if self.results else None

        return self.best_result

    def _generate_combinations(self, space: ParameterSpace) -> list[dict[str, Any]]:
        """
        Generate all parameter combinations from space.

        Args:
            space: Parameter space to search

        Returns:
            List of parameter dictionaries
        """
        # Collect all parameter grids
        param_grids = {}

        # Regime multipliers
        for regime, param_range in space.regime_multipliers.items():
            param_grids[f"regime_mult_{regime.value}"] = param_range.get_grid_values()

        # Risk management
        param_grids["stop_loss_pct"] = space.stop_loss_pct.get_grid_values()
        param_grids["take_profit_pct"] = space.take_profit_pct.get_grid_values()
        param_grids["max_position_size"] = space.max_position_size.get_grid_values()

        # HMM parameters
        param_grids["vol_threshold_high_pct"] = (
            space.volatility_threshold_high_pct.get_grid_values()
        )
        param_grids["vol_threshold_low_pct"] = (
            space.volatility_threshold_low_pct.get_grid_values()
        )

        # Generate all combinations
        keys = list(param_grids.keys())
        value_lists = [param_grids[k] for k in keys]

        combinations = []
        for values in itertools.product(*value_lists):
            param_dict = dict(zip(keys, values))
            combinations.append(param_dict)

        return combinations

    def _evaluate_parameters(
        self,
        params: dict[str, Any],
        df: pd.DataFrame,
        regimes: pd.Series,
        strategies: pd.Series,
        initial_capital: float,
    ) -> OptimizationResult:
        """
        Evaluate a single parameter combination.

        Args:
            params: Parameter dictionary
            df: Price data
            regimes: Regime series
            strategies: Strategy series
            initial_capital: Starting capital

        Returns:
            OptimizationResult
        """
        # Extract backtest parameters
        stop_loss = params.get("stop_loss_pct", 0.10)
        take_profit = params.get("take_profit_pct", None)
        max_position = params.get("max_position_size", 0.20)

        # Create backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital,
            max_position_size=max_position,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
        )

        # Calculate position sizes using regime multipliers from params
        position_sizes = self._calculate_position_sizes(params, regimes)

        # Run backtest
        backtest_results = engine.run_regime_strategy(df, regimes, strategies, position_sizes)

        # Evaluate objective
        objective_value, metrics = self.objective_function.evaluate(backtest_results)

        return OptimizationResult(
            parameters=params, objective_value=objective_value, metrics=metrics
        )

    def _calculate_position_sizes(
        self, params: dict[str, Any], regimes: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes based on regime and parameters.

        Args:
            params: Parameter dictionary with regime multipliers
            regimes: Regime series

        Returns:
            Series of position size multipliers
        """
        # Build regime multiplier dict from params
        regime_multipliers = {}
        for regime in MarketRegime:
            key = f"regime_mult_{regime.value}"
            if key in params:
                regime_multipliers[regime] = params[key]
            else:
                # Use default if not in params
                regime_multipliers[regime] = 0.5

        # Map regimes to position sizes
        position_sizes = regimes.map(regime_multipliers)

        # Handle any NaN values
        position_sizes = position_sizes.fillna(0.0)

        return position_sizes

    def _rank_results(self) -> None:
        """Rank all results by objective value."""
        # Sort by objective value (descending)
        sorted_results = sorted(self.results, key=lambda r: r.objective_value, reverse=True)

        # Assign ranks
        for rank, result in enumerate(sorted_results, start=1):
            result.rank = rank

    def get_top_results(self, n: int = 10) -> list[OptimizationResult]:
        """
        Get top N results.

        Args:
            n: Number of results to return

        Returns:
            List of top results
        """
        valid_results = [r for r in self.results if r.metrics.is_valid]
        sorted_results = sorted(valid_results, key=lambda r: r.objective_value, reverse=True)
        return sorted_results[:n]

    def export_results(self, filename: str) -> None:
        """
        Export all results to CSV.

        Args:
            filename: Output CSV file path
        """
        # Build DataFrame
        data = []
        for result in self.results:
            row = {
                "rank": result.rank,
                "objective_value": result.objective_value,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "total_return": result.metrics.total_return,
                "max_drawdown": result.metrics.max_drawdown,
                "calmar_ratio": result.metrics.calmar_ratio,
                "num_trades": result.metrics.num_trades,
                "win_rate": result.metrics.win_rate,
                "profit_factor": result.metrics.profit_factor,
                "is_valid": result.metrics.is_valid,
            }
            # Add parameters
            row.update(result.parameters)
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values("rank")
        df.to_csv(filename, index=False)

        print(f"Exported {len(df)} results to {filename}")

    def print_summary(self, top_n: int = 5) -> None:
        """
        Print optimization summary.

        Args:
            top_n: Number of top results to display
        """
        print(f"\n=== Optimization Summary ===")
        print(f"Total tests: {len(self.results)}")

        valid_results = [r for r in self.results if r.metrics.is_valid]
        print(f"Valid results: {len(valid_results)}")

        if not valid_results:
            print("No valid results found!")
            return

        print(f"\nTop {top_n} Parameter Combinations:")
        print("-" * 100)

        top_results = self.get_top_results(top_n)
        for i, result in enumerate(top_results, 1):
            print(f"\n#{i} - Objective: {result.objective_value:.3f}")
            print(f"  Sharpe: {result.metrics.sharpe_ratio:.3f}")
            print(f"  Return: {result.metrics.total_return:.2%}")
            print(f"  Max DD: {result.metrics.max_drawdown:.2%}")
            print(f"  Trades: {result.metrics.num_trades}")
            print(f"  Win Rate: {result.metrics.win_rate:.2%}")

            # Show regime multipliers
            print("  Regime Multipliers:")
            for key, value in result.parameters.items():
                if "regime_mult" in key:
                    regime_name = key.replace("regime_mult_", "")
                    print(f"    {regime_name}: {value:.2f}")
