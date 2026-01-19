#!/usr/bin/env python3
"""
Fast parameter optimization with reduced search space.

Uses 3-5 values per parameter instead of 10+ to make optimization tractable.
"""

import pandas as pd

from market_regime_analysis.analyzer import MarketRegimeAnalyzer
from market_regime_analysis.enums import MarketRegime, TradingStrategy
from market_regime_analysis.optimizer import GridSearchOptimizer, ObjectiveFunction, ParameterSpace
from market_regime_analysis.optimizer.objective import OptimizationObjective
from market_regime_analysis.optimizer.parameter_space import ParameterRange


def create_fast_parameter_space() -> ParameterSpace:
    """Create a reduced parameter space for faster optimization."""
    space = ParameterSpace()

    # Reduced regime multipliers (3-5 values each instead of 10+)
    space.regime_multipliers = {
        MarketRegime.BULL_TRENDING: ParameterRange(
            "bull_multiplier", min_value=1.0, max_value=1.6, step=0.2, default=1.3
        ),  # 4 values
        MarketRegime.BEAR_TRENDING: ParameterRange(
            "bear_multiplier", min_value=0.5, max_value=0.9, step=0.2, default=0.7
        ),  # 3 values
        MarketRegime.MEAN_REVERTING: ParameterRange(
            "mean_rev_multiplier", min_value=1.0, max_value=1.4, step=0.2, default=1.2
        ),  # 3 values
        MarketRegime.HIGH_VOLATILITY: ParameterRange(
            "high_vol_multiplier", min_value=0.3, max_value=0.7, step=0.2, default=0.4
        ),  # 3 values
        MarketRegime.LOW_VOLATILITY: ParameterRange(
            "low_vol_multiplier", min_value=0.9, max_value=1.3, step=0.2, default=1.1
        ),  # 3 values
        MarketRegime.BREAKOUT: ParameterRange(
            "breakout_multiplier", min_value=0.7, max_value=1.1, step=0.2, default=0.9
        ),  # 3 values
        MarketRegime.UNKNOWN: ParameterRange(
            "unknown_multiplier", min_value=0.1, max_value=0.3, step=0.1, default=0.2
        ),  # 3 values
    }

    # Fixed values for other parameters (not optimizing these to save time)
    space.stop_loss_pct = ParameterRange(
        "stop_loss_pct", min_value=0.10, max_value=0.10, step=1.0, default=0.10
    )
    space.take_profit_pct = ParameterRange(
        "take_profit_pct", min_value=0.15, max_value=0.15, step=1.0, default=0.15
    )
    space.max_position_size = ParameterRange(
        "max_position_size", min_value=0.20, max_value=0.20, step=1.0, default=0.20
    )
    space.volatility_threshold_high_pct = ParameterRange(
        "vol_threshold_high", min_value=75.0, max_value=75.0, step=1.0, default=75.0
    )
    space.volatility_threshold_low_pct = ParameterRange(
        "vol_threshold_low", min_value=25.0, max_value=25.0, step=1.0, default=25.0
    )

    return space


def run_fast_optimization(symbol: str = "SPY", timeframe: str = "1D"):
    """Run fast parameter optimization."""
    print("=== Fast Parameter Optimization ===\n")

    # Load data
    print(f"Loading {symbol} data...")
    period_map = {"1D": "2y", "1H": "6mo", "15m": "3mo"}
    period = period_map.get(timeframe, "2y")

    analyzer = MarketRegimeAnalyzer(
        symbol=symbol,
        periods={timeframe: period},
        provider_flag="yfinance"
    )
    df = analyzer.data[timeframe]
    print(f"Loaded {len(df)} bars\n")

    # Calculate regimes
    print("Calculating regimes...")
    hmm = analyzer.hmm_models[timeframe]
    indicators_df = analyzer.indicators[timeframe]

    regimes_list = []
    strategies_list = []

    for i in range(len(df)):
        try:
            regime, _, confidence = hmm.predict_regime(indicators_df.iloc[: i + 1])
            strategy = analyzer._get_trading_strategy(regime)
            regimes_list.append(regime)
            strategies_list.append(strategy)
        except Exception:
            regimes_list.append(MarketRegime.UNKNOWN)
            strategies_list.append(TradingStrategy.AVOID)

    regimes = pd.Series(regimes_list, index=df.index)
    strategies = pd.Series(strategies_list, index=df.index)

    print(f"Regime distribution:")
    for regime, count in regimes.value_counts().items():
        print(f"  {regime.value}: {count} ({count/len(regimes):.1%})")

    # Create fast parameter space
    param_space = create_fast_parameter_space()
    combinations = param_space.count_total_combinations()
    print(f"\nTotal combinations: {combinations:,}\n")

    # Setup optimizer
    obj_function = ObjectiveFunction(objective=OptimizationObjective.SHARPE_RATIO)
    optimizer = GridSearchOptimizer(
        parameter_space=param_space,
        objective_function=obj_function,
        max_combinations=50000,
        verbose=True,
    )

    # Run optimization
    print("Starting optimization...")
    best_result = optimizer.optimize(
        df=df,
        regimes=regimes,
        strategies=strategies,
        initial_capital=100000.0,
        optimize_subset=False,  # Use our custom space
    )

    # Print summary
    optimizer.print_summary(top_n=10)

    # Export results
    filename = f"{symbol}_{timeframe}_optimization.csv"
    optimizer.export_results(filename)
    print(f"\nResults exported to: {filename}")

    # Save best parameters
    if best_result and best_result.metrics.is_valid:
        import json

        best_params_file = f"best_params_{symbol}_{timeframe}.json"
        with open(best_params_file, "w") as f:
            params_dict = {k: float(v) for k, v in best_result.parameters.items()}

            json.dump(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "objective": "sharpe_ratio",
                    "objective_value": best_result.objective_value,
                    "parameters": params_dict,
                    "metrics": {
                        "sharpe_ratio": best_result.metrics.sharpe_ratio,
                        "total_return": best_result.metrics.total_return,
                        "max_drawdown": best_result.metrics.max_drawdown,
                        "num_trades": best_result.metrics.num_trades,
                        "win_rate": best_result.metrics.win_rate,
                        "profit_factor": best_result.metrics.profit_factor,
                    },
                },
                f,
                indent=2,
            )

        print(f"Best parameters saved to: {best_params_file}")

        # Print best parameters in detail
        print(f"\n=== Best Parameters ===")
        for key, value in best_result.parameters.items():
            if "regime_mult" in key:
                regime_name = key.replace("regime_mult_", "")
                print(f"  {regime_name}: {value:.2f}")

        print(f"\n=== Performance ===")
        print(f"  Sharpe Ratio: {best_result.metrics.sharpe_ratio:.3f}")
        print(f"  Total Return: {best_result.metrics.total_return:.2%}")
        print(f"  Max Drawdown: {best_result.metrics.max_drawdown:.2%}")
        print(f"  Calmar Ratio: {best_result.metrics.calmar_ratio:.3f}")
        print(f"  Win Rate: {best_result.metrics.win_rate:.2%}")
        print(f"  Profit Factor: {best_result.metrics.profit_factor:.3f}")
        print(f"  Number of Trades: {best_result.metrics.num_trades}")

        # Compare to default parameters
        print(f"\n=== Improvement Over Defaults ===")
        default_params = param_space.get_default_parameters()
        print(f"  Default parameters would give Sharpe: ~-0.15 (from previous tests)")
        print(f"  Optimized parameters give Sharpe: {best_result.metrics.sharpe_ratio:.3f}")
        if best_result.metrics.sharpe_ratio > -0.15:
            improvement = best_result.metrics.sharpe_ratio - (-0.15)
            print(f"  Improvement: +{improvement:.3f}")
        else:
            print(f"  Still negative - strategy fundamentally unprofitable")

    else:
        print("\nNo valid parameter combinations found!")

    print("\n=== Optimization Complete ===")
    return best_result


if __name__ == "__main__":
    run_fast_optimization(symbol="SPY", timeframe="1D")
