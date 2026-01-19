#!/usr/bin/env python3
"""
Quick test of the parameter optimization system.

Tests the optimizer with a minimal parameter space to verify it works.
"""

import pandas as pd

from market_regime_analysis.analyzer import MarketRegimeAnalyzer
from market_regime_analysis.enums import MarketRegime, TradingStrategy
from market_regime_analysis.optimizer import GridSearchOptimizer, ObjectiveFunction, ParameterSpace
from market_regime_analysis.optimizer.objective import OptimizationObjective
from market_regime_analysis.optimizer.parameter_space import ParameterRange


def test_optimizer():
    """Test the optimizer with minimal parameter space."""
    print("=== Testing Parameter Optimizer ===\n")

    # Load data (only 1D timeframe to avoid yfinance period issues)
    print("Loading SPY data...")
    analyzer = MarketRegimeAnalyzer(
        symbol="SPY",
        periods={"1D": "2y"},  # Only load daily data
        provider_flag="yfinance",
    )
    df = analyzer.data["1D"]
    print(f"Loaded {len(df)} bars\n")

    # Calculate regimes for all data
    print("Calculating regimes...")
    hmm = analyzer.hmm_models["1D"]
    indicators_df = analyzer.indicators["1D"]

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
    print()

    # Create minimal parameter space for testing (only 3 values per parameter)
    print("Creating minimal parameter space...")
    param_space = ParameterSpace()

    # Override with minimal ranges for testing
    param_space.regime_multipliers = {
        MarketRegime.BULL_TRENDING: ParameterRange(
            "bull_multiplier", min_value=1.0, max_value=1.5, step=0.25, default=1.3
        ),
        MarketRegime.BEAR_TRENDING: ParameterRange(
            "bear_multiplier", min_value=0.5, max_value=0.9, step=0.2, default=0.7
        ),
        MarketRegime.MEAN_REVERTING: ParameterRange(
            "mean_rev_multiplier", min_value=1.0, max_value=1.4, step=0.2, default=1.2
        ),
        MarketRegime.HIGH_VOLATILITY: ParameterRange(
            "high_vol_multiplier", min_value=0.3, max_value=0.5, step=0.1, default=0.4
        ),
        MarketRegime.LOW_VOLATILITY: ParameterRange(
            "low_vol_multiplier", min_value=1.0, max_value=1.2, step=0.1, default=1.1
        ),
        MarketRegime.BREAKOUT: ParameterRange(
            "breakout_multiplier", min_value=0.8, max_value=1.0, step=0.1, default=0.9
        ),
        MarketRegime.UNKNOWN: ParameterRange(
            "unknown_multiplier", min_value=0.1, max_value=0.3, step=0.1, default=0.2
        ),
    }

    # Use fixed values for other parameters (single value = no optimization)
    param_space.stop_loss_pct = ParameterRange(
        "stop_loss_pct", min_value=0.10, max_value=0.10, step=1.0, default=0.10
    )
    param_space.take_profit_pct = ParameterRange(
        "take_profit_pct", min_value=0.15, max_value=0.15, step=1.0, default=0.15
    )
    param_space.max_position_size = ParameterRange(
        "max_position_size", min_value=0.20, max_value=0.20, step=1.0, default=0.20
    )
    param_space.volatility_threshold_high_pct = ParameterRange(
        "vol_threshold_high", min_value=75.0, max_value=75.0, step=1.0, default=75.0
    )
    param_space.volatility_threshold_low_pct = ParameterRange(
        "vol_threshold_low", min_value=25.0, max_value=25.0, step=1.0, default=25.0
    )

    combinations = param_space.count_total_combinations()
    print(f"Total combinations to test: {combinations}\n")

    # Create optimizer
    obj_function = ObjectiveFunction(objective=OptimizationObjective.SHARPE_RATIO)
    optimizer = GridSearchOptimizer(
        parameter_space=param_space,
        objective_function=obj_function,
        max_combinations=10000,
        verbose=True,
    )

    # Run optimization
    print("Running optimization...\n")
    best_result = optimizer.optimize(
        df=df,
        regimes=regimes,
        strategies=strategies,
        initial_capital=100000.0,
        optimize_subset=False,  # Use our minimal space
    )

    # Print summary
    optimizer.print_summary(top_n=3)

    # Check results
    if best_result and best_result.metrics.is_valid:
        print("\n✓ Optimizer test PASSED")
        print(f"  Best Sharpe: {best_result.metrics.sharpe_ratio:.3f}")
        print(f"  Best Return: {best_result.metrics.total_return:.2%}")
        print(f"  Max Drawdown: {best_result.metrics.max_drawdown:.2%}")
        return True
    else:
        print("\n✗ Optimizer test FAILED - No valid results")
        return False


if __name__ == "__main__":
    success = test_optimizer()
    exit(0 if success else 1)
