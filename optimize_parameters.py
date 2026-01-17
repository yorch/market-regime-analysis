#!/usr/bin/env python3
"""
Parameter optimization script.

Run grid search optimization to find optimal trading strategy parameters.
"""

import sys
from pathlib import Path

import click
import pandas as pd

from market_regime_analysis.analyzer import MarketRegimeAnalyzer
from market_regime_analysis.optimizer import GridSearchOptimizer, ObjectiveFunction, ParameterSpace
from market_regime_analysis.optimizer.objective import OptimizationObjective


@click.group()
def cli():
    """Parameter optimization for market regime analysis."""
    pass


@cli.command()
@click.option("--symbol", default="SPY", help="Symbol to optimize (default: SPY)")
@click.option(
    "--timeframe",
    default="1D",
    type=click.Choice(["1D", "1H", "15m"]),
    help="Timeframe to use (default: 1D)",
)
@click.option(
    "--objective",
    default="sharpe_ratio",
    type=click.Choice(
        [
            "sharpe_ratio",
            "calmar_ratio",
            "sortino_ratio",
            "profit_factor",
            "win_rate",
            "total_return",
        ]
    ),
    help="Objective to optimize (default: sharpe_ratio)",
)
@click.option(
    "--provider",
    default="yfinance",
    type=click.Choice(["yfinance", "alphavantage", "polygon"]),
    help="Data provider (default: yfinance)",
)
@click.option(
    "--max-combinations",
    default=1000,
    type=int,
    help="Maximum parameter combinations to test (default: 1000)",
)
@click.option(
    "--regime-only",
    is_flag=True,
    help="Only optimize regime multipliers (faster)",
)
@click.option(
    "--output",
    default="optimization_results.csv",
    help="Output CSV file for results (default: optimization_results.csv)",
)
def grid_search(symbol, timeframe, objective, provider, max_combinations, regime_only, output):
    """
    Run grid search parameter optimization.

    Example:
        python optimize_parameters.py grid-search --symbol SPY --objective sharpe_ratio
    """
    click.echo(f"\n=== Parameter Optimization: {symbol} ===\n")

    # Initialize analyzer to get data and regimes
    click.echo(f"Loading data for {symbol}...")
    try:
        analyzer = MarketRegimeAnalyzer(symbol=symbol, provider_flag=provider)
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    # Get data for specified timeframe
    if timeframe not in analyzer.data:
        click.echo(f"Error: No data for timeframe {timeframe}", err=True)
        sys.exit(1)

    df = analyzer.data[timeframe]
    click.echo(f"Loaded {len(df)} bars of {timeframe} data")

    # Calculate regimes and strategies using current model
    click.echo("Calculating regimes...")
    try:
        analysis = analyzer.analyze_current_regime(timeframe)
    except Exception as e:
        click.echo(f"Error analyzing regime: {e}", err=True)
        sys.exit(1)

    # Get regime and strategy series for backtest
    # We'll use the HMM model to predict regimes for all historical data
    hmm = analyzer.hmm_models[timeframe]
    indicators_df = analyzer.indicators[timeframe]

    # Predict regimes for all data
    regimes_list = []
    strategies_list = []

    for i in range(len(df)):
        try:
            regime, _, confidence = hmm.predict_regime(indicators_df.iloc[: i + 1])
            strategy = analyzer._get_trading_strategy(regime)
            regimes_list.append(regime)
            strategies_list.append(strategy)
        except Exception:
            # Not enough data or error - use defaults
            from market_regime_analysis.enums import MarketRegime, TradingStrategy

            regimes_list.append(MarketRegime.UNKNOWN)
            strategies_list.append(TradingStrategy.AVOID)

    regimes = pd.Series(regimes_list, index=df.index)
    strategies = pd.Series(strategies_list, index=df.index)

    click.echo(f"Regime distribution:")
    for regime, count in regimes.value_counts().items():
        click.echo(f"  {regime.value}: {count} ({count/len(regimes):.1%})")

    # Setup optimization
    objective_enum = OptimizationObjective[objective.upper()]
    param_space = ParameterSpace()
    obj_function = ObjectiveFunction(objective=objective_enum)

    optimizer = GridSearchOptimizer(
        parameter_space=param_space,
        objective_function=obj_function,
        max_combinations=max_combinations,
        verbose=True,
    )

    # Run optimization
    click.echo(f"\nStarting optimization...")
    best_result = optimizer.optimize(
        df=df,
        regimes=regimes,
        strategies=strategies,
        initial_capital=100000.0,
        optimize_subset=regime_only,
    )

    # Print results
    optimizer.print_summary(top_n=5)

    # Export results
    if output:
        optimizer.export_results(output)
        click.echo(f"\nResults exported to: {output}")

    # Save best parameters
    if best_result and best_result.metrics.is_valid:
        import json

        best_params_file = f"best_params_{symbol}_{timeframe}.json"
        with open(best_params_file, "w") as f:
            # Convert to serializable format
            params_dict = {}
            for key, value in best_result.parameters.items():
                params_dict[key] = float(value)

            json.dump(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "objective": objective,
                    "objective_value": best_result.objective_value,
                    "parameters": params_dict,
                    "metrics": {
                        "sharpe_ratio": best_result.metrics.sharpe_ratio,
                        "total_return": best_result.metrics.total_return,
                        "max_drawdown": best_result.metrics.max_drawdown,
                        "num_trades": best_result.metrics.num_trades,
                        "win_rate": best_result.metrics.win_rate,
                    },
                },
                f,
                indent=2,
            )

        click.echo(f"Best parameters saved to: {best_params_file}")
    else:
        click.echo("\nWarning: No valid parameter combination found!", err=True)

    click.echo("\n=== Optimization Complete ===\n")


@cli.command()
@click.option("--symbol", default="SPY", help="Symbol to test")
@click.option(
    "--timeframe",
    default="1D",
    type=click.Choice(["1D", "1H", "15m"]),
    help="Timeframe",
)
@click.option("--params-file", required=True, help="JSON file with optimized parameters")
@click.option(
    "--provider",
    default="yfinance",
    type=click.Choice(["yfinance", "alphavantage", "polygon"]),
    help="Data provider",
)
def validate(symbol, timeframe, params_file, provider):
    """
    Validate optimized parameters on out-of-sample data.

    Example:
        python optimize_parameters.py validate --symbol SPY --params-file best_params_SPY_1D.json
    """
    import json

    from market_regime_analysis.backtester.engine import BacktestEngine

    click.echo(f"\n=== Validating Parameters: {symbol} ===\n")

    # Load parameters
    with open(params_file) as f:
        params_data = json.load(f)

    click.echo(f"Loaded parameters from: {params_file}")
    click.echo(f"Original objective: {params_data['objective']}")
    click.echo(f"Original objective value: {params_data['objective_value']:.3f}")

    # Load data
    click.echo(f"\nLoading data for {symbol}...")
    analyzer = MarketRegimeAnalyzer(symbol=symbol, provider_flag=provider)
    df = analyzer.data[timeframe]

    # Split data: 70% train, 30% test
    split_idx = int(len(df) * 0.7)
    test_df = df.iloc[split_idx:]

    click.echo(f"Using last 30% of data for validation ({len(test_df)} bars)")

    # Calculate regimes on test data
    click.echo("Calculating regimes...")
    hmm = analyzer.hmm_models[timeframe]
    indicators_df = analyzer.indicators[timeframe]

    regimes_list = []
    strategies_list = []

    for i in range(split_idx, len(df)):
        try:
            regime, _, confidence = hmm.predict_regime(indicators_df.iloc[: i + 1])
            strategy = analyzer._get_trading_strategy(regime)
            regimes_list.append(regime)
            strategies_list.append(strategy)
        except Exception:
            from market_regime_analysis.enums import MarketRegime, TradingStrategy

            regimes_list.append(MarketRegime.UNKNOWN)
            strategies_list.append(TradingStrategy.AVOID)

    regimes = pd.Series(regimes_list, index=test_df.index)
    strategies = pd.Series(strategies_list, index=test_df.index)

    # Run backtest with optimized parameters
    params = params_data["parameters"]

    # Calculate position sizes
    from market_regime_analysis.enums import MarketRegime

    regime_multipliers = {}
    for regime in MarketRegime:
        key = f"regime_mult_{regime.value}"
        regime_multipliers[regime] = params.get(key, 0.5)

    position_sizes = regimes.map(regime_multipliers).fillna(0.0)

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=100000.0,
        max_position_size=params.get("max_position_size", 0.20),
        stop_loss_pct=params.get("stop_loss_pct", 0.10),
        take_profit_pct=params.get("take_profit_pct", None),
    )

    # Run backtest
    results = engine.run_regime_strategy(test_df, regimes, strategies, position_sizes)
    perf = results["performance"]

    # Print results
    click.echo(f"\n=== Out-of-Sample Validation Results ===")
    click.echo(f"Sharpe Ratio: {perf.sharpe_ratio:.3f}")
    click.echo(f"Total Return: {perf.total_return:.2%}")
    click.echo(f"Max Drawdown: {perf.max_drawdown:.2%}")
    click.echo(f"Calmar Ratio: {perf.calmar_ratio:.3f}")
    click.echo(f"Number of Trades: {len(results['trades'])}")
    click.echo(f"Win Rate: {perf.win_rate:.2%}")
    click.echo(f"Profit Factor: {perf.profit_factor:.3f}")

    click.echo(f"\n=== Comparison to In-Sample ===")
    in_sample_metrics = params_data["metrics"]
    click.echo(
        f"Sharpe Ratio: {in_sample_metrics['sharpe_ratio']:.3f} → {perf.sharpe_ratio:.3f}"
    )
    click.echo(
        f"Total Return: {in_sample_metrics['total_return']:.2%} → {perf.total_return:.2%}"
    )
    click.echo(
        f"Max Drawdown: {in_sample_metrics['max_drawdown']:.2%} → {perf.max_drawdown:.2%}"
    )

    # Calculate degradation
    sharpe_degradation = (
        (perf.sharpe_ratio - in_sample_metrics["sharpe_ratio"])
        / abs(in_sample_metrics["sharpe_ratio"])
        * 100
    )
    click.echo(f"\nSharpe Ratio Degradation: {sharpe_degradation:+.1f}%")

    if abs(sharpe_degradation) < 20:
        click.echo("✓ Performance degradation acceptable (<20%)")
    else:
        click.echo("✗ Warning: Significant performance degradation (>20%)")

    click.echo("\n=== Validation Complete ===\n")


if __name__ == "__main__":
    cli()
