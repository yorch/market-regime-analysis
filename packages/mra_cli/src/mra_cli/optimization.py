"""
Run strategy parameter optimization with walk-forward validation.

Usage:
    uv run run_optimization.py [--mode grid|random] [--symbol SPY] [--provider yfinance]

This script:
1. Loads historical data (as much as possible)
2. Runs grid or random search over strategy parameters
3. Evaluates each parameter set via walk-forward validation
4. Reports the best parameter combinations
"""

import argparse
import json
import sys
import time
from datetime import datetime

from mra_lib.backtesting import (
    RegimeStrategy,
    StrategyOptimizer,
    WalkForwardValidator,
)
from mra_lib.data_providers import MarketDataProvider


def load_data(symbol: str, provider_name: str, period: str = "5y"):
    """Load historical data."""
    print(f"\nLoading {period} of {symbol} data via {provider_name}...")
    provider = MarketDataProvider.create_provider(provider_name)
    df = provider.fetch(symbol, period, "1d")
    print(f"  Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def run_baseline(df, verbose: bool = True):
    """Run baseline (default parameters) for comparison."""
    print("\n" + "=" * 100)
    print("BASELINE: Default Parameters")
    print("=" * 100)

    strategy = RegimeStrategy()  # All defaults

    validator = WalkForwardValidator(
        strategy=strategy,
        n_hmm_states=4,
        hmm_n_iter=50,
        retrain_frequency=20,
        min_train_days=252,
        test_days=63,
        anchored=True,
    )

    results = validator.run(df, verbose=verbose)

    print(f"\n  Compounded Return: {results['compounded_strategy_return']:+.2%}")
    print(f"  Buy & Hold Return: {results['compounded_bh_return']:+.2%}")
    print(f"  Excess Return:     {results['excess_return']:+.2%}")
    print(f"  Sharpe (approx):   {results['sharpe_approx']:.2f}")
    print(f"  Trade Win Rate:    {results['trade_win_rate']:.1%}")
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"  Max Drawdown:      {results['max_drawdown']:.2%}")

    return results


def run_grid_search(df, verbose: bool = True):
    """Run grid search optimization."""
    print("\n" + "=" * 100)
    print("GRID SEARCH OPTIMIZATION")
    print("=" * 100)

    # Focused grid — balances coverage with runtime
    param_grid = {
        "bull_mult": [1.0, 1.5, 2.0],
        "bear_mult": [0.0, 0.5, 1.0],
        "mr_mult": [0.8, 1.2],
        "lv_mult": [0.8, 1.2],
        "hv_mult": [0.0],
        "bo_mult": [0.5, 1.0],
        "base_fraction": [0.08, 0.12],
        "stop_loss": [0.03, 0.05, 0.08],
        "bear_short": [0, 1],
        "min_confidence": [0.0, 0.3],
    }

    optimizer = StrategyOptimizer(
        df=df,
        min_train_days=252,
        test_days=63,
        anchored=True,
        n_hmm_states=4,
        hmm_n_iter=50,
        retrain_frequency=20,
    )

    optimizer.grid_search(param_grid=param_grid, verbose=verbose)
    optimizer.print_top_results(n=15)
    return optimizer


def run_random_search(df, n_iterations: int = 30, verbose: bool = True):
    """Run random search optimization."""
    print("\n" + "=" * 100)
    print("RANDOM SEARCH OPTIMIZATION")
    print("=" * 100)

    param_ranges = {
        "bull_mult": (0.5, 2.5),
        "bear_mult": (0.0, 1.5),
        "mr_mult": (0.3, 2.0),
        "lv_mult": (0.3, 2.0),
        "hv_mult": (0.0, 0.3),
        "bo_mult": (0.3, 1.5),
        "base_fraction": (0.05, 0.20),
        "max_position": (0.10, 0.30),
        "stop_loss": (0.02, 0.10),
        "min_confidence": (0.0, 0.5),
        "bear_short": (0, 1),
    }

    optimizer = StrategyOptimizer(
        df=df,
        min_train_days=252,
        test_days=63,
        anchored=True,
        n_hmm_states=4,
        hmm_n_iter=50,
        retrain_frequency=20,
    )

    optimizer.random_search(
        param_ranges=param_ranges,
        n_iterations=n_iterations,
        verbose=verbose,
    )
    optimizer.print_top_results(n=15)
    return optimizer


def run_detailed_validation(df, params: dict):
    """Run detailed walk-forward validation on best params."""
    print("\n" + "=" * 100)
    print("DETAILED VALIDATION OF BEST PARAMETERS")
    print("=" * 100)

    strategy = RegimeStrategy.from_param_vector(params)

    print("\nStrategy Parameters:")
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    validator = WalkForwardValidator(
        strategy=strategy,
        n_hmm_states=4,
        hmm_n_iter=50,
        retrain_frequency=20,
        min_train_days=252,
        test_days=63,
        anchored=True,
    )

    results = validator.run(df, verbose=True)

    print("\n" + "-" * 80)
    print("DETAILED RESULTS")
    print("-" * 80)
    print(f"  Windows:              {results['n_windows']}")
    print(f"  Total Test Days:      {results['total_test_days']}")
    print(f"  Years Tested:         {results['years']:.2f}")
    print(f"\n  Strategy Return:      {results['compounded_strategy_return']:+.2%}")
    print(f"  Buy & Hold Return:    {results['compounded_bh_return']:+.2%}")
    print(f"  Excess Return:        {results['excess_return']:+.2%}")
    print(f"  Annualized Return:    {results['annualized_strategy_return']:+.2%}")
    print(f"  Annualized B&H:       {results['annualized_bh_return']:+.2%}")
    print(f"\n  Sharpe Ratio:         {results['sharpe_approx']:.2f}")
    print(f"  Max Drawdown:         {results['max_drawdown']:.2%}")
    print(f"  Total Trades:         {results['total_trades']}")
    print(f"  Trade Win Rate:       {results['trade_win_rate']:.1%}")
    print(f"  Profit Factor:        {results['profit_factor']:.2f}")
    print(f"  Avg Win:              ${results['avg_win']:.2f}")
    print(f"  Avg Loss:             ${results['avg_loss']:.2f}")
    print(f"  Window Win Rate:      {results['window_win_rate']:.1%}")

    print("\nPer-Window Returns:")
    for i, (sr, bhr) in enumerate(
        zip(results["per_window_returns"], results["per_window_bh_returns"], strict=True)
    ):
        marker = "+" if sr > bhr else "-"
        print(f"  Window {i + 1}: strategy={sr:+.2%}  b&h={bhr:+.2%}  [{marker}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize regime strategy parameters")
    parser.add_argument("--mode", choices=["grid", "random", "baseline"], default="grid")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--provider", default="yfinance")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--iterations", type=int, default=30, help="Random search iterations")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output",
        default="optimization_results.json",
        help="Output file path for best parameters (default: optimization_results.json)",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    print("=" * 100)
    print("MARKET REGIME STRATEGY OPTIMIZER")
    print(f"Symbol: {args.symbol} | Provider: {args.provider} | Mode: {args.mode}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # Load data
    try:
        df = load_data(args.symbol, args.provider, args.period)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    start = time.time()

    if args.mode == "baseline":
        run_baseline(df, verbose=verbose)
    elif args.mode == "grid":
        # Run baseline first
        run_baseline(df, verbose=False)

        # Run grid search
        optimizer = run_grid_search(df, verbose=verbose)

        # Detailed validation of best params
        if optimizer.results:
            best = optimizer.results[0]
            run_detailed_validation(df, best.params)

            # Save best params
            output = {
                "symbol": args.symbol,
                "date": datetime.now().isoformat(),
                "best_params": best.params,
                "score": best.score,
                "sharpe": best.sharpe,
                "total_return": best.total_return,
                "excess_return": best.excess_return,
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nBest parameters saved to {args.output}")
    elif args.mode == "random":
        run_baseline(df, verbose=False)
        optimizer = run_random_search(df, n_iterations=args.iterations, verbose=verbose)

        if optimizer.results:
            best = optimizer.results[0]
            run_detailed_validation(df, best.params)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
