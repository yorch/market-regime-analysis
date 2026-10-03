"""
Run strategy parameter optimization with walk-forward validation.

Usage:
    uv run mra-optimize [--mode grid|random|baseline] [--symbol SPY] [--provider yfinance]
                        [--holdout-frac 0.2] [--seed 42] [--output results.json]

This script:
1. Loads historical data (as much as possible)
2. Splits off the last ``--holdout-frac`` of bars as an untouched holdout
3. Runs grid or random search over strategy parameters on the earlier portion,
   evaluating each parameter set via walk-forward validation
4. Reports the best parameters' search-period (IN-SAMPLE) metrics and, separately,
   their performance on the holdout (OUT-OF-SAMPLE), plus the number of trials

Provider API keys are read from the environment (e.g. ``ALPHA_VANTAGE_API_KEY``,
``POLYGON_API_KEY``, ``TIINGO_API_KEY``, ``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY``).
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from typing import Any

import pandas as pd

from mra_lib.backtesting import (
    RegimeStrategy,
    StrategyOptimizer,
    WalkForwardValidator,
)
from mra_lib.data_providers import MarketDataProvider, required_env_vars, resolve_api_key

# Walk-forward settings shared by every mode
WF_SETTINGS: dict[str, Any] = {
    "n_hmm_states": 4,
    "hmm_n_iter": 50,
    "retrain_frequency": 20,
    "min_train_days": 252,
    "test_days": 63,
    "anchored": True,
}

GRID: dict[str, list] = {
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

RANGES: dict[str, tuple] = {
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


class _StdoutLogHandler(logging.Handler):
    """Write ``mra_lib`` log records to the *current* ``sys.stdout``.

    The library logs its search progress instead of printing; this keeps that
    progress interleaved with the report in this script's stdout. INFO records
    are written as bare messages, other levels with a ``LEVEL name:`` prefix.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno == logging.INFO:
                msg = record.getMessage()
            else:
                msg = f"{record.levelname} {record.name}: {record.getMessage()}"
            sys.stdout.write(msg + "\n")
        except Exception:  # noqa: BLE001 - logging must never raise; report via handleError
            self.handleError(record)


def configure_logging(verbose: bool) -> None:
    """Show library progress (INFO) when ``verbose``, otherwise only warnings. Idempotent."""
    lib_logger = logging.getLogger("mra_lib")
    for handler in [h for h in lib_logger.handlers if isinstance(h, _StdoutLogHandler)]:
        lib_logger.removeHandler(handler)
    lib_logger.addHandler(_StdoutLogHandler())
    lib_logger.setLevel(logging.INFO if verbose else logging.WARNING)


def load_data(symbol: str, provider_name: str, period: str = "5y") -> pd.DataFrame:
    """Load historical daily data, resolving the provider API key from the environment."""
    print(f"\nLoading {period} of {symbol} data via {provider_name}...")
    api_key = resolve_api_key(provider_name, None)
    if api_key is None:
        env_vars = " / ".join(required_env_vars(provider_name))
        raise ValueError(f"Provider '{provider_name}' requires an API key; set {env_vars}")
    kwargs: dict[str, Any] = {"api_key": api_key} if api_key else {}
    provider = MarketDataProvider.create_provider(provider_name, **kwargs)
    df = provider.fetch(symbol, period, "1d")
    print(f"  Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def _fmt_pf(pf: float) -> str:
    if math.isnan(pf):
        return "n/a"
    return "inf" if math.isinf(pf) else f"{pf:.2f}"


def _json_float(value: float) -> float | str | None:
    """JSON-safe float (inf/nan are not valid JSON numbers)."""
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def print_wf_results(results: dict, label: str) -> None:
    """Print a walk-forward result dict under a clear label."""
    print("\n" + "-" * 80)
    print(label)
    print("-" * 80)
    print(f"  Windows:              {results['n_windows']}")
    print(f"  Total Test Days:      {results['total_test_days']}")
    print(f"  Years Tested:         {results['years']:.2f}")
    print(f"\n  Strategy Return:      {results['compounded_strategy_return']:+.2%}")
    print(f"  Buy & Hold Return:    {results['compounded_bh_return']:+.2%}")
    print(f"  Excess Return:        {results['excess_return']:+.2%}")
    print(f"  Annualized Return:    {results['annualized_strategy_return']:+.2%}")
    print(f"  Annualized B&H:       {results['annualized_bh_return']:+.2%}")
    print(f"\n  Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:         {results['max_drawdown']:.2%}")
    print(f"  Total Trades:         {results['total_trades']}")
    print(f"  Trade Win Rate:       {results['trade_win_rate']:.1%}")
    print(f"  Profit Factor:        {_fmt_pf(results['profit_factor'])}")
    print(f"  Avg Win:              ${results['avg_win']:.2f}")
    print(f"  Avg Loss:             ${results['avg_loss']:.2f}")
    print(f"  Window Win Rate:      {results['window_win_rate']:.1%}")

    print("\n  Per-Window Returns:")
    for i, (sr, bhr) in enumerate(
        zip(results["per_window_returns"], results["per_window_bh_returns"], strict=True)
    ):
        marker = "+" if sr > bhr else "-"
        print(f"    Window {i + 1}: strategy={sr:+.2%}  b&h={bhr:+.2%}  [{marker}]")


def run_baseline(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run baseline (default parameters, nothing tuned) for comparison."""
    print("\n" + "=" * 100)
    print("BASELINE: Default Parameters (no parameter selection)")
    print("=" * 100)

    validator = WalkForwardValidator(strategy=RegimeStrategy(), **WF_SETTINGS)
    results = validator.run(df, verbose=verbose)
    if "error" in results:
        print(f"  Baseline failed: {results['error']}")
        return results

    print(f"\n  Compounded Return: {results['compounded_strategy_return']:+.2%}")
    print(f"  Buy & Hold Return: {results['compounded_bh_return']:+.2%}")
    print(f"  Excess Return:     {results['excess_return']:+.2%}")
    print(f"  Sharpe:            {results['sharpe_ratio']:.2f}")
    print(f"  Trade Win Rate:    {results['trade_win_rate']:.1%}")
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"  Max Drawdown:      {results['max_drawdown']:.2%}")
    return results


def _make_optimizer(df: pd.DataFrame, holdout_frac: float) -> StrategyOptimizer:
    optimizer = StrategyOptimizer(df=df, holdout_frac=holdout_frac, **WF_SETTINGS)
    summary = optimizer.search_summary()
    print(
        f"\nSearch period: {summary['search_bars']} bars"
        + (
            f" ({summary['search_start']} .. {summary['search_end']})"
            if "search_start" in summary
            else ""
        )
    )
    if optimizer.has_holdout:
        print(
            f"Holdout:       {summary['holdout_bars']} bars "
            f"({summary['holdout_start']} .. {summary['holdout_end']}) - untouched by search"
        )
    else:
        print("Holdout:       none (all reported results are in-sample)")
    return optimizer


def run_grid_search(
    df: pd.DataFrame, holdout_frac: float = 0.2, verbose: bool = True
) -> StrategyOptimizer:
    """Run grid search optimization on the search (non-holdout) period."""
    print("\n" + "=" * 100)
    print("GRID SEARCH OPTIMIZATION")
    print("=" * 100)

    optimizer = _make_optimizer(df, holdout_frac)
    optimizer.grid_search(param_grid=GRID, verbose=verbose)
    print(optimizer.format_top_results(n=15))
    return optimizer


def run_random_search(
    df: pd.DataFrame,
    n_iterations: int = 30,
    holdout_frac: float = 0.2,
    seed: int | None = None,
    verbose: bool = True,
) -> StrategyOptimizer:
    """Run random search optimization on the search (non-holdout) period."""
    print("\n" + "=" * 100)
    print("RANDOM SEARCH OPTIMIZATION")
    print("=" * 100)

    optimizer = _make_optimizer(df, holdout_frac)
    optimizer.random_search(
        param_ranges=RANGES, n_iterations=n_iterations, verbose=verbose, seed=seed
    )
    print(optimizer.format_top_results(n=15))
    return optimizer


def report_best(optimizer: StrategyOptimizer) -> dict | None:
    """Print in-sample and holdout results for the best params; return holdout results."""
    best = optimizer.results[0]
    print("\n" + "=" * 100)
    print("BEST PARAMETERS")
    print("=" * 100)
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(
        f"\nIN-SAMPLE (search period, selected from {optimizer.n_trials} trials - "
        "optimistic): "
        f"Sharpe={best.sharpe:.2f} Return={best.total_return:+.2%} "
        f"Excess={best.excess_return:+.2%} MaxDD={best.max_drawdown:.2%} "
        f"Trades={best.total_trades}"
    )

    if not optimizer.has_holdout:
        print("\nNo holdout requested (--holdout-frac 0): no out-of-sample estimate available.")
        return None

    holdout = optimizer.evaluate_holdout(best.params, verbose=True)
    if holdout is None or "error" in holdout:
        print("\nHoldout evaluation produced no valid windows (holdout too short?).")
        return None
    print_wf_results(holdout, "HOLDOUT (OUT-OF-SAMPLE) RESULTS - data never seen by the search")
    return holdout


def build_output(
    args: argparse.Namespace, optimizer: StrategyOptimizer, holdout: dict | None
) -> dict:
    """Assemble the JSON report for the best parameters."""
    best = optimizer.results[0]
    output: dict[str, Any] = {
        "symbol": args.symbol,
        "provider": args.provider,
        "mode": args.mode,
        "seed": args.seed,
        "date": datetime.now().isoformat(),
        "best_params": best.params,
        "search": optimizer.search_summary(),
        "in_sample": {
            "note": "Metrics on the data used to select parameters (optimistic)",
            "score": best.score,
            "sharpe": best.sharpe,
            "total_return": best.total_return,
            "excess_return": best.excess_return,
            "max_drawdown": best.max_drawdown,
            "profit_factor": _json_float(best.profit_factor),
            "total_trades": best.total_trades,
        },
        "holdout": None,
    }
    if holdout is not None:
        output["holdout"] = {
            "note": "Out-of-sample metrics on data withheld from the search",
            "sharpe": holdout["sharpe_ratio"],
            "total_return": holdout["compounded_strategy_return"],
            "buy_hold_return": holdout["compounded_bh_return"],
            "excess_return": holdout["excess_return"],
            "max_drawdown": holdout["max_drawdown"],
            "profit_factor": _json_float(holdout["profit_factor"]),
            "total_trades": holdout["total_trades"],
            "n_windows": holdout["n_windows"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize regime strategy parameters")
    parser.add_argument("--mode", choices=["grid", "random", "baseline"], default="grid")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--provider", default="yfinance")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--iterations", type=int, default=30, help="Random search iterations")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random search seed for reproducibility"
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction of the most recent bars withheld from the search and used for an "
        "out-of-sample evaluation of the best parameters (0 disables; default: 0.2)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output",
        default="optimization_results.json",
        help="Output file path for best parameters (default: optimization_results.json)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.holdout_frac < 1.0:
        parser.error("--holdout-frac must be in [0, 1)")

    verbose = not args.quiet
    configure_logging(verbose)

    print("=" * 100)
    print("MARKET REGIME STRATEGY OPTIMIZER")
    print(f"Symbol: {args.symbol} | Provider: {args.provider} | Mode: {args.mode}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    try:
        df = load_data(args.symbol, args.provider, args.period)
    except Exception as e:  # noqa: BLE001 - top-level CLI: report and exit non-zero
        print(f"Failed to load data: {e}")
        sys.exit(1)

    start = time.time()

    if args.mode == "baseline":
        run_baseline(df, verbose=verbose)
    else:
        run_baseline(df, verbose=False)
        try:
            if args.mode == "grid":
                optimizer = run_grid_search(df, holdout_frac=args.holdout_frac, verbose=verbose)
            else:
                optimizer = run_random_search(
                    df,
                    n_iterations=args.iterations,
                    holdout_frac=args.holdout_frac,
                    seed=args.seed,
                    verbose=verbose,
                )
        except ValueError as e:
            print(f"Optimization failed: {e}")
            sys.exit(1)

        if optimizer.results:
            holdout = report_best(optimizer)
            with open(args.output, "w") as f:
                json.dump(build_output(args, optimizer, holdout), f, indent=2, default=str)
            print(f"\nBest parameters saved to {args.output}")
        else:
            print(f"\nNo valid results ({optimizer.n_failures} failed trials).")

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
