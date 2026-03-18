"""
Strategy parameter optimizer.

Grid search and random search over strategy parameters,
evaluated via walk-forward validation.
"""

import itertools
import random
import time
import warnings
from dataclasses import dataclass

import pandas as pd

from .strategy import RegimeStrategy
from .walk_forward import WalkForwardValidator

warnings.filterwarnings("ignore")


@dataclass
class OptimizationResult:
    """Result from a single parameter evaluation."""

    params: dict
    sharpe: float
    total_return: float
    excess_return: float
    trade_win_rate: float
    profit_factor: float
    total_trades: int
    max_drawdown: float
    window_win_rate: float

    @property
    def score(self) -> float:
        """
        Composite score for ranking.

        Prioritizes:
        1. Sharpe ratio (risk-adjusted returns)
        2. Positive excess return over buy-and-hold
        3. Reasonable trade count (not too few)
        4. Controlled drawdown
        """
        # Penalize negative Sharpe heavily
        sharpe_component = self.sharpe * 2.0

        # Reward excess return
        excess_component = self.excess_return * 5.0

        # Penalize extreme drawdowns
        dd_penalty = min(0, self.max_drawdown + 0.20) * 3.0  # Penalize DD > 20%

        # Penalize too few trades (not statistically significant)
        trade_penalty = -0.5 if self.total_trades < 15 else 0.0

        return sharpe_component + excess_component + dd_penalty + trade_penalty


class StrategyOptimizer:
    """
    Optimize strategy parameters via grid/random search with walk-forward validation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        min_train_days: int = 252,
        test_days: int = 63,
        anchored: bool = True,
        initial_capital: float = 100000.0,
        n_hmm_states: int = 4,
        hmm_n_iter: int = 50,
        retrain_frequency: int = 20,
    ) -> None:
        """
        Initialize optimizer.

        Args:
            df: Full OHLCV DataFrame (as much history as possible)
            min_train_days: Minimum training period
            test_days: Test window size
            anchored: Anchored walk-forward
            initial_capital: Starting capital
            n_hmm_states: Number of HMM states to use
            hmm_n_iter: HMM training iterations
            retrain_frequency: Days between HMM retrains
        """
        self.df = df
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.anchored = anchored
        self.initial_capital = initial_capital
        self.n_hmm_states = n_hmm_states
        self.hmm_n_iter = hmm_n_iter
        self.retrain_frequency = retrain_frequency
        self.results: list[OptimizationResult] = []

    def _evaluate_params(self, params: dict, verbose: bool = False) -> OptimizationResult | None:
        """Evaluate a single parameter set via walk-forward validation."""
        try:
            strategy = RegimeStrategy.from_param_vector(params)

            validator = WalkForwardValidator(
                strategy=strategy,
                n_hmm_states=self.n_hmm_states,
                hmm_n_iter=self.hmm_n_iter,
                retrain_frequency=self.retrain_frequency,
                min_train_days=self.min_train_days,
                test_days=self.test_days,
                anchored=self.anchored,
                initial_capital=self.initial_capital,
            )

            wf_results = validator.run(self.df, verbose=verbose)

            if "error" in wf_results:
                return None

            return OptimizationResult(
                params=params,
                sharpe=wf_results["sharpe_approx"],
                total_return=wf_results["compounded_strategy_return"],
                excess_return=wf_results["excess_return"],
                trade_win_rate=wf_results["trade_win_rate"],
                profit_factor=wf_results["profit_factor"],
                total_trades=wf_results["total_trades"],
                max_drawdown=wf_results["max_drawdown"],
                window_win_rate=wf_results["window_win_rate"],
            )
        except Exception as e:
            if verbose:
                print(f"    Error evaluating params: {e}")
            return None

    def grid_search(
        self, param_grid: dict[str, list] | None = None, verbose: bool = True
    ) -> list[OptimizationResult]:
        """
        Exhaustive grid search over parameter combinations.

        Args:
            param_grid: Dictionary mapping param names to lists of values.
                       If None, uses default grid.
            verbose: Print progress

        Returns:
            Sorted list of OptimizationResult (best first)
        """
        if param_grid is None:
            param_grid = self._default_grid()

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        total = len(combinations)

        if verbose:
            print(f"\nGrid search: {total} combinations")
            print(f"Parameters: {keys}")

        self.results = []
        start = time.time()

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo, strict=True))

            if verbose:
                elapsed = time.time() - start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total - idx - 1) / rate if rate > 0 else 0
                print(
                    f"\n  [{idx + 1}/{total}] "
                    f"ETA: {eta:.0f}s | "
                    f"params: {self._format_params(params)}"
                )

            result = self._evaluate_params(params, verbose=verbose)
            if result is not None:
                self.results.append(result)
                if verbose:
                    print(
                        f"    -> Sharpe={result.sharpe:.2f} "
                        f"Return={result.total_return:+.2%} "
                        f"Excess={result.excess_return:+.2%} "
                        f"WinRate={result.trade_win_rate:.0%} "
                        f"Trades={result.total_trades} "
                        f"MaxDD={result.max_drawdown:.2%}"
                    )

        # Sort by composite score
        self.results.sort(key=lambda r: r.score, reverse=True)

        if verbose:
            elapsed = time.time() - start
            print(f"\nGrid search complete in {elapsed:.1f}s")
            print(f"Valid results: {len(self.results)}/{total}")

        return self.results

    def random_search(
        self,
        param_ranges: dict[str, tuple] | None = None,
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> list[OptimizationResult]:
        """
        Random search over parameter space.

        Args:
            param_ranges: Dict mapping param names to (min, max) tuples
            n_iterations: Number of random samples
            verbose: Print progress

        Returns:
            Sorted list of OptimizationResult (best first)
        """
        if param_ranges is None:
            param_ranges = self._default_ranges()

        if verbose:
            print(f"\nRandom search: {n_iterations} iterations")

        self.results = []
        start = time.time()

        for idx in range(n_iterations):
            params = {}
            for key, (lo, hi) in param_ranges.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[key] = random.randint(lo, hi)
                elif isinstance(lo, bool):
                    params[key] = random.choice([True, False])
                else:
                    # Round to 2 decimal places for cleaner params
                    params[key] = round(random.uniform(lo, hi), 2)

            if verbose:
                print(f"\n  [{idx + 1}/{n_iterations}] params: {self._format_params(params)}")

            result = self._evaluate_params(params, verbose=verbose)
            if result is not None:
                self.results.append(result)
                if verbose:
                    print(
                        f"    -> Sharpe={result.sharpe:.2f} "
                        f"Return={result.total_return:+.2%} "
                        f"Excess={result.excess_return:+.2%} "
                        f"Trades={result.total_trades}"
                    )

        self.results.sort(key=lambda r: r.score, reverse=True)

        if verbose:
            elapsed = time.time() - start
            print(f"\nRandom search complete in {elapsed:.1f}s")

        return self.results

    def print_top_results(self, n: int = 10) -> None:
        """Print top N results."""
        print("\n" + "=" * 120)
        print("TOP OPTIMIZATION RESULTS")
        print("=" * 120)

        if not self.results:
            print("No results to display.")
            return

        print(
            f"{'Rank':<5} {'Score':<8} {'Sharpe':<8} {'Return':<10} "
            f"{'Excess':<10} {'WinRate':<9} {'PF':<7} {'Trades':<8} "
            f"{'MaxDD':<9} {'WinWin':<8} {'Key Params'}"
        )
        print("-" * 120)

        for i, r in enumerate(self.results[:n]):
            key_params = (
                f"SL={r.params.get('stop_loss', '?'):.0%} "
                f"bull={r.params.get('bull_mult', '?'):.1f} "
                f"bear={r.params.get('bear_mult', '?'):.1f} "
                f"base={r.params.get('base_fraction', '?'):.0%}"
            )
            print(
                f"{i + 1:<5} {r.score:<8.3f} {r.sharpe:<8.2f} "
                f"{r.total_return:<10.2%} {r.excess_return:<10.2%} "
                f"{r.trade_win_rate:<9.1%} {r.profit_factor:<7.2f} "
                f"{r.total_trades:<8} {r.max_drawdown:<9.2%} "
                f"{r.window_win_rate:<8.1%} {key_params}"
            )

        print("=" * 120)

        # Print best params in detail
        if self.results:
            best = self.results[0]
            print("\nBEST PARAMETERS:")
            for k, v in sorted(best.params.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    def _default_grid(self) -> dict[str, list]:
        """Default parameter grid — focused and practical."""
        return {
            "bull_mult": [1.0, 1.5, 2.0],
            "bear_mult": [0.0, 0.5, 1.0],
            "mr_mult": [0.8, 1.2, 1.5],
            "lv_mult": [0.8, 1.0, 1.5],
            "hv_mult": [0.0],
            "bo_mult": [0.5, 1.0],
            "base_fraction": [0.08, 0.12, 0.15],
            "stop_loss": [0.03, 0.05, 0.08],
            "bear_short": [0, 1],
            "min_confidence": [0.0, 0.3],
        }

    def _default_ranges(self) -> dict[str, tuple]:
        """Default parameter ranges for random search."""
        return {
            "bull_mult": (0.5, 2.5),
            "bear_mult": (0.0, 1.5),
            "mr_mult": (0.5, 2.0),
            "lv_mult": (0.5, 2.0),
            "hv_mult": (0.0, 0.5),
            "bo_mult": (0.3, 1.5),
            "base_fraction": (0.05, 0.20),
            "max_position": (0.10, 0.30),
            "stop_loss": (0.02, 0.10),
            "min_confidence": (0.0, 0.5),
            "bear_short": (0, 1),
        }

    @staticmethod
    def _format_params(params: dict) -> str:
        """Format params for display."""
        parts = []
        for k, v in params.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)
