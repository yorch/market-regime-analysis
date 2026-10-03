"""
Strategy parameter optimizer.

Grid search and random search over strategy parameters, evaluated via
walk-forward validation.

Evaluation honesty:

* Every parameter set is scored on walk-forward test windows, but the *choice*
  of the best set uses those same windows, so the reported best score is
  in-sample with respect to parameter selection (and optimistic in proportion
  to the number of trials, reported as ``n_trials``).
* With ``holdout_frac > 0`` the last fraction of the data is withheld from the
  search entirely. :meth:`StrategyOptimizer.evaluate_holdout` then runs the
  chosen parameters once on that untouched segment (the HMM still trains on all
  data before each test bar), which is the number to trust.

Performance: regime detection does not depend on strategy parameters, so the
walk-forward regimes for the search period are computed once
(:meth:`WalkForwardValidator.compute_regimes`) and reused for every trial.
"""

import itertools
import logging
import random
import time
from dataclasses import dataclass

import pandas as pd

from mra_lib._deprecation import write_deprecated_report
from mra_lib.config.regime_tables import TRADING_DAYS_PER_YEAR

from .strategy import RegimeStrategy
from .transaction_costs import TransactionCostModel
from .walk_forward import RegimeCache, WalkForwardValidator

logger = logging.getLogger(__name__)


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
        holdout_frac: float = 0.0,
        cost_model: TransactionCostModel | None = None,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
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
            retrain_frequency: Bars between HMM retrains
            holdout_frac: Fraction of bars (at the end of ``df``) withheld from
                the search for out-of-sample evaluation; 0 disables the holdout
            cost_model: Transaction cost model (default: equity costs)
            periods_per_year: Bars per year used for annualization
        """
        if not 0.0 <= holdout_frac < 1.0:
            raise ValueError("holdout_frac must be in [0, 1)")
        self.df = df
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.anchored = anchored
        self.initial_capital = initial_capital
        self.n_hmm_states = n_hmm_states
        self.hmm_n_iter = hmm_n_iter
        self.retrain_frequency = retrain_frequency
        self.holdout_frac = holdout_frac
        self.cost_model = cost_model
        self.periods_per_year = periods_per_year

        holdout_bars = round(len(df) * holdout_frac)
        self.split_index = len(df) - holdout_bars
        self.search_df = df.iloc[: self.split_index]

        self.results: list[OptimizationResult] = []
        self.n_trials = 0
        self.n_failures = 0
        self.n_skipped_duplicates = 0
        self._regime_cache: RegimeCache | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_validator(self, strategy: RegimeStrategy) -> WalkForwardValidator:
        return WalkForwardValidator(
            strategy=strategy,
            cost_model=self.cost_model,
            n_hmm_states=self.n_hmm_states,
            hmm_n_iter=self.hmm_n_iter,
            retrain_frequency=self.retrain_frequency,
            min_train_days=self.min_train_days,
            test_days=self.test_days,
            anchored=self.anchored,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year,
        )

    def _prepare_search(self, keys: list[str], verbose: bool) -> None:
        """Validate the search space and pre-compute search-period regimes once."""
        RegimeStrategy.validate_param_keys(keys)

        needed = self.min_train_days + self.test_days
        if len(self.search_df) < needed:
            raise ValueError(
                f"Search period has {len(self.search_df)} bars but walk-forward needs "
                f"at least {needed} (reduce holdout_frac or load more history)"
            )

        if self._regime_cache is None:
            start = time.time()
            validator = self._make_validator(RegimeStrategy())
            self._regime_cache = validator.compute_regimes(self.search_df)
            if verbose:
                logger.info(
                    "  Pre-computed regimes for %d windows (%d HMM fits) in %.1fs",
                    len(self._regime_cache.windows),
                    self._regime_cache.fit_count,
                    time.time() - start,
                )

        self.results = []
        self.n_trials = 0
        self.n_failures = 0
        self.n_skipped_duplicates = 0

    @staticmethod
    def _canonical_key(params: dict) -> tuple:
        return tuple(sorted(RegimeStrategy.canonical_params(params).items()))

    def _evaluate_params(self, params: dict, verbose: bool = False) -> OptimizationResult | None:
        """Evaluate a single parameter set via walk-forward on the search period."""
        self.n_trials += 1
        try:
            strategy = RegimeStrategy.from_param_vector(params)
            validator = self._make_validator(strategy)
            wf_results = validator.run(
                self.search_df, verbose=verbose, regime_cache=self._regime_cache
            )
        except Exception as e:
            self.n_failures += 1
            logger.warning(
                "Error evaluating params %s (%s): %s",
                self._format_params(params),
                type(e).__name__,
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return None

        if "error" in wf_results:
            self.n_failures += 1
            logger.warning(
                "No valid walk-forward windows for params %s", self._format_params(params)
            )
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def grid_search(
        self, param_grid: dict[str, list] | None = None, verbose: bool = True
    ) -> list[OptimizationResult]:
        """
        Exhaustive grid search over parameter combinations.

        Combinations that are equivalent (e.g. any ``bear_short`` when
        ``bear_mult == 0``) are evaluated only once.

        Args:
            param_grid: Dictionary mapping param names to lists of values.
                       If None, uses default grid.
            verbose: Log progress at INFO level (``mra_lib`` logger)

        Returns:
            Sorted list of OptimizationResult (best first)

        Raises:
            ValueError: On unknown parameter names or too little search data
        """
        if param_grid is None:
            param_grid = self._default_grid()

        keys = list(param_grid.keys())
        self._prepare_search(keys, verbose)

        seen: set[tuple] = set()
        combinations: list[dict] = []
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(keys, combo, strict=True))
            key = self._canonical_key(params)
            if key in seen:
                self.n_skipped_duplicates += 1
                continue
            seen.add(key)
            combinations.append(params)
        total = len(combinations)

        if verbose:
            logger.info(
                "Grid search: %d combinations (%d equivalent combinations skipped)",
                total,
                self.n_skipped_duplicates,
            )
            logger.info("Parameters: %s", keys)

        start = time.time()
        for idx, params in enumerate(combinations):
            if verbose:
                elapsed = time.time() - start
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (total - idx) / rate if rate > 0 else 0
                logger.info(
                    "  [%d/%d] ETA: %.0fs | params: %s",
                    idx + 1,
                    total,
                    eta,
                    self._format_params(params),
                )

            result = self._evaluate_params(params, verbose=False)
            if result is not None:
                self.results.append(result)
                if verbose:
                    logger.info(
                        "    -> Sharpe=%.2f Return=%+.2f%% Excess=%+.2f%% WinRate=%.0f%% "
                        "Trades=%d MaxDD=%.2f%%",
                        result.sharpe,
                        result.total_return * 100,
                        result.excess_return * 100,
                        result.trade_win_rate * 100,
                        result.total_trades,
                        result.max_drawdown * 100,
                    )

        self.results.sort(key=lambda r: r.score, reverse=True)

        if verbose:
            elapsed = time.time() - start
            logger.info("Grid search complete in %.1fs", elapsed)
            logger.info(
                "Valid results: %d/%d (trials=%d, failures=%d)",
                len(self.results),
                total,
                self.n_trials,
                self.n_failures,
            )

        return self.results

    def random_search(
        self,
        param_ranges: dict[str, tuple] | None = None,
        n_iterations: int = 50,
        verbose: bool = True,
        seed: int | None = None,
    ) -> list[OptimizationResult]:
        """
        Random search over parameter space.

        Args:
            param_ranges: Dict mapping param names to (min, max) tuples
            n_iterations: Number of random samples (duplicates of an already
                evaluated equivalent sample are skipped and counted)
            verbose: Log progress at INFO level (``mra_lib`` logger)
            seed: Seed for reproducible sampling (None = nondeterministic)

        Returns:
            Sorted list of OptimizationResult (best first)

        Raises:
            ValueError: On unknown parameter names or too little search data
        """
        if param_ranges is None:
            param_ranges = self._default_ranges()

        self._prepare_search(list(param_ranges), verbose)
        rng = random.Random(seed)

        if verbose:
            logger.info("Random search: %d iterations (seed=%s)", n_iterations, seed)

        seen: set[tuple] = set()
        start = time.time()

        for idx in range(n_iterations):
            params: dict[str, float | int | bool] = {}
            for key, (lo, hi) in param_ranges.items():
                if isinstance(lo, bool) or isinstance(hi, bool):
                    params[key] = rng.choice([True, False])
                elif isinstance(lo, int) and isinstance(hi, int):
                    params[key] = rng.randint(lo, hi)
                else:
                    # Round to 2 decimal places for cleaner params
                    params[key] = round(rng.uniform(lo, hi), 2)

            canon = self._canonical_key(params)
            if canon in seen:
                self.n_skipped_duplicates += 1
                continue
            seen.add(canon)

            if verbose:
                logger.info(
                    "  [%d/%d] params: %s", idx + 1, n_iterations, self._format_params(params)
                )

            result = self._evaluate_params(params, verbose=False)
            if result is not None:
                self.results.append(result)
                if verbose:
                    logger.info(
                        "    -> Sharpe=%.2f Return=%+.2f%% Excess=%+.2f%% Trades=%d",
                        result.sharpe,
                        result.total_return * 100,
                        result.excess_return * 100,
                        result.total_trades,
                    )

        self.results.sort(key=lambda r: r.score, reverse=True)

        if verbose:
            elapsed = time.time() - start
            logger.info("Random search complete in %.1fs", elapsed)
            logger.info(
                "Valid results: %d (trials=%d, failures=%d, duplicates skipped=%d)",
                len(self.results),
                self.n_trials,
                self.n_failures,
                self.n_skipped_duplicates,
            )

        return self.results

    # ------------------------------------------------------------------
    # Holdout evaluation and reporting
    # ------------------------------------------------------------------

    @property
    def has_holdout(self) -> bool:
        """True if a holdout segment was withheld from the search."""
        return self.split_index < len(self.df)

    def evaluate_holdout(self, params: dict, verbose: bool = False) -> dict | None:
        """
        Evaluate ``params`` once on the untouched holdout segment.

        Walk-forward windows start at the first holdout bar; the HMM trains on
        all data before each test bar (including the search period), but no
        holdout bar influenced parameter selection.

        Returns:
            Walk-forward result dict (see :meth:`WalkForwardValidator.run`), or
            None if there is no holdout segment.
        """
        if not self.has_holdout:
            return None
        validator = self._make_validator(RegimeStrategy.from_param_vector(params))
        return validator.run(self.df, verbose=verbose, start_index=self.split_index)

    def search_summary(self) -> dict:
        """Describe the search: trial counts and the search/holdout split."""
        summary: dict = {
            "n_trials": self.n_trials,
            "n_failures": self.n_failures,
            "n_skipped_duplicates": self.n_skipped_duplicates,
            "n_valid_results": len(self.results),
            "search_bars": len(self.search_df),
            "holdout_bars": len(self.df) - self.split_index,
            "holdout_frac": self.holdout_frac,
        }
        if len(self.search_df):
            summary["search_start"] = str(self.search_df.index[0])
            summary["search_end"] = str(self.search_df.index[-1])
        if self.has_holdout:
            summary["holdout_start"] = str(self.df.index[self.split_index])
            summary["holdout_end"] = str(self.df.index[-1])
        return summary

    def format_top_results(self, n: int = 10) -> str:
        """Format the top N results (in-sample with respect to parameter selection)."""
        lines: list[str] = []
        lines.append("\n" + "=" * 120)
        lines.append("TOP OPTIMIZATION RESULTS (IN-SAMPLE: ranked on the data used for selection)")
        lines.append("=" * 120)

        if not self.results:
            lines.append("No results to display.")
            return "\n".join(lines)
        lines.append(
            f"Trials: {self.n_trials} | failures: {self.n_failures} | "
            f"equivalent combinations skipped: {self.n_skipped_duplicates}"
        )
        lines.append(
            f"{'Rank':<5} {'Score':<8} {'Sharpe':<8} {'Return':<10} "
            f"{'Excess':<10} {'WinRate':<9} {'PF':<7} {'Trades':<8} "
            f"{'MaxDD':<9} {'WinWin':<8} {'Key Params'}"
        )
        lines.append("-" * 120)

        for i, r in enumerate(self.results[:n]):
            sl = r.params.get("stop_loss")
            bull = r.params.get("bull_mult")
            bear = r.params.get("bear_mult")
            base = r.params.get("base_fraction")
            key_params = (
                f"SL={f'{sl:.0%}' if sl is not None else '?'} "
                f"bull={f'{bull:.1f}' if bull is not None else '?'} "
                f"bear={f'{bear:.1f}' if bear is not None else '?'} "
                f"base={f'{base:.0%}' if base is not None else '?'}"
            )
            lines.append(
                f"{i + 1:<5} {r.score:<8.3f} {r.sharpe:<8.2f} "
                f"{r.total_return:<10.2%} {r.excess_return:<10.2%} "
                f"{r.trade_win_rate:<9.1%} {r.profit_factor:<7.2f} "
                f"{r.total_trades:<8} {r.max_drawdown:<9.2%} "
                f"{r.window_win_rate:<8.1%} {key_params}"
            )

        lines.append("=" * 120)

        best = self.results[0]
        lines.append("\nBEST PARAMETERS:")
        for k, v in sorted(best.params.items()):
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def print_top_results(self, n: int = 10) -> None:
        """
        Print the top N results to stdout.

        .. deprecated::
            The library no longer prints. Use :meth:`format_top_results` and print
            or log the returned string.
        """
        write_deprecated_report(
            self.format_top_results(n),
            "StrategyOptimizer.print_top_results",
            "format_top_results",
        )

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
