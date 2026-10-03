"""
Walk-forward validation framework.

Implements anchored and rolling walk-forward analysis to test
regime-based strategies on truly out-of-sample data.

Regime detection depends only on the price data and the HMM settings, not on
strategy parameters, so :meth:`WalkForwardValidator.compute_regimes` can be run
once and the resulting cache reused across many strategy evaluations (see
:class:`~mra_lib.backtesting.optimizer.StrategyOptimizer`).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.exceptions import ConvergenceWarning

from mra_lib.config.enums import MarketRegime
from mra_lib.config.regime_tables import TRADING_DAYS_PER_YEAR
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector

from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .strategy import RegimeStrategy
from .trade_stats import compute_trade_stats
from .transaction_costs import EquityCostModel, TransactionCostModel

logger = logging.getLogger(__name__)

#: Minimum test window length (bars); shorter trailing windows are skipped.
MIN_TEST_BARS = 10

#: Minimum training rows needed before an HMM is fitted.
MIN_TRAIN_BARS = 60


@contextmanager
def _quiet_model_warnings() -> Iterator[None]:
    """Silence noisy HMM/sklearn warnings for the enclosed block only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


@dataclass
class RegimeCache:
    """
    Pre-computed walk-forward regime detections.

    Keyed by ``(train_end, test_end)`` window bounds (integer positions into the
    DataFrame the cache was computed on). Valid for any strategy parameters, as
    long as the data and HMM/window settings are unchanged; ``fingerprint``
    (see :meth:`WalkForwardValidator.cache_fingerprint`) identifies both.
    """

    fingerprint: str
    windows: dict[tuple[int, int], tuple[pd.Series, pd.Series]] = field(default_factory=dict)
    fit_count: int = 0
    refit_failures: int = 0

    def matches(self, fingerprint: str) -> bool:
        """Return True if this cache was computed for ``fingerprint``."""
        return self.fingerprint == fingerprint


class WalkForwardValidator:
    """
    Walk-forward validation for regime-based strategies.

    Splits data into train/test windows and runs the full pipeline:
    1. Train HMM on training window (refit every ``retrain_frequency`` bars)
    2. Predict regimes on test window using only past data
    3. Run backtest on test window
    4. Aggregate results across all windows on a stitched equity curve
    """

    def __init__(
        self,
        strategy: RegimeStrategy,
        cost_model: TransactionCostModel | None = None,
        n_hmm_states: int = 6,
        hmm_n_iter: int = 200,
        retrain_frequency: int = 20,
        min_train_days: int = 252,
        test_days: int = 63,
        anchored: bool = True,
        initial_capital: float = 100000.0,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = 0.02,
        hmm_n_init: int = 3,
    ) -> None:
        """
        Initialize walk-forward validator.

        Args:
            strategy: Trading strategy with parameters to test
            cost_model: Transaction cost model
            n_hmm_states: Number of HMM states
            hmm_n_iter: Max HMM training iterations
            retrain_frequency: Bars between HMM retrains
            min_train_days: Minimum training window size
            test_days: Test window size (quarter = 63 trading days)
            anchored: If True, training window grows from start.
                      If False, rolling window of fixed size.
            initial_capital: Starting capital for each window
            periods_per_year: Bars per year used for annualization
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
            hmm_n_init: EM restarts per HMM fit (best log-likelihood wins). Lower
                than the analyzer's default (10) because walk-forward refits every
                ``retrain_frequency`` bars; the model class, features, priors and
                labelling are otherwise identical
        """
        self.strategy = strategy
        self.cost_model = cost_model or EquityCostModel()
        self.n_hmm_states = n_hmm_states
        self.hmm_n_iter = hmm_n_iter
        self.retrain_frequency = max(1, retrain_frequency)
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.anchored = anchored
        self.initial_capital = initial_capital
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.hmm_n_init = hmm_n_init

    # ------------------------------------------------------------------
    # Window layout
    # ------------------------------------------------------------------

    def window_bounds(self, n_rows: int, start_index: int | None = None) -> list[tuple[int, int]]:
        """
        Return ``(train_end, test_end)`` bounds for every test window.

        Args:
            n_rows: Number of rows in the DataFrame
            start_index: First bar that may be used for testing. Defaults to
                ``min_train_days``; larger values are used for holdout evaluation.
        """
        windows = []
        train_end = max(self.min_train_days, start_index or 0)
        while train_end + MIN_TEST_BARS < n_rows:
            test_end = min(train_end + self.test_days, n_rows)
            windows.append((train_end, test_end))
            train_end = test_end
        return windows

    def cache_fingerprint(self, df: pd.DataFrame) -> str:
        """Hash of the price data and every setting that affects regime detection."""
        h = hashlib.sha256()
        h.update(
            repr(
                (
                    self.n_hmm_states,
                    self.hmm_n_iter,
                    self.hmm_n_init,
                    self.retrain_frequency,
                    self.min_train_days,
                    self.test_days,
                    self.anchored,
                    len(df),
                )
            ).encode()
        )
        if len(df):
            h.update(hash_pandas_object(df, index=True).to_numpy().tobytes())
        return h.hexdigest()

    def _train_slice(self, df: pd.DataFrame, i: int) -> pd.DataFrame:
        start = 0 if self.anchored else max(0, i - self.min_train_days)
        train: pd.DataFrame = df.iloc[start:i]
        return train

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _fit_detector(self, train_df: pd.DataFrame) -> TrueHMMDetector:
        # Same model class and defaults as MarketRegimeAnalyzer, so walk-forward
        # results describe the model users see
        hmm = TrueHMMDetector(
            n_states=self.n_hmm_states, n_iter=self.hmm_n_iter, n_init=self.hmm_n_init
        )
        with _quiet_model_warnings():
            hmm.fit(train_df)
        return hmm

    @staticmethod
    def _predict_one(hmm: TrueHMMDetector, df: pd.DataFrame, i: int) -> tuple[MarketRegime, float]:
        """Reference (slow) prediction for bar ``i`` using ``df[:i]``."""
        try:
            with _quiet_model_warnings():
                regime, _, conf = hmm.predict_regime(df.iloc[:i], use_viterbi=False)
            return regime, float(conf)
        except Exception:  # noqa: BLE001 - any prediction failure scores the bar UNKNOWN
            return MarketRegime.UNKNOWN, 0.0

    @staticmethod
    def _filtered_segment(
        hmm: TrueHMMDetector, df: pd.DataFrame, start: int, end: int
    ) -> list[tuple[MarketRegime, float]]:
        """
        Predict bars ``start..end-1`` with one fitted model in O(n).

        ``predict_regime(df[:i])`` returns the argmax of the filtered (forward-only)
        posterior at the last feature row of ``df[:i]``. Features are causal and
        filtering never looks ahead, so one forward pass over ``df[:end-1]``
        yields the same posterior for every prefix instead of recomputing
        features and posteriors per bar.
        """
        with _quiet_model_warnings():
            posteriors = hmm.filtered_posteriors(df.iloc[: end - 1])
        probs = posteriors.to_numpy()
        states = np.argmax(probs, axis=1)
        state_regime = hmm.get_state_regime_map()

        positions = df.index.get_indexer(posteriors.index)
        out: list[tuple[MarketRegime, float]] = []
        for i in range(start, end):
            # Last feature row at or before bar i-1
            k = int(np.searchsorted(positions, i - 1, side="right")) - 1
            if k < 0:
                out.append((MarketRegime.UNKNOWN, 0.0))
                continue
            state = int(states[k])
            out.append((state_regime[state], float(probs[k, state])))
        return out

    def _predict_segment(
        self, hmm: TrueHMMDetector, df: pd.DataFrame, start: int, end: int
    ) -> list[tuple[MarketRegime, float]]:
        """Predict a run of bars sharing one model, verifying the fast path."""
        try:
            fast = self._filtered_segment(hmm, df, start, end)
            # Cross-check first, middle and last bars against the detector's own
            # prediction so a change in its internals cannot silently skew results.
            for i in sorted({start, (start + end - 1) // 2, end - 1}):
                ref_regime, ref_conf = self._predict_one(hmm, df, i)
                fast_regime, fast_conf = fast[i - start]
                if fast_regime != ref_regime or abs(fast_conf - ref_conf) >= 1e-6:
                    logger.warning(
                        "Fast regime filter disagrees with detector at %s (%s/%.6f vs "
                        "%s/%.6f); falling back to per-bar prediction",
                        df.index[i],
                        fast_regime,
                        fast_conf,
                        ref_regime,
                        ref_conf,
                    )
                    break
            else:
                return fast
        except Exception as exc:  # noqa: BLE001 - fast path is optional (hmmlearn internals)
            logger.debug("Fast regime filter unavailable (%s); using per-bar prediction", exc)
        return [self._predict_one(hmm, df, i) for i in range(start, end)]

    def _detect_regimes_walk_forward(
        self,
        df: pd.DataFrame,
        train_end: int,
        test_end: int,
        cache: RegimeCache | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Detect regimes for the test window using only past data.

        The HMM is refit every ``retrain_frequency`` bars on data strictly before
        the bar being predicted. If a refit fails, the previously fitted model is
        kept (with a warning); bars before any successful fit are UNKNOWN.

        Returns:
            Tuple of (regimes, confidences) series aligned to test window.
        """
        results: list[tuple[MarketRegime, float]] = []
        hmm: TrueHMMDetector | None = None

        refit_points = list(range(train_end, test_end, self.retrain_frequency))
        refit_points.append(test_end)

        for seg_start, seg_end in itertools.pairwise(refit_points):
            train_df = self._train_slice(df, seg_start)
            if len(train_df) >= MIN_TRAIN_BARS:
                try:
                    new_hmm = self._fit_detector(train_df)
                    hmm = new_hmm
                    if cache is not None:
                        cache.fit_count += 1
                except Exception as exc:  # noqa: BLE001 - documented: keep the previous model
                    if cache is not None:
                        cache.refit_failures += 1
                    if hmm is not None:
                        logger.warning(
                            "HMM refit at %s failed (%s); keeping previous model",
                            df.index[seg_start],
                            exc,
                        )
                    else:
                        logger.warning(
                            "HMM fit at %s failed (%s); regimes UNKNOWN until next refit",
                            df.index[seg_start],
                            exc,
                        )

            if hmm is None:
                results.extend([(MarketRegime.UNKNOWN, 0.0)] * (seg_end - seg_start))
            else:
                results.extend(self._predict_segment(hmm, df, seg_start, seg_end))

        test_index = df.index[train_end:test_end]
        return (
            pd.Series([r for r, _ in results], index=test_index),
            pd.Series([c for _, c in results], index=test_index),
        )

    def compute_regimes(self, df: pd.DataFrame, start_index: int | None = None) -> RegimeCache:
        """
        Run regime detection for every walk-forward window once.

        The returned cache can be passed to :meth:`run` for any number of
        strategy parameter sets, so the HMM is fit once per refit point rather
        than once per parameter combination.
        """
        cache = RegimeCache(fingerprint=self.cache_fingerprint(df))
        for train_end, test_end in self.window_bounds(len(df), start_index):
            cache.windows[(train_end, test_end)] = self._detect_regimes_walk_forward(
                df, train_end, test_end, cache
            )
        return cache

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def _run_single_window(
        self,
        df: pd.DataFrame,
        train_end: int,
        test_end: int,
        regime_cache: RegimeCache | None = None,
    ) -> dict | None:
        """Run backtest on a single train/test window."""
        test_end = min(test_end, len(df))
        if test_end - train_end < MIN_TEST_BARS:
            return None

        cached = regime_cache.windows.get((train_end, test_end)) if regime_cache else None
        if cached is not None:
            regimes, confidences = cached
        else:
            regimes, confidences = self._detect_regimes_walk_forward(df, train_end, test_end)

        if len(regimes) == 0:
            return None

        strategies, directions, position_sizes = self.strategy.generate_signals(
            regimes, confidences
        )

        df_test = df.iloc[train_end:test_end].copy()

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
            max_position_size=self.strategy.max_position_size,
            stop_loss_pct=self.strategy.stop_loss_pct,
            take_profit_pct=self.strategy.take_profit_pct,
            periods_per_year=self.periods_per_year,
        )

        results = engine.run_regime_strategy(
            df=df_test,
            regimes=regimes,
            strategies=strategies,
            position_sizes=position_sizes,
            directions=directions,
        )

        # Buy-and-hold benchmark for same window
        bh_return = df_test["Close"].iloc[-1] / df_test["Close"].iloc[0] - 1

        return {
            "train_start": df.index[
                0 if self.anchored else max(0, train_end - self.min_train_days)
            ],
            "train_end": df.index[train_end - 1],
            "test_start": df.index[train_end],
            "test_end": df.index[test_end - 1],
            "test_days": test_end - train_end,
            "strategy_return": results["total_return"],
            "buy_hold_return": bh_return,
            "excess_return": results["total_return"] - bh_return,
            "trades": len(results["trades"]),
            "final_capital": results["final_capital"],
            "performance": results["performance"],
            "regime_distribution": regimes.value_counts().to_dict(),
        }

    def run(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        regime_cache: RegimeCache | None = None,
        start_index: int | None = None,
    ) -> dict:
        """
        Run full walk-forward validation.

        Args:
            df: Full OHLCV DataFrame
            verbose: Log progress at INFO level (``mra_lib`` logger)
            regime_cache: Optional pre-computed regimes from :meth:`compute_regimes`
                on the same ``df`` (ignored if it does not match ``df``)
            start_index: First bar eligible for testing (default ``min_train_days``).
                Bars before it are only used for training, e.g. to evaluate on a
                holdout segment at the end of the data.

        Returns:
            Dictionary with aggregated results
        """
        if len(df) < self.min_train_days + self.test_days:
            raise ValueError(
                f"Need at least {self.min_train_days + self.test_days} bars, got {len(df)}"
            )

        if regime_cache is not None and not regime_cache.matches(self.cache_fingerprint(df)):
            logger.warning("Regime cache does not match data; recomputing regimes")
            regime_cache = None

        windows = self.window_bounds(len(df), start_index)

        if verbose:
            logger.info(
                "  Walk-forward: %d windows, %d-day test periods", len(windows), self.test_days
            )

        window_results = []
        for idx, (te, tend) in enumerate(windows):
            result = self._run_single_window(df, te, tend, regime_cache)
            if result is not None:
                window_results.append(result)
                if verbose:
                    logger.info(
                        "    Window %d/%d: strategy=%+.2f%% b&h=%+.2f%% trades=%d",
                        idx + 1,
                        len(windows),
                        result["strategy_return"] * 100,
                        result["buy_hold_return"] * 100,
                        result["trades"],
                    )

        if not window_results:
            return {"error": "No valid windows"}

        return self._aggregate_results(window_results, df)

    def _stitch_equity(self, window_results: list[dict]) -> pd.Series:
        """Chain per-window equity curves into one continuous curve."""
        pieces = []
        level = self.initial_capital
        for w in window_results:
            perf = w["performance"]
            eq = perf.equity_curve.astype(float)
            if len(eq) == 0:
                continue
            base = perf.initial_capital if perf.initial_capital is not None else eq.iloc[0]
            if base <= 0:
                continue
            scaled = eq * (level / base)
            pieces.append(scaled)
            level = float(scaled.iloc[-1])
        if not pieces:
            return pd.Series(dtype=float)
        stitched: pd.Series = pd.concat(pieces)
        return stitched

    def _aggregate_results(self, window_results: list[dict], df: pd.DataFrame) -> dict:
        """
        Aggregate results across all walk-forward windows.

        Sharpe, Sortino and max drawdown are computed on the stitched per-bar
        equity curve, so partial (shorter) final windows are weighted by their
        actual length and drawdowns spanning window boundaries are captured.
        """
        strategy_returns = [w["strategy_return"] for w in window_results]
        bh_returns = [w["buy_hold_return"] for w in window_results]
        all_trades = []
        for w in window_results:
            all_trades.extend(w["performance"].trades)

        total_trades = sum(w["trades"] for w in window_results)
        n_windows = len(window_results)
        total_test_days = sum(w["test_days"] for w in window_results)
        years = total_test_days / self.periods_per_year

        compounded_strategy = float(np.prod([1 + r for r in strategy_returns]))
        compounded_bh = float(np.prod([1 + r for r in bh_returns]))
        compounded_strategy_return = compounded_strategy - 1
        compounded_bh_return = compounded_bh - 1

        stitched = self._stitch_equity(window_results)
        stitched_perf = PerformanceMetrics(
            [],
            stitched,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            initial_capital=self.initial_capital,
        )
        sm = stitched_perf.metrics

        trade_stats = compute_trade_stats(all_trades)
        winning_windows = sum(1 for r in strategy_returns if r > 0)
        per_window_dd = [w["performance"].metrics.get("max_drawdown", 0.0) for w in window_results]

        def _annualize(total: float) -> float:
            if years <= 0:
                return 0.0
            if total <= -1:
                return -1.0
            return float((1 + total) ** (1 / years) - 1)

        return {
            "n_windows": n_windows,
            "total_test_days": total_test_days,
            "years": years,
            "compounded_strategy_return": compounded_strategy_return,
            "compounded_bh_return": compounded_bh_return,
            "excess_return": compounded_strategy_return - compounded_bh_return,
            "annualized_strategy_return": _annualize(compounded_strategy_return),
            "annualized_bh_return": _annualize(compounded_bh_return),
            # Standard per-bar Sharpe of the stitched curve (key kept for compatibility)
            "sharpe_approx": sm["sharpe_ratio"],
            "sharpe_ratio": sm["sharpe_ratio"],
            "sortino_ratio": sm["sortino_ratio"],
            "calmar_ratio": sm["calmar_ratio"],
            "total_trades": total_trades,
            "trade_win_rate": trade_stats["win_rate"],
            "avg_win": trade_stats["avg_win"],
            "avg_loss": trade_stats["avg_loss"],
            "profit_factor": trade_stats["profit_factor"],
            "winning_windows": winning_windows,
            "window_win_rate": winning_windows / n_windows if n_windows > 0 else 0,
            "max_drawdown": sm["max_drawdown"],
            "max_drawdown_duration": sm["max_drawdown_duration"],
            "worst_window_drawdown": min(per_window_dd) if per_window_dd else 0.0,
            "avg_window_return": float(np.mean(strategy_returns)),
            "std_window_return": float(np.std(strategy_returns)),
            "per_window_returns": strategy_returns,
            "per_window_bh_returns": bh_returns,
            "stitched_equity_curve": stitched,
            "window_results": window_results,
        }
