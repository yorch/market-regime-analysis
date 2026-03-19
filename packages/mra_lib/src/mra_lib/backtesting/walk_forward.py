"""
Walk-forward validation framework.

Implements anchored and rolling walk-forward analysis to test
regime-based strategies on truly out-of-sample data.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from mra_lib.config.enums import MarketRegime
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector

from .engine import BacktestEngine
from .strategy import RegimeStrategy
from .transaction_costs import EquityCostModel, TransactionCostModel

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class WalkForwardValidator:
    """
    Walk-forward validation for regime-based strategies.

    Splits data into train/test windows and runs the full pipeline:
    1. Train HMM on training window
    2. Predict regimes on test window using only past data
    3. Run backtest on test window
    4. Aggregate results across all windows
    """

    def __init__(
        self,
        strategy: RegimeStrategy,
        cost_model: TransactionCostModel | None = None,
        n_hmm_states: int = 6,
        hmm_n_iter: int = 100,
        retrain_frequency: int = 20,
        min_train_days: int = 252,
        test_days: int = 63,
        anchored: bool = True,
        initial_capital: float = 100000.0,
    ) -> None:
        """
        Initialize walk-forward validator.

        Args:
            strategy: Trading strategy with parameters to test
            cost_model: Transaction cost model
            n_hmm_states: Number of HMM states
            hmm_n_iter: Max HMM training iterations
            retrain_frequency: Days between HMM retrains
            min_train_days: Minimum training window size
            test_days: Test window size (quarter = 63 trading days)
            anchored: If True, training window grows from start.
                      If False, rolling window of fixed size.
            initial_capital: Starting capital for each window
        """
        self.strategy = strategy
        self.cost_model = cost_model or EquityCostModel()
        self.n_hmm_states = n_hmm_states
        self.hmm_n_iter = hmm_n_iter
        self.retrain_frequency = retrain_frequency
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.anchored = anchored
        self.initial_capital = initial_capital

    def _detect_regimes_walk_forward(
        self, df: pd.DataFrame, train_end: int, test_end: int
    ) -> tuple[pd.Series, pd.Series]:
        """
        Detect regimes for the test window using only past data.

        Returns:
            Tuple of (regimes, confidences) series aligned to test window.
        """
        regime_list = []
        confidence_list = []

        hmm = None
        for i in range(train_end, test_end):
            days_since_train_end = i - train_end

            # Retrain periodically
            if hmm is None or days_since_train_end % self.retrain_frequency == 0:
                if self.anchored:
                    train_df = df.iloc[:i]
                else:
                    start = max(0, i - self.min_train_days)
                    train_df = df.iloc[start:i]

                if len(train_df) < 60:
                    regime_list.append(MarketRegime.UNKNOWN)
                    confidence_list.append(0.0)
                    continue

                try:
                    hmm = TrueHMMDetector(n_states=self.n_hmm_states, n_iter=self.hmm_n_iter)
                    hmm.fit(train_df)
                except Exception:
                    regime_list.append(MarketRegime.UNKNOWN)
                    confidence_list.append(0.0)
                    continue

            # Predict using data up to (not including) current day
            predict_df = df.iloc[:i]
            try:
                regime, _, conf = hmm.predict_regime(predict_df, use_viterbi=False)
            except Exception:
                regime = MarketRegime.UNKNOWN
                conf = 0.0

            regime_list.append(regime)
            confidence_list.append(conf)

        test_index = df.index[train_end:test_end]
        return (
            pd.Series(regime_list, index=test_index),
            pd.Series(confidence_list, index=test_index),
        )

    def _run_single_window(self, df: pd.DataFrame, train_end: int, test_end: int) -> dict | None:
        """Run backtest on a single train/test window."""
        test_end = min(test_end, len(df))
        if test_end - train_end < 10:
            return None

        # Detect regimes
        regimes, confidences = self._detect_regimes_walk_forward(df, train_end, test_end)

        if len(regimes) == 0:
            return None

        # Generate strategy signals
        strategies, directions, position_sizes = self.strategy.generate_signals(
            regimes, confidences
        )

        # Slice price data for test window
        df_test = df.iloc[train_end:test_end].copy()

        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
            max_position_size=self.strategy.max_position_size,
            stop_loss_pct=self.strategy.stop_loss_pct,
            take_profit_pct=self.strategy.take_profit_pct,
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

    def run(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Run full walk-forward validation.

        Args:
            df: Full OHLCV DataFrame
            verbose: Print progress

        Returns:
            Dictionary with aggregated results
        """
        if len(df) < self.min_train_days + self.test_days:
            raise ValueError(
                f"Need at least {self.min_train_days + self.test_days} bars, got {len(df)}"
            )

        windows = []
        train_end = self.min_train_days

        while train_end + 10 < len(df):
            test_end = min(train_end + self.test_days, len(df))
            windows.append((train_end, test_end))
            train_end = test_end

        if verbose:
            print(f"  Walk-forward: {len(windows)} windows, {self.test_days}-day test periods")

        # Run each window
        window_results = []
        for idx, (te, tend) in enumerate(windows):
            result = self._run_single_window(df, te, tend)
            if result is not None:
                window_results.append(result)
                if verbose:
                    print(
                        f"    Window {idx + 1}/{len(windows)}: "
                        f"strategy={result['strategy_return']:+.2%} "
                        f"b&h={result['buy_hold_return']:+.2%} "
                        f"trades={result['trades']}"
                    )

        if not window_results:
            return {"error": "No valid windows"}

        return self._aggregate_results(window_results, df)

    def _aggregate_results(self, window_results: list[dict], df: pd.DataFrame) -> dict:
        """Aggregate results across all walk-forward windows."""
        strategy_returns = [w["strategy_return"] for w in window_results]
        bh_returns = [w["buy_hold_return"] for w in window_results]
        all_trades = []
        for w in window_results:
            all_trades.extend(w["performance"].trades)

        total_trades = sum(w["trades"] for w in window_results)

        # Compound returns across windows
        compounded_strategy = 1.0
        compounded_bh = 1.0
        for sr, bhr in zip(strategy_returns, bh_returns, strict=True):
            compounded_strategy *= 1 + sr
            compounded_bh *= 1 + bhr

        compounded_strategy_return = compounded_strategy - 1
        compounded_bh_return = compounded_bh - 1

        # Calculate aggregate metrics from all trades
        n_windows = len(window_results)
        years = sum(w["test_days"] for w in window_results) / 252

        # Per-window Sharpe approximation
        if len(strategy_returns) > 1:
            window_excess = np.array(strategy_returns) - 0.02 / (252 / self.test_days)
            sharpe_approx = (
                np.mean(window_excess) / np.std(window_excess) * np.sqrt(252 / self.test_days)
                if np.std(window_excess) > 0
                else 0.0
            )
        else:
            sharpe_approx = 0.0

        # Win rate across windows
        winning_windows = sum(1 for r in strategy_returns if r > 0)

        # Trade-level stats
        if all_trades:
            trade_pnls = [t["pnl"] for t in all_trades]
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p < 0]
            trade_win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
            avg_win = float(np.mean(winning_trades)) if winning_trades else 0
            avg_loss = float(np.mean([abs(t) for t in losing_trades])) if losing_trades else 0
            total_gross_wins = sum(winning_trades) if winning_trades else 0
            total_gross_losses = sum(abs(t) for t in losing_trades) if losing_trades else 0
            profit_factor = total_gross_wins / total_gross_losses if total_gross_losses > 0 else 0
        else:
            trade_win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Max drawdown across windows
        max_dd_per_window = []
        for w in window_results:
            dd = w["performance"].metrics.get("max_drawdown", 0)
            max_dd_per_window.append(dd)

        return {
            "n_windows": n_windows,
            "total_test_days": sum(w["test_days"] for w in window_results),
            "years": years,
            "compounded_strategy_return": compounded_strategy_return,
            "compounded_bh_return": compounded_bh_return,
            "excess_return": compounded_strategy_return - compounded_bh_return,
            "annualized_strategy_return": (
                (1 + compounded_strategy_return) ** (1 / years) - 1 if years > 0 else 0
            ),
            "annualized_bh_return": (
                (1 + compounded_bh_return) ** (1 / years) - 1 if years > 0 else 0
            ),
            "sharpe_approx": sharpe_approx,
            "total_trades": total_trades,
            "trade_win_rate": trade_win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "winning_windows": winning_windows,
            "window_win_rate": winning_windows / n_windows if n_windows > 0 else 0,
            "max_drawdown": min(max_dd_per_window) if max_dd_per_window else 0,
            "avg_window_return": float(np.mean(strategy_returns)),
            "std_window_return": float(np.std(strategy_returns)),
            "per_window_returns": strategy_returns,
            "per_window_bh_returns": bh_returns,
            "window_results": window_results,
        }
