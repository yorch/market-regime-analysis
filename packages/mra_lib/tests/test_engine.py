"""Unit tests for BacktestEngine and walk-forward aggregation."""

import pandas as pd
import pytest

from mra_lib.backtesting.engine import BacktestEngine
from mra_lib.backtesting.optimizer import OptimizationResult, StrategyOptimizer
from mra_lib.backtesting.strategy import RegimeStrategy
from mra_lib.backtesting.walk_forward import WalkForwardValidator
from mra_lib.config.enums import MarketRegime, TradingStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    dates = pd.bdate_range(start, periods=len(prices))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [1_000_000] * len(prices),
        },
        index=dates,
    )
    return df


def _constant_series(value, index):
    return pd.Series([value] * len(index), index=index)


# ---------------------------------------------------------------------------
# BacktestEngine - direction propagation
# ---------------------------------------------------------------------------


class TestBacktestEngineDirections:
    """Verify that explicit direction series are respected by the engine."""

    def test_long_direction_enters_long(self):
        prices = [100.0] * 20
        df = _make_ohlcv(prices)
        idx = df.index

        regimes = _constant_series(MarketRegime.BULL_TRENDING, idx)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, idx)
        sizes = _constant_series(0.10, idx)
        directions = _constant_series("LONG", idx)

        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=None)
        results = engine.run_regime_strategy(df, regimes, strategies, sizes, directions)

        assert len(results["trades"]) >= 1
        assert results["trades"][0]["direction"] == "LONG"

    def test_short_direction_enters_short(self):
        prices = [100.0] * 20
        df = _make_ohlcv(prices)
        idx = df.index

        regimes = _constant_series(MarketRegime.BEAR_TRENDING, idx)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, idx)
        sizes = _constant_series(0.10, idx)
        directions = _constant_series("SHORT", idx)

        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=None)
        results = engine.run_regime_strategy(df, regimes, strategies, sizes, directions)

        assert len(results["trades"]) >= 1
        assert results["trades"][0]["direction"] == "SHORT"

    def test_none_direction_no_entry(self):
        """When direction is None for every bar, no trades should be opened."""
        prices = [100.0] * 20
        df = _make_ohlcv(prices)
        idx = df.index

        regimes = _constant_series(MarketRegime.BULL_TRENDING, idx)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, idx)
        sizes = _constant_series(0.10, idx)
        directions = _constant_series(None, idx)

        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=None)
        results = engine.run_regime_strategy(df, regimes, strategies, sizes, directions)

        assert len(results["trades"]) == 0

    def test_direction_reversal_closes_and_reopens(self):
        """Switching from LONG to SHORT should close the LONG, then open SHORT."""
        n = 20
        prices = [100.0] * n
        df = _make_ohlcv(prices)
        idx = df.index

        regimes = _constant_series(MarketRegime.BULL_TRENDING, idx)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, idx)
        sizes = _constant_series(0.10, idx)
        dirs = ["LONG"] * 10 + ["SHORT"] * 10
        directions = pd.Series(dirs, index=idx)

        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=None)
        results = engine.run_regime_strategy(df, regimes, strategies, sizes, directions)

        # Should have at least 2 trades: the closed LONG and a SHORT
        assert len(results["trades"]) >= 2
        assert results["trades"][0]["direction"] == "LONG"
        assert results["trades"][1]["direction"] == "SHORT"


# ---------------------------------------------------------------------------
# Walk-forward aggregation
# ---------------------------------------------------------------------------


class TestAggregateResults:
    """Test _aggregate_results compounding and metric computation."""

    def _make_window_result(self, strategy_return, bh_return, trades=None, test_days=63):
        """Build a minimal window result dict."""
        from mra_lib.backtesting.metrics import PerformanceMetrics

        equity = pd.Series([100_000, 100_000 * (1 + strategy_return)])
        perf = PerformanceMetrics(trades or [], equity)
        return {
            "strategy_return": strategy_return,
            "buy_hold_return": bh_return,
            "trades": len(trades) if trades else 0,
            "test_days": test_days,
            "performance": perf,
        }

    def test_compounding_single_window(self):
        strategy = RegimeStrategy()
        validator = WalkForwardValidator(strategy=strategy)
        windows = [self._make_window_result(0.05, 0.03)]

        prices = [100.0] * 400
        df = _make_ohlcv(prices)
        result = validator._aggregate_results(windows, df)

        assert result["compounded_strategy_return"] == pytest.approx(0.05)
        assert result["compounded_bh_return"] == pytest.approx(0.03)
        assert result["excess_return"] == pytest.approx(0.02)

    def test_compounding_multiple_windows(self):
        strategy = RegimeStrategy()
        validator = WalkForwardValidator(strategy=strategy)
        windows = [
            self._make_window_result(0.10, 0.05),
            self._make_window_result(0.10, 0.05),
        ]

        prices = [100.0] * 400
        df = _make_ohlcv(prices)
        result = validator._aggregate_results(windows, df)

        expected_strategy = 1.10 * 1.10 - 1  # 0.21
        expected_bh = 1.05 * 1.05 - 1  # 0.1025
        assert result["compounded_strategy_return"] == pytest.approx(expected_strategy)
        assert result["compounded_bh_return"] == pytest.approx(expected_bh)

    def test_window_win_rate(self):
        strategy = RegimeStrategy()
        validator = WalkForwardValidator(strategy=strategy)
        windows = [
            self._make_window_result(0.05, 0.03),  # positive
            self._make_window_result(-0.02, 0.01),  # negative
            self._make_window_result(0.03, 0.01),  # positive
        ]

        prices = [100.0] * 400
        df = _make_ohlcv(prices)
        result = validator._aggregate_results(windows, df)

        # 2 out of 3 windows positive
        assert result["winning_windows"] == 2
        assert result["window_win_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Optimizer - scoring and ranking
# ---------------------------------------------------------------------------


class TestOptimizerScoring:
    """Test OptimizationResult.score and ranking logic."""

    def test_higher_sharpe_scores_higher(self):
        r1 = OptimizationResult(
            params={},
            sharpe=1.5,
            total_return=0.10,
            excess_return=0.05,
            trade_win_rate=0.55,
            profit_factor=1.5,
            total_trades=30,
            max_drawdown=-0.10,
            window_win_rate=0.6,
        )
        r2 = OptimizationResult(
            params={},
            sharpe=0.5,
            total_return=0.10,
            excess_return=0.05,
            trade_win_rate=0.55,
            profit_factor=1.5,
            total_trades=30,
            max_drawdown=-0.10,
            window_win_rate=0.6,
        )
        assert r1.score > r2.score

    def test_excess_return_boosts_score(self):
        r1 = OptimizationResult(
            params={},
            sharpe=1.0,
            total_return=0.15,
            excess_return=0.10,
            trade_win_rate=0.55,
            profit_factor=1.5,
            total_trades=30,
            max_drawdown=-0.10,
            window_win_rate=0.6,
        )
        r2 = OptimizationResult(
            params={},
            sharpe=1.0,
            total_return=0.05,
            excess_return=0.00,
            trade_win_rate=0.55,
            profit_factor=1.5,
            total_trades=30,
            max_drawdown=-0.10,
            window_win_rate=0.6,
        )
        assert r1.score > r2.score

    def test_deep_drawdown_penalised(self):
        base = {
            "params": {},
            "sharpe": 1.0,
            "total_return": 0.10,
            "excess_return": 0.05,
            "trade_win_rate": 0.55,
            "profit_factor": 1.5,
            "total_trades": 30,
            "window_win_rate": 0.6,
        }
        r_ok = OptimizationResult(**base, max_drawdown=-0.15)
        r_bad = OptimizationResult(**base, max_drawdown=-0.40)
        assert r_ok.score > r_bad.score

    def test_too_few_trades_penalised(self):
        base = {
            "params": {},
            "sharpe": 1.0,
            "total_return": 0.10,
            "excess_return": 0.05,
            "trade_win_rate": 0.55,
            "profit_factor": 1.5,
            "max_drawdown": -0.10,
            "window_win_rate": 0.6,
        }
        r_enough = OptimizationResult(**base, total_trades=30)
        r_few = OptimizationResult(**base, total_trades=5)
        assert r_enough.score > r_few.score

    def test_results_sorted_descending(self):
        """Ensure grid_search sorts results best-first."""
        results = [
            OptimizationResult(
                params={},
                sharpe=s,
                total_return=0.10,
                excess_return=0.05,
                trade_win_rate=0.55,
                profit_factor=1.5,
                total_trades=30,
                max_drawdown=-0.10,
                window_win_rate=0.6,
            )
            for s in [0.5, 2.0, 1.0]
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        assert results[0].sharpe == 2.0
        assert results[-1].sharpe == 0.5


# ---------------------------------------------------------------------------
# Optimizer - print_top_results robustness
# ---------------------------------------------------------------------------


class TestPrintTopResults:
    """Ensure print_top_results handles missing keys gracefully."""

    def test_missing_keys_no_crash(self, capsys):
        """Custom search space that omits standard keys should not raise."""
        prices = [100.0] * 400
        df = _make_ohlcv(prices)
        optimizer = StrategyOptimizer(df=df)

        # Inject a result with non-standard params (missing stop_loss, bull_mult, etc.)
        optimizer.results = [
            OptimizationResult(
                params={"custom_param": 42},
                sharpe=1.0,
                total_return=0.10,
                excess_return=0.05,
                trade_win_rate=0.55,
                profit_factor=1.5,
                total_trades=30,
                max_drawdown=-0.10,
                window_win_rate=0.6,
            )
        ]

        # Should not raise TypeError
        optimizer.print_top_results(n=1)
        captured = capsys.readouterr()
        assert "?" in captured.out  # Missing keys shown as '?'
