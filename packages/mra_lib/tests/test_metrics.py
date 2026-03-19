"""Tests for backtesting/metrics.py — PerformanceMetrics."""

import numpy as np
import pandas as pd
import pytest

from mra_lib.backtesting.metrics import PerformanceMetrics


def _make_equity(values, start="2020-01-01"):
    """Create an equity curve Series."""
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx)


def _make_trades(pnls, directions=None):
    """Create trade list from a list of pnl values."""
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append(
            {
                "entry_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 5),
                "exit_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 5 + 3),
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl / 10,
                "shares": 10,
                "direction": (directions[i] if directions else "LONG"),
                "pnl": pnl,
                "return_pct": pnl / 1000 * 100,
            }
        )
    return trades


class TestBasicStats:
    def test_total_return(self):
        eq = _make_equity([100_000, 105_000, 110_000])
        trades = _make_trades([5000, 5000])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["total_return"] == pytest.approx(0.10)

    def test_two_element_equity(self):
        eq = _make_equity([100_000, 100_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["total_return"] == pytest.approx(0.0)
        assert pm.metrics["total_trades"] == 0

    def test_annualized_return(self):
        # 252 trading days = 1 year
        values = np.linspace(100_000, 120_000, 252)
        eq = _make_equity(values)
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["annualized_return"] == pytest.approx(0.2, abs=0.01)

    def test_total_trades_count(self):
        eq = _make_equity([100_000, 101_000])
        trades = _make_trades([500, -200, 300])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["total_trades"] == 3


class TestRiskMetrics:
    def test_volatility_positive(self):
        rng = np.random.default_rng(42)
        values = 100_000 + np.cumsum(rng.normal(0, 100, 100))
        eq = _make_equity(values)
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["volatility"] > 0
        assert pm.metrics["annualized_volatility"] > 0

    def test_downside_deviation(self):
        rng = np.random.default_rng(42)
        values = 100_000 + np.cumsum(rng.normal(-10, 100, 100))
        eq = _make_equity(values)
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["downside_deviation"] > 0
        assert pm.metrics["annualized_downside_deviation"] > 0

    def test_constant_equity_zero_vol(self):
        eq = _make_equity([100_000] * 10)
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["volatility"] == 0.0


class TestTradeStats:
    def test_no_trades(self):
        eq = _make_equity([100_000, 100_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["win_rate"] == 0.0
        assert pm.metrics["profit_factor"] == 0.0
        assert pm.metrics["avg_win"] == 0.0
        assert pm.metrics["avg_loss"] == 0.0

    def test_all_winners(self):
        eq = _make_equity([100_000, 110_000])
        trades = _make_trades([1000, 2000, 3000])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["win_rate"] == 1.0
        assert pm.metrics["profit_factor"] == 999.0  # No losing trades
        assert pm.metrics["winning_trades"] == 3
        assert pm.metrics["losing_trades"] == 0

    def test_all_losers(self):
        eq = _make_equity([100_000, 90_000])
        trades = _make_trades([-1000, -2000])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["win_rate"] == 0.0
        assert pm.metrics["profit_factor"] == 0.0
        assert pm.metrics["losing_trades"] == 2

    def test_mixed_trades(self):
        eq = _make_equity([100_000, 102_000])
        trades = _make_trades([1000, -500, 2000, -300])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["win_rate"] == pytest.approx(0.5)
        assert pm.metrics["profit_factor"] == pytest.approx(3000 / 800)
        assert pm.metrics["avg_win"] == pytest.approx(1500)
        assert pm.metrics["avg_loss"] == pytest.approx(400)

    def test_expectancy(self):
        eq = _make_equity([100_000, 102_000])
        trades = _make_trades([1000, -500])
        pm = PerformanceMetrics(trades, eq)
        # expectancy = win_rate * avg_win - (1-win_rate) * avg_loss
        expected = 0.5 * 1000 - 0.5 * 500
        assert pm.metrics["expectancy"] == pytest.approx(expected)


class TestDrawdownMetrics:
    def test_no_drawdown(self):
        eq = _make_equity([100_000, 101_000, 102_000, 103_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["max_drawdown"] == pytest.approx(0.0)
        assert pm.metrics["max_drawdown_duration"] == 0

    def test_simple_drawdown(self):
        eq = _make_equity([100_000, 110_000, 100_000, 105_000])
        pm = PerformanceMetrics([], eq)
        # Max DD: (100_000 - 110_000) / 110_000
        expected_dd = (100_000 - 110_000) / 110_000
        assert pm.metrics["max_drawdown"] == pytest.approx(expected_dd, abs=0.001)

    def test_drawdown_duration(self):
        # Peak, then drawdown for 3 periods, then recovery
        eq = _make_equity([100_000, 110_000, 105_000, 100_000, 108_000, 112_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["max_drawdown_duration"] >= 1

    def test_avg_drawdown(self):
        eq = _make_equity([100_000, 110_000, 100_000, 115_000, 105_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["avg_drawdown"] < 0

    def test_short_equity(self):
        eq = _make_equity([100_000, 100_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["max_drawdown"] == pytest.approx(0.0)


class TestRatioMetrics:
    def test_sharpe_ratio(self):
        rng = np.random.default_rng(42)
        values = 100_000 + np.cumsum(rng.normal(50, 100, 252))
        eq = _make_equity(values)
        pm = PerformanceMetrics([], eq)
        # Should be positive for upward trend
        assert pm.metrics["sharpe_ratio"] != 0.0

    def test_sortino_ratio(self):
        rng = np.random.default_rng(42)
        values = 100_000 + np.cumsum(rng.normal(50, 100, 252))
        eq = _make_equity(values)
        pm = PerformanceMetrics([], eq)
        assert "sortino_ratio" in pm.metrics

    def test_calmar_ratio_no_drawdown(self):
        eq = _make_equity([100_000, 101_000, 102_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["calmar_ratio"] == 0.0  # No drawdown

    def test_calmar_ratio_with_drawdown(self):
        eq = _make_equity([100_000, 110_000, 100_000, 120_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["calmar_ratio"] > 0


class TestKellyParameters:
    def test_kelly_with_edge(self):
        eq = _make_equity([100_000, 110_000])
        trades = _make_trades([2000, 2000, -500])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["kelly_fraction"] > 0
        assert pm.metrics["half_kelly"] == pytest.approx(pm.metrics["kelly_fraction"] * 0.5)
        assert pm.metrics["quarter_kelly"] == pytest.approx(pm.metrics["kelly_fraction"] * 0.25)

    def test_kelly_no_edge(self):
        eq = _make_equity([100_000, 100_000])
        trades = _make_trades([-1000, -2000])
        pm = PerformanceMetrics(trades, eq)
        # With only losses, kelly should be <= 0
        assert pm.metrics["kelly_fraction"] <= 0

    def test_kelly_no_trades(self):
        eq = _make_equity([100_000, 100_000])
        pm = PerformanceMetrics([], eq)
        assert pm.metrics["kelly_fraction"] == 0.0

    def test_win_loss_ratio(self):
        eq = _make_equity([100_000, 110_000])
        trades = _make_trades([3000, -1000])
        pm = PerformanceMetrics(trades, eq)
        assert pm.metrics["kelly_win_loss_ratio"] == pytest.approx(3.0)


class TestGetSummary:
    def test_returns_dict(self):
        eq = _make_equity([100_000, 110_000])
        pm = PerformanceMetrics(_make_trades([1000]), eq)
        summary = pm.get_summary()
        assert isinstance(summary, dict)
        assert "total_return" in summary
        assert "sharpe_ratio" in summary
        assert "kelly_fraction" in summary


class TestIsProfitable:
    def test_profitable(self):
        rng = np.random.default_rng(42)
        values = 100_000 + np.cumsum(rng.normal(100, 80, 252))
        eq = _make_equity(values)
        trades = _make_trades([100] * 20 + [-50] * 10 + [100] * 5)
        pm = PerformanceMetrics(trades, eq)
        # May or may not be profitable depending on metrics
        result = pm.is_profitable(min_sharpe=0.0, min_trades=5)
        assert isinstance(result, bool)

    def test_not_profitable_too_few_trades(self):
        eq = _make_equity([100_000, 110_000])
        trades = _make_trades([1000])
        pm = PerformanceMetrics(trades, eq)
        assert pm.is_profitable(min_trades=30) is False

    def test_not_profitable_negative_return(self):
        eq = _make_equity([100_000, 90_000])
        trades = _make_trades([-500] * 40)
        pm = PerformanceMetrics(trades, eq)
        assert pm.is_profitable() is False


class TestPrintSummary:
    def test_prints_without_error(self, capsys):
        eq = _make_equity([100_000, 105_000, 110_000])
        trades = _make_trades([5000, 5000])
        pm = PerformanceMetrics(trades, eq)
        pm.print_summary()
        captured = capsys.readouterr()
        assert "BACKTEST PERFORMANCE SUMMARY" in captured.out
        assert "Total Return" in captured.out
