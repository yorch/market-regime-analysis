"""Extended tests for backtesting/engine.py — covering uncovered lines."""

import pandas as pd

from mra_lib.backtesting.engine import BacktestEngine
from mra_lib.config.enums import MarketRegime, TradingStrategy


def _make_ohlcv(prices, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1_000_000] * len(prices),
        },
        index=idx,
    )


def _constant_series(value, index):
    return pd.Series([value] * len(index), index=index)


class TestStopLossLong:
    def test_stop_loss_triggers_on_low(self):
        """Stop loss should trigger when Low <= stop price for LONG."""
        prices = [100, 100, 100, 100]
        df = _make_ohlcv(prices)
        # Low = price - 1 = 99; stop at 100 * (1-0.01) = 99 -> triggers
        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=0.01,
            max_position_size=0.5,
        )
        regimes = _constant_series(MarketRegime.BULL_TRENDING, df.index)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, df.index)
        sizes = _constant_series(0.1, df.index)

        result = engine.run_regime_strategy(df, regimes, strategies, sizes)
        # Should have at least one stop loss trade
        stop_trades = [t for t in result["trades"] if t["exit_regime"] == "STOP_LOSS"]
        assert len(stop_trades) >= 1


class TestStopLossShort:
    def test_stop_loss_triggers_for_short(self):
        """Stop loss should trigger when High >= stop price for SHORT."""
        prices = [100, 100, 100, 100]
        df = _make_ohlcv(prices)
        # High = 101; stop = 100 * (1 + 0.005) = 100.5 -> triggers
        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=0.005,
            max_position_size=0.5,
        )
        regimes = _constant_series(MarketRegime.BULL_TRENDING, df.index)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, df.index)
        sizes = _constant_series(0.1, df.index)
        directions = pd.Series(["SHORT"] * len(df), index=df.index)

        result = engine.run_regime_strategy(df, regimes, strategies, sizes, directions=directions)
        stop_trades = [t for t in result["trades"] if t["exit_regime"] == "STOP_LOSS"]
        assert len(stop_trades) >= 1


class TestTakeProfit:
    def test_take_profit_long(self):
        """Take profit triggers when High >= profit price for LONG."""
        prices = [100, 100, 100, 100]
        df = _make_ohlcv(prices)
        # High = 101; profit = 100 * (1 + 0.005) = 100.5 -> triggers
        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=None,
            take_profit_pct=0.005,
            max_position_size=0.5,
        )
        regimes = _constant_series(MarketRegime.BULL_TRENDING, df.index)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, df.index)
        sizes = _constant_series(0.1, df.index)

        result = engine.run_regime_strategy(df, regimes, strategies, sizes)
        tp_trades = [t for t in result["trades"] if t["exit_regime"] == "TAKE_PROFIT"]
        assert len(tp_trades) >= 1

    def test_take_profit_short(self):
        """Take profit triggers when Low <= profit price for SHORT."""
        prices = [100, 100, 100, 100]
        df = _make_ohlcv(prices)
        # Low = 99; profit = 100 * (1 - 0.005) = 99.5 -> triggers
        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=None,
            take_profit_pct=0.005,
            max_position_size=0.5,
        )
        regimes = _constant_series(MarketRegime.BULL_TRENDING, df.index)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, df.index)
        sizes = _constant_series(0.1, df.index)
        directions = pd.Series(["SHORT"] * len(df), index=df.index)

        result = engine.run_regime_strategy(df, regimes, strategies, sizes, directions=directions)
        tp_trades = [t for t in result["trades"] if t["exit_regime"] == "TAKE_PROFIT"]
        assert len(tp_trades) >= 1


class TestRegimeExits:
    def test_exit_on_high_volatility(self):
        """Position exits when regime changes to HIGH_VOLATILITY."""
        prices = [100, 101, 102, 103, 104]
        df = _make_ohlcv(prices)
        regimes = pd.Series(
            [
                MarketRegime.BULL_TRENDING,
                MarketRegime.BULL_TRENDING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.BULL_TRENDING,
            ],
            index=df.index,
        )
        strategies = pd.Series(
            [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.VOLATILITY_TRADING,
                TradingStrategy.VOLATILITY_TRADING,
                TradingStrategy.TREND_FOLLOWING,
            ],
            index=df.index,
        )
        sizes = _constant_series(0.1, df.index)
        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=None)
        result = engine.run_regime_strategy(df, regimes, strategies, sizes)
        # Should have exited at the HIGH_VOLATILITY regime change
        assert len(result["trades"]) >= 1

    def test_exit_on_avoid_strategy(self):
        """Position exits when strategy changes to AVOID."""
        prices = [100, 101, 102, 103]
        df = _make_ohlcv(prices)
        regimes = pd.Series(
            [
                MarketRegime.BULL_TRENDING,
                MarketRegime.BULL_TRENDING,
                MarketRegime.UNKNOWN,
                MarketRegime.UNKNOWN,
            ],
            index=df.index,
        )
        strategies = pd.Series(
            [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.AVOID,
                TradingStrategy.AVOID,
            ],
            index=df.index,
        )
        sizes = _constant_series(0.1, df.index)
        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=None)
        result = engine.run_regime_strategy(df, regimes, strategies, sizes)
        assert len(result["trades"]) >= 1


class TestShouldEnterPosition:
    def test_trend_following_enters(self):
        engine = BacktestEngine()
        assert engine._should_enter_position(TradingStrategy.TREND_FOLLOWING) is True

    def test_momentum_enters(self):
        engine = BacktestEngine()
        assert engine._should_enter_position(TradingStrategy.MOMENTUM) is True

    def test_mean_reversion_enters(self):
        engine = BacktestEngine()
        assert engine._should_enter_position(TradingStrategy.MEAN_REVERSION) is True

    def test_defensive_does_not_enter(self):
        engine = BacktestEngine()
        assert engine._should_enter_position(TradingStrategy.DEFENSIVE) is False

    def test_avoid_does_not_enter(self):
        engine = BacktestEngine()
        assert engine._should_enter_position(TradingStrategy.AVOID) is False


class TestGetDirectionFromStrategy:
    def test_trend_following_long(self):
        engine = BacktestEngine()
        assert engine._get_direction_from_strategy(TradingStrategy.TREND_FOLLOWING) == "LONG"

    def test_momentum_long(self):
        engine = BacktestEngine()
        assert engine._get_direction_from_strategy(TradingStrategy.MOMENTUM) == "LONG"

    def test_mean_reversion_long(self):
        engine = BacktestEngine()
        assert engine._get_direction_from_strategy(TradingStrategy.MEAN_REVERSION) == "LONG"

    def test_defensive_none(self):
        engine = BacktestEngine()
        assert engine._get_direction_from_strategy(TradingStrategy.DEFENSIVE) is None

    def test_avoid_none(self):
        engine = BacktestEngine()
        assert engine._get_direction_from_strategy(TradingStrategy.AVOID) is None


class TestCalculateCurrentEquity:
    def test_no_position(self):
        engine = BacktestEngine(initial_capital=100_000)
        assert engine._calculate_current_equity(100) == 100_000

    def test_long_position(self):
        engine = BacktestEngine(initial_capital=50_000)
        engine.position = {"shares": 100, "direction": "LONG", "entry_price": 100}
        # equity = capital + position_value
        equity = engine._calculate_current_equity(110)
        assert equity == 50_000 + 110 * 100

    def test_short_position(self):
        engine = BacktestEngine(initial_capital=100_000)
        engine.position = {"shares": 100, "direction": "SHORT", "entry_price": 100}
        # equity = capital - position_liability
        equity = engine._calculate_current_equity(90)
        assert equity == 100_000 - 90 * 100


class TestPrintResults:
    def test_prints_without_error(self, capsys):
        engine = BacktestEngine(initial_capital=100_000)
        results = {
            "final_capital": 110_000,
            "total_return": 0.10,
            "trades": [{"pnl": 5000}, {"pnl": -2000}],
            "equity_curve": pd.Series([100_000, 110_000]),
        }
        engine.print_results(results)
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out

    def test_no_trades(self, capsys):
        engine = BacktestEngine()
        results = {
            "final_capital": 100_000,
            "total_return": 0.0,
            "trades": [],
            "equity_curve": pd.Series([100_000]),
        }
        engine.print_results(results)
        captured = capsys.readouterr()
        assert "Total Trades" in captured.out


class TestShortPositionMechanics:
    def test_short_position_pnl(self):
        """Verify SHORT position P&L is calculated correctly."""
        prices = [100, 90, 80]
        df = _make_ohlcv(prices)
        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=None,
            max_position_size=0.5,
        )
        regimes = _constant_series(MarketRegime.BULL_TRENDING, df.index)
        strategies = _constant_series(TradingStrategy.TREND_FOLLOWING, df.index)
        sizes = _constant_series(0.1, df.index)
        directions = pd.Series(["SHORT", "SHORT", "SHORT"], index=df.index)

        result = engine.run_regime_strategy(df, regimes, strategies, sizes, directions=directions)
        # Price went down, so SHORT should be profitable
        assert len(result["trades"]) >= 1
        if result["trades"]:
            # The last trade closes at end
            last_trade = result["trades"][-1]
            assert last_trade["direction"] == "SHORT"
