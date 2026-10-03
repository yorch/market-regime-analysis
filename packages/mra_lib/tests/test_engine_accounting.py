"""Regression tests for BacktestEngine cash/equity accounting and fills."""

import numpy as np
import pandas as pd
import pytest

from mra_lib.backtesting.engine import BacktestEngine
from mra_lib.backtesting.transaction_costs import TransactionCostModel
from mra_lib.config.enums import MarketRegime, TradingStrategy


def _zero_cost() -> TransactionCostModel:
    return TransactionCostModel(
        spread_bps=0.0,
        commission_per_share=0.0,
        commission_min=0.0,
        slippage_bps=0.0,
        market_impact_coeff=0.0,
    )


def _ohlcv(closes, opens=None, highs=None, lows=None, volume=1_000_000.0):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    highs = np.maximum(opens, closes) * 1.001 if highs is None else np.asarray(highs, float)
    lows = np.minimum(opens, closes) * 0.999 if lows is None else np.asarray(lows, float)
    idx = pd.bdate_range("2021-01-04", periods=len(closes))
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
        index=idx,
    )


def _run(df, direction_list, *, cost_model=None, stop=None, tp=None):
    idx = df.index
    n = len(idx)
    regimes = pd.Series([MarketRegime.BULL_TRENDING] * n, index=idx)
    strategies = pd.Series([TradingStrategy.TREND_FOLLOWING] * n, index=idx)
    sizes = pd.Series([0.2] * n, index=idx)
    directions = pd.Series(direction_list, index=idx)
    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=cost_model or _zero_cost(),
        max_position_size=1.0,
        stop_loss_pct=stop,
        take_profit_pct=tp,
    )
    return engine, engine.run_regime_strategy(df, regimes, strategies, sizes, directions)


class TestRoundTripAccounting:
    @pytest.mark.parametrize("direction", ["LONG", "SHORT"])
    def test_flat_price_round_trip_ends_at_initial_capital(self, direction):
        df = _ohlcv([100.0] * 20)
        _, res = _run(df, [direction] * 20)
        assert len(res["trades"]) == 1
        assert res["trades"][0]["pnl"] == pytest.approx(0.0)
        assert res["final_capital"] == pytest.approx(100_000.0)
        assert res["total_return"] == pytest.approx(0.0)
        np.testing.assert_allclose(res["equity_curve"].to_numpy(), 100_000.0)

    def test_short_profit_matches_known_prices(self):
        # Short 200 shares at 100 (20% of 100k), price falls to 90 -> +2,000
        closes = [100.0] + [95.0] * 5 + [90.0]
        df = _ohlcv(closes)
        _, res = _run(df, ["SHORT"] * len(closes))
        trade = res["trades"][0]
        assert trade["shares"] == 200
        assert trade["gross_pnl"] == pytest.approx(2_000.0)
        assert res["final_capital"] == pytest.approx(102_000.0)
        # Marked-to-market: at 95 the short is up 1,000
        assert res["equity_curve"].iloc[1] == pytest.approx(101_000.0)

    def test_short_loss_matches_known_prices(self):
        closes = [100.0, 105.0, 110.0]
        df = _ohlcv(closes)
        _, res = _run(df, ["SHORT"] * 3)
        assert res["final_capital"] == pytest.approx(98_000.0)
        assert res["trades"][0]["pnl"] == pytest.approx(-2_000.0)

    def test_cash_change_equals_trade_pnl_with_costs(self):
        closes = [100.0, 97.0, 99.0, 94.0]
        df = _ohlcv(closes)
        _, res = _run(df, ["SHORT"] * 4, cost_model=TransactionCostModel())
        total_pnl = sum(t["pnl"] for t in res["trades"])
        assert res["final_capital"] - 100_000.0 == pytest.approx(total_pnl)


class TestEquityCurve:
    def test_one_point_per_bar_no_duplicate_timestamp(self):
        df = _ohlcv([100.0] * 10)
        _, res = _run(df, ["LONG"] * 10)
        eq = res["equity_curve"]
        assert len(eq) == len(df)
        assert eq.index.is_unique
        assert list(eq.index) == list(df.index)

    def test_last_point_reflects_forced_close_costs(self):
        df = _ohlcv([100.0] * 10)
        _, res = _run(df, ["LONG"] * 10, cost_model=TransactionCostModel())
        assert res["equity_curve"].iloc[-1] == pytest.approx(res["final_capital"])
        assert res["performance"].metrics["total_return"] == pytest.approx(res["total_return"])


class TestGapAwareStops:
    def test_long_stop_gap_down_fills_at_open(self):
        # Enter long at 100 (bar 0); bar 1 gaps down to open 80 (stop at 95)
        df = _ohlcv([100.0, 82.0, 82.0], opens=[100.0, 80.0, 82.0])
        _, res = _run(df, ["LONG", None, None], stop=0.05)
        trade = res["trades"][0]
        assert trade["exit_regime"] == "STOP_LOSS"
        assert trade["exit_price"] == pytest.approx(80.0)

    def test_long_stop_intrabar_fills_at_stop(self):
        df = _ohlcv(
            [100.0, 99.0, 99.0],
            opens=[100.0, 99.0, 99.0],
            highs=[100.5, 99.5, 99.5],
            lows=[99.5, 94.0, 98.5],
        )
        _, res = _run(df, ["LONG", None, None], stop=0.05)
        assert res["trades"][0]["exit_price"] == pytest.approx(95.0)

    def test_short_stop_gap_up_fills_at_open(self):
        df = _ohlcv([100.0, 118.0, 118.0], opens=[100.0, 120.0, 118.0])
        _, res = _run(df, ["SHORT", None, None], stop=0.05)
        trade = res["trades"][0]
        assert trade["exit_regime"] == "STOP_LOSS"
        assert trade["exit_price"] == pytest.approx(120.0)

    def test_long_take_profit_gap_up_fills_at_open(self):
        df = _ohlcv([100.0, 120.0, 120.0], opens=[100.0, 115.0, 120.0])
        _, res = _run(df, ["LONG", None, None], tp=0.10)
        trade = res["trades"][0]
        assert trade["exit_regime"] == "TAKE_PROFIT"
        assert trade["exit_price"] == pytest.approx(115.0)

    def test_gap_through_take_profit_fills_tp_at_open_even_if_stop_touched(self):
        # Opens at 115 (TP at 110) then trades down to 90 (stop at 95)
        df = _ohlcv(
            [100.0, 100.0, 100.0],
            opens=[100.0, 115.0, 100.0],
            highs=[100.5, 116.0, 100.5],
            lows=[99.5, 90.0, 99.5],
        )
        _, res = _run(df, ["LONG", None, None], stop=0.05, tp=0.10)
        trade = res["trades"][0]
        assert trade["exit_regime"] == "TAKE_PROFIT"
        assert trade["exit_price"] == pytest.approx(115.0)

    def test_without_open_column_falls_back_to_level(self):
        df = _ohlcv([100.0, 82.0, 82.0], opens=[100.0, 80.0, 82.0]).drop(columns=["Open"])
        _, res = _run(df, ["LONG", None, None], stop=0.05)
        assert res["trades"][0]["exit_price"] == pytest.approx(95.0)


class TestMarketImpact:
    def test_engine_passes_average_volume(self):
        model = TransactionCostModel(
            spread_bps=0.0,
            commission_per_share=0.0,
            commission_min=0.0,
            slippage_bps=0.0,
            market_impact_coeff=0.1,
        )
        df = _ohlcv([100.0] * 5, volume=20_000.0)
        _, res = _run(df, ["LONG"] * 5, cost_model=model)
        # 200 shares / 20,000 avg volume -> sqrt(0.01) = 0.1; impact = 20,000 * 0.1 * 0.1
        assert res["trades"][0]["entry_costs"] == pytest.approx(200.0)

    def test_no_volume_column_no_impact(self):
        model = TransactionCostModel(
            spread_bps=0.0,
            commission_per_share=0.0,
            commission_min=0.0,
            slippage_bps=0.0,
            market_impact_coeff=0.1,
        )
        df = _ohlcv([100.0] * 5).drop(columns=["Volume"])
        _, res = _run(df, ["LONG"] * 5, cost_model=model)
        assert res["trades"][0]["entry_costs"] == pytest.approx(0.0)
