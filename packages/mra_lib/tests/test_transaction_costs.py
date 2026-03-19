"""Tests for transaction cost models."""

import pytest

from mra_lib.backtesting.transaction_costs import (
    EquityCostModel,
    FuturesCostModel,
    HighFrequencyCostModel,
    RetailCostModel,
    TransactionCostModel,
)

# ---------------------------------------------------------------------------
# Base TransactionCostModel
# ---------------------------------------------------------------------------


class TestTransactionCostModel:
    """Tests for the base TransactionCostModel."""

    def test_default_params(self):
        model = TransactionCostModel()
        assert model.spread_bps == 5.0
        assert model.commission_per_share == 0.005
        assert model.commission_min == 1.0
        assert model.slippage_bps == 2.0

    def test_custom_params(self):
        model = TransactionCostModel(spread_bps=10.0, commission_per_share=0.01, commission_min=2.0)
        assert model.spread_bps == 10.0
        assert model.commission_per_share == 0.01
        assert model.commission_min == 2.0


class TestCalculateTotalCost:
    """Tests for calculate_total_cost."""

    def test_returns_all_keys(self):
        model = TransactionCostModel()
        result = model.calculate_total_cost(100.0, 100, "BUY")
        expected_keys = {
            "spread_cost",
            "commission",
            "slippage",
            "market_impact",
            "total_cost",
            "total_cost_bps",
            "notional_value",
        }
        assert set(result.keys()) == expected_keys

    def test_spread_cost_calculation(self):
        model = TransactionCostModel(
            spread_bps=10.0, commission_per_share=0.0, commission_min=0.0, slippage_bps=0.0
        )
        result = model.calculate_total_cost(100.0, 100, "BUY")
        # 10 bps on $10,000 notional = $10
        assert result["spread_cost"] == pytest.approx(10.0)

    def test_commission_minimum_applied(self):
        model = TransactionCostModel(
            spread_bps=0.0, commission_per_share=0.001, commission_min=5.0, slippage_bps=0.0
        )
        # 10 shares * $0.001 = $0.01, but min is $5
        result = model.calculate_total_cost(100.0, 10, "BUY")
        assert result["commission"] == 5.0

    def test_commission_per_share_when_above_min(self):
        model = TransactionCostModel(
            spread_bps=0.0, commission_per_share=0.01, commission_min=0.0, slippage_bps=0.0
        )
        result = model.calculate_total_cost(100.0, 1000, "BUY")
        assert result["commission"] == pytest.approx(10.0)

    def test_slippage_calculation(self):
        model = TransactionCostModel(
            spread_bps=0.0, commission_per_share=0.0, commission_min=0.0, slippage_bps=5.0
        )
        result = model.calculate_total_cost(100.0, 100, "BUY")
        # 5 bps on $10,000 = $5
        assert result["slippage"] == pytest.approx(5.0)

    def test_no_market_impact_without_volume(self):
        model = TransactionCostModel()
        result = model.calculate_total_cost(100.0, 100, "BUY")
        assert result["market_impact"] == 0.0

    def test_market_impact_with_volume(self):
        model = TransactionCostModel(
            spread_bps=0.0,
            commission_per_share=0.0,
            commission_min=0.0,
            slippage_bps=0.0,
            market_impact_coeff=0.1,
        )
        result = model.calculate_total_cost(100.0, 100, "BUY", avg_volume=10000)
        # participation = 100/10000 = 0.01, sqrt = 0.1
        # impact = $10,000 * 0.1 * 0.1 = $100
        assert result["market_impact"] == pytest.approx(100.0)

    def test_total_cost_is_sum_of_components(self):
        model = TransactionCostModel()
        result = model.calculate_total_cost(100.0, 100, "BUY")
        expected_total = (
            result["spread_cost"]
            + result["commission"]
            + result["slippage"]
            + result["market_impact"]
        )
        assert result["total_cost"] == pytest.approx(expected_total)

    def test_total_cost_bps_calculation(self):
        model = TransactionCostModel()
        result = model.calculate_total_cost(100.0, 100, "BUY")
        expected_bps = (result["total_cost"] / result["notional_value"]) * 10000
        assert result["total_cost_bps"] == pytest.approx(expected_bps)

    def test_negative_shares_treated_as_positive(self):
        model = TransactionCostModel()
        result_pos = model.calculate_total_cost(100.0, 100, "BUY")
        result_neg = model.calculate_total_cost(100.0, -100, "BUY")
        assert result_pos["total_cost"] == result_neg["total_cost"]

    def test_buy_and_sell_same_cost_without_impact(self):
        model = TransactionCostModel()
        buy = model.calculate_total_cost(100.0, 100, "BUY")
        sell = model.calculate_total_cost(100.0, 100, "SELL")
        assert buy["total_cost"] == sell["total_cost"]


class TestRoundtripCost:
    """Tests for calculate_roundtrip_cost."""

    def test_roundtrip_is_double_one_way(self):
        model = TransactionCostModel()
        one_way = model.calculate_total_cost(100.0, 100, "BUY")["total_cost"]
        roundtrip = model.calculate_roundtrip_cost(100.0, 100)
        assert roundtrip == pytest.approx(2 * one_way)

    def test_roundtrip_with_volume(self):
        model = TransactionCostModel()
        roundtrip = model.calculate_roundtrip_cost(100.0, 100, avg_volume=50000)
        assert roundtrip > 0


class TestMinimumProfitThreshold:
    """Tests for get_minimum_profit_threshold."""

    def test_threshold_positive(self):
        model = TransactionCostModel()
        threshold = model.get_minimum_profit_threshold(100.0, 100)
        assert threshold > 0

    def test_threshold_proportional_to_costs(self):
        cheap = TransactionCostModel(
            spread_bps=1.0, slippage_bps=0.0, commission_per_share=0.0, commission_min=0.0
        )
        expensive = TransactionCostModel(
            spread_bps=20.0, slippage_bps=0.0, commission_per_share=0.0, commission_min=0.0
        )
        assert cheap.get_minimum_profit_threshold(
            100.0, 100
        ) < expensive.get_minimum_profit_threshold(100.0, 100)


class TestApplyCostsToReturns:
    """Tests for apply_costs_to_returns."""

    def test_long_profitable_trade(self):
        model = TransactionCostModel(
            spread_bps=0.0, commission_per_share=0.0, commission_min=0.0, slippage_bps=0.0
        )
        result = model.apply_costs_to_returns(100.0, 110.0, 100, "LONG")
        assert result["gross_pnl"] == pytest.approx(1000.0)
        assert result["net_pnl"] == pytest.approx(1000.0)

    def test_short_profitable_trade(self):
        model = TransactionCostModel(
            spread_bps=0.0, commission_per_share=0.0, commission_min=0.0, slippage_bps=0.0
        )
        result = model.apply_costs_to_returns(110.0, 100.0, 100, "SHORT")
        assert result["gross_pnl"] == pytest.approx(1000.0)

    def test_costs_reduce_net_pnl(self):
        model = TransactionCostModel()
        result = model.apply_costs_to_returns(100.0, 110.0, 100, "LONG")
        assert result["net_pnl"] < result["gross_pnl"]
        assert result["total_costs"] > 0

    def test_return_pct_calculation(self):
        model = TransactionCostModel(
            spread_bps=0.0, commission_per_share=0.0, commission_min=0.0, slippage_bps=0.0
        )
        result = model.apply_costs_to_returns(100.0, 110.0, 100, "LONG")
        # 10% return
        assert result["return_pct"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Preset cost models
# ---------------------------------------------------------------------------


class TestPresetModels:
    """Test that preset models instantiate correctly and have expected properties."""

    def test_equity_model_creation(self):
        model = EquityCostModel()
        assert model.spread_bps == 5.0
        result = model.calculate_total_cost(100.0, 100, "BUY")
        assert result["total_cost"] > 0

    def test_futures_model_tighter_spreads(self):
        equity = EquityCostModel()
        futures = FuturesCostModel()
        assert futures.spread_bps < equity.spread_bps

    def test_hft_model_lowest_costs(self):
        hft = HighFrequencyCostModel()
        equity = EquityCostModel()
        # HFT should have tighter spreads
        assert hft.spread_bps < equity.spread_bps
        assert hft.slippage_bps < equity.slippage_bps

    def test_retail_model_wider_spreads(self):
        retail = RetailCostModel()
        equity = EquityCostModel()
        assert retail.spread_bps > equity.spread_bps

    def test_retail_zero_commission(self):
        retail = RetailCostModel()
        assert retail.commission_per_share == 0.0
        assert retail.commission_min == 0.0

    def test_all_models_produce_valid_output(self):
        models = [
            EquityCostModel(),
            FuturesCostModel(),
            HighFrequencyCostModel(),
            RetailCostModel(),
        ]
        for model in models:
            result = model.calculate_total_cost(150.0, 50, "BUY")
            assert result["total_cost"] >= 0
            assert result["notional_value"] == pytest.approx(7500.0)
