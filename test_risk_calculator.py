"""Tests for SimonsRiskCalculator and PortfolioPositionLimits."""

import pytest

from market_regime_analysis.enums import MarketRegime
from market_regime_analysis.risk_calculator import (
    PortfolioPositionLimits,
    PositionRecord,
    SimonsRiskCalculator,
)

# ===========================================================================
# SimonsRiskCalculator
# ===========================================================================


class TestKellyOptimalSize:
    """Tests for calculate_kelly_optimal_size."""

    def test_basic_kelly(self):
        # 60% win rate, avg win $2, avg loss $1 -> b=2, f=(2*0.6-0.4)/2=0.4
        size = SimonsRiskCalculator.calculate_kelly_optimal_size(0.6, 2.0, 1.0)
        assert size == pytest.approx(0.25)  # capped at 25%

    def test_no_edge_returns_zero(self):
        # 40% win rate, avg win $1, avg loss $1 -> f=(1*0.4-0.6)/1=-0.2 -> 0
        size = SimonsRiskCalculator.calculate_kelly_optimal_size(0.4, 1.0, 1.0)
        assert size == 0.0

    def test_zero_win_rate(self):
        size = SimonsRiskCalculator.calculate_kelly_optimal_size(0.0, 2.0, 1.0)
        assert size == 0.0

    def test_perfect_win_rate(self):
        size = SimonsRiskCalculator.calculate_kelly_optimal_size(1.0, 2.0, 1.0, confidence=0.5)
        assert size == pytest.approx(0.5)  # confidence scaling

    def test_confidence_scaling(self):
        full = SimonsRiskCalculator.calculate_kelly_optimal_size(0.6, 2.0, 1.0, confidence=1.0)
        half = SimonsRiskCalculator.calculate_kelly_optimal_size(0.6, 2.0, 1.0, confidence=0.5)
        assert half < full

    def test_safety_cap_at_25pct(self):
        # Very high edge: 90% win rate, avg win $10
        size = SimonsRiskCalculator.calculate_kelly_optimal_size(0.9, 10.0, 1.0)
        assert size <= 0.25

    def test_invalid_win_rate_raises(self):
        with pytest.raises(ValueError, match="Win rate"):
            SimonsRiskCalculator.calculate_kelly_optimal_size(1.5, 2.0, 1.0)

    def test_negative_avg_win_raises(self):
        with pytest.raises(ValueError, match="Average win"):
            SimonsRiskCalculator.calculate_kelly_optimal_size(0.5, -1.0, 1.0)

    def test_negative_avg_loss_raises(self):
        with pytest.raises(ValueError, match="Average loss"):
            SimonsRiskCalculator.calculate_kelly_optimal_size(0.5, 1.0, -1.0)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="Confidence"):
            SimonsRiskCalculator.calculate_kelly_optimal_size(0.5, 1.0, 1.0, confidence=2.0)


class TestRegimeAdjustedSize:
    """Tests for calculate_regime_adjusted_size."""

    def test_bull_trending_increases_size(self):
        base = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.BULL_TRENDING, 0.8, 0.8
        )
        unknown = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.UNKNOWN, 0.8, 0.8
        )
        assert base > unknown

    def test_high_volatility_reduces_size(self):
        high_vol = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.HIGH_VOLATILITY, 0.8, 0.8
        )
        low_vol = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.LOW_VOLATILITY, 0.8, 0.8
        )
        assert high_vol < low_vol

    def test_higher_confidence_larger_size(self):
        high = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.BULL_TRENDING, 1.0, 0.8
        )
        low = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.BULL_TRENDING, 0.2, 0.8
        )
        assert high > low

    def test_higher_persistence_larger_size(self):
        high = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.BULL_TRENDING, 0.8, 1.0
        )
        low = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.10, MarketRegime.BULL_TRENDING, 0.8, 0.0
        )
        assert high > low

    def test_minimum_cap(self):
        size = SimonsRiskCalculator.calculate_regime_adjusted_size(
            0.001, MarketRegime.UNKNOWN, 0.0, 0.0
        )
        assert size >= 0.01

    def test_maximum_cap(self):
        size = SimonsRiskCalculator.calculate_regime_adjusted_size(
            1.0, MarketRegime.BULL_TRENDING, 1.0, 1.0
        )
        assert size <= 0.50

    def test_invalid_base_size_raises(self):
        with pytest.raises(ValueError):
            SimonsRiskCalculator.calculate_regime_adjusted_size(
                -0.1, MarketRegime.BULL_TRENDING, 0.8, 0.8
            )

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            SimonsRiskCalculator.calculate_regime_adjusted_size(
                0.10, MarketRegime.BULL_TRENDING, 1.5, 0.8
            )


class TestCorrelationAdjustedSize:
    """Tests for calculate_correlation_adjusted_size."""

    def test_low_correlation_no_change(self):
        size = SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, 0.1)
        assert size == pytest.approx(0.10)

    def test_high_correlation_reduces_size(self):
        low_corr = SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, 0.1)
        high_corr = SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, 0.9)
        assert high_corr < low_corr

    def test_negative_correlation_same_as_positive(self):
        pos = SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, 0.8)
        neg = SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, -0.8)
        assert pos == pytest.approx(neg)

    def test_safety_minimum(self):
        size = SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, 1.0)
        assert size >= 0.01

    def test_invalid_correlation_raises(self):
        with pytest.raises(ValueError):
            SimonsRiskCalculator.calculate_correlation_adjusted_size(0.10, 1.5)


class TestVolatilityAdjustedSize:
    """Tests for calculate_volatility_adjusted_size."""

    def test_high_vol_reduces_size(self):
        high = SimonsRiskCalculator.calculate_volatility_adjusted_size(
            0.10, current_volatility=0.40, historical_volatility=0.20
        )
        low = SimonsRiskCalculator.calculate_volatility_adjusted_size(
            0.10, current_volatility=0.10, historical_volatility=0.20
        )
        assert high < low

    def test_zero_current_vol_raises(self):
        with pytest.raises(ValueError, match="Current volatility"):
            SimonsRiskCalculator.calculate_volatility_adjusted_size(
                0.10, current_volatility=0.0, historical_volatility=0.20
            )


class TestComprehensivePositionSize:
    """Tests for calculate_comprehensive_position_size."""

    def test_returns_all_keys(self):
        result = SimonsRiskCalculator.calculate_comprehensive_position_size(
            0.10, MarketRegime.BULL_TRENDING, 0.8, 0.8
        )
        expected_keys = {
            "base_size",
            "regime_adjusted",
            "correlation_adjusted",
            "kelly_optimal",
            "volatility_adjusted",
            "final_size",
        }
        assert set(result.keys()) == expected_keys

    def test_with_kelly_params(self):
        result = SimonsRiskCalculator.calculate_comprehensive_position_size(
            0.10,
            MarketRegime.BULL_TRENDING,
            0.8,
            0.8,
            win_rate=0.6,
            avg_win=200.0,
            avg_loss=100.0,
        )
        assert result["kelly_optimal"] > 0

    def test_with_volatility_params(self):
        result = SimonsRiskCalculator.calculate_comprehensive_position_size(
            0.10,
            MarketRegime.BULL_TRENDING,
            0.8,
            0.8,
            current_vol=0.20,
            historical_vol=0.15,
        )
        assert result["volatility_adjusted"] > 0

    def test_final_size_always_positive(self):
        result = SimonsRiskCalculator.calculate_comprehensive_position_size(
            0.10, MarketRegime.UNKNOWN, 0.1, 0.1
        )
        assert result["final_size"] > 0


# ===========================================================================
# PortfolioPositionLimits
# ===========================================================================


class TestPositionRecord:
    """Tests for PositionRecord dataclass."""

    def test_creation(self):
        p = PositionRecord(symbol="SPY", direction="LONG", notional=10000.0)
        assert p.symbol == "SPY"
        assert p.direction == "LONG"
        assert p.notional == 10000.0
        assert p.sector == ""

    def test_with_sector(self):
        p = PositionRecord(symbol="AAPL", direction="LONG", notional=5000.0, sector="tech")
        assert p.sector == "tech"


class TestPortfolioPositionLimits:
    """Tests for PortfolioPositionLimits."""

    def _make_limits(self, **kwargs) -> PortfolioPositionLimits:
        defaults = {
            "capital": 100_000.0,
            "max_total_exposure": 1.0,
            "max_per_asset_exposure": 0.20,
            "max_positions": 5,
            "max_net_exposure": 0.60,
            "max_sector_exposure": 0.40,
        }
        defaults.update(kwargs)
        return PortfolioPositionLimits(**defaults)

    # --- Basic exposure tracking ---

    def test_empty_portfolio(self):
        limits = self._make_limits()
        assert limits.get_gross_exposure() == 0.0
        assert limits.get_net_exposure() == 0.0
        assert len(limits.positions) == 0

    def test_add_and_remove_position(self):
        limits = self._make_limits()
        limits.add_position(PositionRecord("SPY", "LONG", 10000.0))
        assert limits.get_gross_exposure() == 10000.0
        limits.remove_position("SPY")
        assert limits.get_gross_exposure() == 0.0

    def test_gross_exposure(self):
        limits = self._make_limits()
        limits.add_position(PositionRecord("SPY", "LONG", 20000.0))
        limits.add_position(PositionRecord("QQQ", "SHORT", 15000.0))
        assert limits.get_gross_exposure() == 35000.0

    def test_net_exposure(self):
        limits = self._make_limits()
        limits.add_position(PositionRecord("SPY", "LONG", 20000.0))
        limits.add_position(PositionRecord("QQQ", "SHORT", 15000.0))
        assert limits.get_net_exposure() == pytest.approx(5000.0)

    def test_long_short_exposure(self):
        limits = self._make_limits()
        limits.add_position(PositionRecord("SPY", "LONG", 20000.0))
        limits.add_position(PositionRecord("QQQ", "SHORT", 10000.0))
        assert limits.get_long_exposure() == 20000.0
        assert limits.get_short_exposure() == 10000.0

    def test_sector_exposure(self):
        limits = self._make_limits()
        limits.add_position(PositionRecord("AAPL", "LONG", 10000.0, sector="tech"))
        limits.add_position(PositionRecord("MSFT", "LONG", 15000.0, sector="tech"))
        limits.add_position(PositionRecord("XOM", "LONG", 5000.0, sector="energy"))
        assert limits.get_sector_exposure("tech") == 25000.0
        assert limits.get_sector_exposure("energy") == 5000.0
        assert limits.get_sector_exposure("finance") == 0.0

    # --- Limit checking ---

    def test_trade_within_limits_allowed(self):
        limits = self._make_limits()
        result = limits.check_limits("SPY", "LONG", 15000.0)
        assert result["allowed"] is True
        assert len(result["violations"]) == 0

    def test_per_asset_limit_violated(self):
        limits = self._make_limits(max_per_asset_exposure=0.10)
        result = limits.check_limits("SPY", "LONG", 15000.0)  # 15% > 10%
        assert result["allowed"] is False
        assert any("Per-asset" in v for v in result["violations"])

    def test_total_exposure_limit_violated(self):
        limits = self._make_limits(max_total_exposure=0.50)
        limits.add_position(PositionRecord("SPY", "LONG", 40000.0))
        result = limits.check_limits("QQQ", "LONG", 15000.0)  # 55% > 50%
        assert result["allowed"] is False
        assert any("Total exposure" in v for v in result["violations"])

    def test_max_positions_limit_violated(self):
        limits = self._make_limits(max_positions=2)
        limits.add_position(PositionRecord("SPY", "LONG", 10000.0))
        limits.add_position(PositionRecord("QQQ", "LONG", 10000.0))
        result = limits.check_limits("IWM", "LONG", 5000.0)
        assert result["allowed"] is False
        assert any("Max positions" in v for v in result["violations"])

    def test_net_exposure_limit_violated(self):
        limits = self._make_limits(max_net_exposure=0.30)
        limits.add_position(PositionRecord("SPY", "LONG", 25000.0))
        result = limits.check_limits("QQQ", "LONG", 10000.0)
        # net = 25000 + 10000 = 35000 -> 35% > 30%
        assert result["allowed"] is False
        assert any("Net exposure" in v for v in result["violations"])

    def test_sector_limit_violated(self):
        limits = self._make_limits(max_sector_exposure=0.30)
        limits.add_position(PositionRecord("AAPL", "LONG", 20000.0, sector="tech"))
        result = limits.check_limits("MSFT", "LONG", 15000.0, sector="tech")
        # 35% > 30%
        assert result["allowed"] is False
        assert any("Sector" in v for v in result["violations"])

    def test_replacing_existing_position(self):
        """Replacing a position in the same symbol should not double-count."""
        limits = self._make_limits(max_total_exposure=0.50, max_per_asset_exposure=0.25)
        limits.add_position(PositionRecord("SPY", "LONG", 20000.0))
        # Replace SPY with a different notional — should not add to gross
        result = limits.check_limits("SPY", "SHORT", 20000.0)
        assert result["allowed"] is True

    def test_short_reduces_net_exposure(self):
        limits = self._make_limits(max_net_exposure=0.50)
        limits.add_position(PositionRecord("SPY", "LONG", 40000.0))
        # Adding a short should reduce net
        result = limits.check_limits("QQQ", "SHORT", 10000.0)
        # Net goes from 40k to 30k (30%) — within 50%
        assert result["allowed"] is True

    # --- Clamping ---

    def test_clamp_within_limits(self):
        limits = self._make_limits()
        clamped = limits.clamp_position_size("SPY", "LONG", 15000.0)
        assert clamped == 15000.0

    def test_clamp_reduces_to_per_asset_limit(self):
        limits = self._make_limits(max_per_asset_exposure=0.10)
        clamped = limits.clamp_position_size("SPY", "LONG", 50000.0)
        assert clamped == pytest.approx(10000.0)

    def test_clamp_reduces_to_gross_headroom(self):
        limits = self._make_limits(max_total_exposure=0.50)
        limits.add_position(PositionRecord("SPY", "LONG", 40000.0))
        clamped = limits.clamp_position_size("QQQ", "LONG", 30000.0)
        assert clamped == pytest.approx(10000.0)  # only 10k headroom

    def test_clamp_returns_zero_when_no_room(self):
        limits = self._make_limits(max_total_exposure=0.30)
        limits.add_position(PositionRecord("SPY", "LONG", 30000.0))
        clamped = limits.clamp_position_size("QQQ", "LONG", 10000.0)
        assert clamped == 0.0

    # --- Portfolio summary ---

    def test_portfolio_summary_empty(self):
        limits = self._make_limits()
        summary = limits.get_portfolio_summary()
        assert summary["n_positions"] == 0
        assert summary["gross_exposure"] == 0.0
        assert summary["headroom_gross"] == 100000.0

    def test_portfolio_summary_with_positions(self):
        limits = self._make_limits()
        limits.add_position(PositionRecord("SPY", "LONG", 30000.0, sector="equity"))
        limits.add_position(PositionRecord("QQQ", "SHORT", 20000.0, sector="equity"))
        summary = limits.get_portfolio_summary()
        assert summary["n_positions"] == 2
        assert summary["gross_exposure"] == 50000.0
        assert summary["gross_exposure_pct"] == pytest.approx(0.50)
        assert summary["net_exposure"] == pytest.approx(10000.0)
        assert summary["long_exposure"] == 30000.0
        assert summary["short_exposure"] == 20000.0
        assert "equity" in summary["sector_exposures"]

    def test_update_capital(self):
        limits = self._make_limits()
        limits.update_capital(200000.0)
        assert limits.capital == 200000.0
        # Per-asset limit should now be based on new capital
        clamped = limits.clamp_position_size("SPY", "LONG", 50000.0)
        # 20% of 200k = 40k
        assert clamped == pytest.approx(40000.0)


# ---------------------------------------------------------------------------
# Integration: BacktestEngine with position limits
# ---------------------------------------------------------------------------


class TestBacktestEngineWithLimits:
    """Test BacktestEngine respects portfolio position limits."""

    def _make_ohlcv(self, prices, start="2020-01-01"):
        import pandas as pd

        dates = pd.bdate_range(start, periods=len(prices))
        return pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Close": prices,
                "Volume": [1_000_000] * len(prices),
            },
            index=dates,
        )

    def test_limits_reduce_position_size(self):
        import pandas as pd

        from market_regime_analysis.backtester.engine import BacktestEngine
        from market_regime_analysis.enums import TradingStrategy

        # Limits: max 10% per asset on 100k capital = $10k
        limits = PortfolioPositionLimits(
            capital=100_000.0,
            max_per_asset_exposure=0.10,
            max_total_exposure=1.0,
            max_net_exposure=1.0,
        )

        prices = [100.0] * 20
        df = self._make_ohlcv(prices)
        idx = df.index

        regimes = pd.Series([MarketRegime.BULL_TRENDING] * 20, index=idx)
        strategies = pd.Series([TradingStrategy.TREND_FOLLOWING] * 20, index=idx)
        # Request 50% position — should be clamped to 10%
        sizes = pd.Series([0.50] * 20, index=idx)
        directions = pd.Series(["LONG"] * 20, index=idx)

        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=None,
            position_limits=limits,
            symbol="SPY",
        )
        results = engine.run_regime_strategy(df, regimes, strategies, sizes, directions)

        # Should have at least 1 trade
        assert len(results["trades"]) >= 1
        # The position should have been clamped to ~100 shares ($10k / $100)
        first_trade = results["trades"][0]
        assert first_trade["shares"] <= 100

    def test_no_limits_allows_full_size(self):
        import pandas as pd

        from market_regime_analysis.backtester.engine import BacktestEngine
        from market_regime_analysis.enums import TradingStrategy

        prices = [100.0] * 20
        df = self._make_ohlcv(prices)
        idx = df.index

        regimes = pd.Series([MarketRegime.BULL_TRENDING] * 20, index=idx)
        strategies = pd.Series([TradingStrategy.TREND_FOLLOWING] * 20, index=idx)
        sizes = pd.Series([0.20] * 20, index=idx)
        directions = pd.Series(["LONG"] * 20, index=idx)

        engine = BacktestEngine(
            initial_capital=100_000,
            stop_loss_pct=None,
            # No position_limits
            symbol="SPY",
        )
        results = engine.run_regime_strategy(df, regimes, strategies, sizes, directions)
        assert len(results["trades"]) >= 1
        # Without limits, 20% of $100k = $20k -> 200 shares at $100
        first_trade = results["trades"][0]
        assert first_trade["shares"] == 200
