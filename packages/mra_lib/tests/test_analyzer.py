"""Tests for analyzer.py — MarketRegimeAnalyzer internals."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mra_lib.analyzer import MarketRegimeAnalyzer
from mra_lib.config.enums import MarketRegime, TradingStrategy


def _make_ohlcv(n=300, seed=42):
    """Create realistic OHLCV data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.015, n))
    close = np.maximum(close, 10)  # Ensure positive
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 1)
    opn = close + rng.normal(0, 0.5, n)
    vol = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _build_analyzer(df=None, n=300):
    """Build analyzer without calling __init__ to avoid data loading."""
    analyzer = object.__new__(MarketRegimeAnalyzer)
    analyzer.symbol = "TEST"
    analyzer.periods = {"1D": "2y"}
    analyzer.regime_multipliers = {
        MarketRegime.BULL_TRENDING: 1.3,
        MarketRegime.BEAR_TRENDING: 0.7,
        MarketRegime.MEAN_REVERTING: 1.2,
        MarketRegime.HIGH_VOLATILITY: 0.4,
        MarketRegime.LOW_VOLATILITY: 1.1,
        MarketRegime.BREAKOUT: 0.9,
        MarketRegime.UNKNOWN: 0.2,
    }
    if df is None:
        df = _make_ohlcv(n)
    analyzer.data = {"1D": df}
    analyzer.indicators = {}
    analyzer.hmm_models = {}
    return analyzer


class TestGetTradingStrategy:
    def test_bull_trending(self):
        a = _build_analyzer()
        assert (
            a._get_trading_strategy(MarketRegime.BULL_TRENDING) == TradingStrategy.TREND_FOLLOWING
        )

    def test_bear_trending(self):
        a = _build_analyzer()
        assert a._get_trading_strategy(MarketRegime.BEAR_TRENDING) == TradingStrategy.DEFENSIVE

    def test_mean_reverting(self):
        a = _build_analyzer()
        assert (
            a._get_trading_strategy(MarketRegime.MEAN_REVERTING) == TradingStrategy.MEAN_REVERSION
        )

    def test_high_volatility(self):
        a = _build_analyzer()
        assert (
            a._get_trading_strategy(MarketRegime.HIGH_VOLATILITY)
            == TradingStrategy.VOLATILITY_TRADING
        )

    def test_low_volatility(self):
        a = _build_analyzer()
        assert a._get_trading_strategy(MarketRegime.LOW_VOLATILITY) == TradingStrategy.MOMENTUM

    def test_breakout(self):
        a = _build_analyzer()
        assert a._get_trading_strategy(MarketRegime.BREAKOUT) == TradingStrategy.MOMENTUM

    def test_unknown(self):
        a = _build_analyzer()
        assert a._get_trading_strategy(MarketRegime.UNKNOWN) == TradingStrategy.AVOID


class TestGetPositionSizingMultiplier:
    def test_bull_high_confidence(self):
        a = _build_analyzer()
        mult = a._get_position_sizing_multiplier(MarketRegime.BULL_TRENDING, 1.0)
        assert 0.01 <= mult <= 0.5

    def test_unknown_low_confidence(self):
        a = _build_analyzer()
        mult = a._get_position_sizing_multiplier(MarketRegime.UNKNOWN, 0.0)
        assert mult == pytest.approx(0.06, abs=0.03)

    def test_caps_at_boundaries(self):
        a = _build_analyzer()
        mult = a._get_position_sizing_multiplier(MarketRegime.BULL_TRENDING, 1.0)
        assert mult <= 0.5
        assert mult >= 0.01

    def test_confidence_scaling(self):
        a = _build_analyzer()
        low_conf = a._get_position_sizing_multiplier(MarketRegime.BULL_TRENDING, 0.0)
        high_conf = a._get_position_sizing_multiplier(MarketRegime.BULL_TRENDING, 1.0)
        assert high_conf >= low_conf


class TestIdentifyArbitrageOpportunities:
    def test_empty_df(self):
        a = _build_analyzer()
        result = a._identify_arbitrage_opportunities(pd.DataFrame())
        assert result == []

    def test_mean_reversion_signal(self):
        a = _build_analyzer()
        df = pd.DataFrame({"price_zscore": [3.0], "autocorr_1": [0.05], "vol_rank": [0.5]})
        result = a._identify_arbitrage_opportunities(df)
        assert any("Mean Reversion" in opp for opp in result)

    def test_momentum_breakdown_signal(self):
        a = _build_analyzer()
        df = pd.DataFrame({"price_zscore": [0.5], "autocorr_1": [0.05], "vol_rank": [0.5]})
        result = a._identify_arbitrage_opportunities(df)
        assert any("Momentum Breakdown" in opp for opp in result)

    def test_high_vol_regime_signal(self):
        a = _build_analyzer()
        df = pd.DataFrame({"price_zscore": [0.5], "autocorr_1": [0.5], "vol_rank": [0.85]})
        result = a._identify_arbitrage_opportunities(df)
        assert any("Vol Regime" in opp for opp in result)

    def test_low_vol_regime_signal(self):
        a = _build_analyzer()
        df = pd.DataFrame({"price_zscore": [0.5], "autocorr_1": [0.5], "vol_rank": [0.15]})
        result = a._identify_arbitrage_opportunities(df)
        assert any("Vol Regime" in opp for opp in result)


class TestGenerateStatisticalSignals:
    def test_empty_df(self):
        a = _build_analyzer()
        result = a._generate_statistical_signals(pd.DataFrame(), MarketRegime.BULL_TRENDING)
        assert result == []

    def test_bb_signal_mean_reverting(self):
        a = _build_analyzer()
        df = pd.DataFrame(
            {
                "Close": [110.0],
                "bb_upper": [105.0],
                "bb_lower": [95.0],
                "rsi": [50.0],
                "macd": [1.0],
                "macd_signal": [0.5],
            }
        )
        result = a._generate_statistical_signals(df, MarketRegime.MEAN_REVERTING)
        assert any("SHORT" in s for s in result)

    def test_ema_signal_bull(self):
        a = _build_analyzer()
        df = pd.DataFrame(
            {
                "Close": [100.0],
                "ema_9": [102.0],
                "ema_34": [98.0],
                "rsi": [50.0],
                "macd": [1.0],
                "macd_signal": [0.5],
            }
        )
        result = a._generate_statistical_signals(df, MarketRegime.BULL_TRENDING)
        assert any("Bullish crossover" in s for s in result)

    def test_rsi_overbought(self):
        a = _build_analyzer()
        df = pd.DataFrame({"Close": [100.0], "rsi": [75.0], "macd": [1.0], "macd_signal": [0.5]})
        result = a._generate_statistical_signals(df, MarketRegime.BULL_TRENDING)
        assert any("Overbought" in s for s in result)

    def test_rsi_oversold(self):
        a = _build_analyzer()
        df = pd.DataFrame({"Close": [100.0], "rsi": [25.0], "macd": [1.0], "macd_signal": [0.5]})
        result = a._generate_statistical_signals(df, MarketRegime.BULL_TRENDING)
        assert any("Oversold" in s for s in result)

    def test_macd_bearish(self):
        a = _build_analyzer()
        df = pd.DataFrame({"Close": [100.0], "rsi": [50.0], "macd": [0.5], "macd_signal": [1.0]})
        result = a._generate_statistical_signals(df, MarketRegime.BULL_TRENDING)
        assert any("Bearish" in s for s in result)


class TestIdentifyKeyLevels:
    def test_short_df_returns_empty(self):
        a = _build_analyzer()
        df = pd.DataFrame({"High": [1, 2], "Low": [0.5, 1], "Close": [1, 2]})
        result = a._identify_key_levels(df)
        assert result == {}

    def test_returns_support_resistance(self):
        a = _build_analyzer()
        df = _make_ohlcv(100)
        a.indicators["1D"] = a._calculate_technical_indicators(df)
        result = a._identify_key_levels(a.indicators["1D"])
        assert "resistance" in result
        assert "support" in result
        assert result["resistance"] >= result["support"]

    def test_includes_ma_levels(self):
        a = _build_analyzer()
        df = _make_ohlcv(300)
        indicators = a._calculate_technical_indicators(df)
        result = a._identify_key_levels(indicators)
        assert "sma_50" in result


class TestCalculateTechnicalIndicators:
    def test_basic_indicators(self):
        a = _build_analyzer()
        df = _make_ohlcv(100)
        result = a._calculate_technical_indicators(df)
        assert "returns" in result.columns
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "volatility" in result.columns

    def test_zero_volume(self):
        a = _build_analyzer()
        df = _make_ohlcv(100)
        df["Volume"] = 0
        result = a._calculate_technical_indicators(df)
        assert "volume_ma" in result.columns
        assert (result["volume_ma"] == 1).all()

    def test_autocorrelation_features(self):
        a = _build_analyzer()
        df = _make_ohlcv(100)
        result = a._calculate_technical_indicators(df)
        assert "autocorr_1" in result.columns
        assert "autocorr_2" in result.columns
        assert "autocorr_5" in result.columns


class TestAnalyzeCurrentRegime:
    def test_missing_timeframe_raises(self):
        a = _build_analyzer()
        with pytest.raises(ValueError, match="not available"):
            a.analyze_current_regime("1H")

    def test_missing_hmm_raises(self):
        a = _build_analyzer()
        a.indicators["1D"] = a._calculate_technical_indicators(a.data["1D"])
        with pytest.raises(ValueError, match="HMM model not available"):
            a.analyze_current_regime("1D")

    def test_full_analysis(self):
        """Integration test: full pipeline with mock data."""
        a = _build_analyzer(n=300)
        df = a.data["1D"]
        a.indicators["1D"] = a._calculate_technical_indicators(df)

        from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector

        hmm = HiddenMarkovRegimeDetector(n_states=4)
        hmm.fit(df)
        a.hmm_models["1D"] = hmm

        result = a.analyze_current_regime("1D")
        assert result.current_regime in list(MarketRegime)
        assert 0 <= result.regime_confidence <= 1
        assert 0 <= result.regime_persistence <= 1
        assert result.recommended_strategy in list(TradingStrategy)
        assert result.risk_level in ["Low", "Medium", "High"]
        assert 0.01 <= result.position_sizing_multiplier <= 0.5


class TestInitialization:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            MarketRegimeAnalyzer("SPY", provider_flag="nonexistent")

    @patch("mra_lib.analyzer.MarketDataProvider")
    def test_init_with_mock_provider(self, mock_provider_cls):
        """Test that the full init flow works with mocked provider."""
        df = _make_ohlcv(300)
        mock_provider = MagicMock()
        mock_provider.fetch.return_value = df
        mock_provider_cls.create_provider.return_value = mock_provider

        analyzer = MarketRegimeAnalyzer("TEST", periods={"1D": "2y"}, provider_flag="yfinance")
        assert "1D" in analyzer.data
        assert "1D" in analyzer.indicators
        assert "1D" in analyzer.hmm_models
