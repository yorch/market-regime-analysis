"""Tests for analyzer.py — MarketRegimeAnalyzer internals."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mra_lib.analyzer import MarketRegimeAnalyzer
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.config.regime_tables import REGIME_MULTIPLIERS, REGIME_STRATEGIES


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
    analyzer.regime_multipliers = dict(REGIME_MULTIPLIERS)
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

    def test_unknown_gets_no_allocation(self):
        a = _build_analyzer()
        assert a._get_position_sizing_multiplier(MarketRegime.UNKNOWN, 0.0) == 0.0
        assert a._get_position_sizing_multiplier(MarketRegime.UNKNOWN, 1.0) == 0.0

    @pytest.mark.parametrize("confidence", [0.5, 0.8, 1.0])
    def test_bull_sized_larger_than_bear(self, confidence):
        """Regression: raw multipliers saturated the 50% cap so Bull == Bear."""
        a = _build_analyzer()
        bull = a._get_position_sizing_multiplier(MarketRegime.BULL_TRENDING, confidence)
        bear = a._get_position_sizing_multiplier(MarketRegime.BEAR_TRENDING, confidence)
        assert bull > bear
        assert bear / bull == pytest.approx(0.7 / 1.3)

    def test_strongest_regime_full_confidence_hits_cap(self):
        a = _build_analyzer()
        assert a._get_position_sizing_multiplier(MarketRegime.BULL_TRENDING, 1.0) == pytest.approx(
            0.5
        )

    def test_multipliers_distinct_across_regimes(self):
        a = _build_analyzer()
        values = {
            r: a._get_position_sizing_multiplier(r, 0.9)
            for r in MarketRegime
            if r != MarketRegime.UNKNOWN
        }
        # All base multipliers are distinct, so no two regimes may collapse to the cap.
        assert len(set(values.values())) == len(values)

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
        mult = result.position_sizing_multiplier
        assert mult == 0.0 or 0.01 <= mult <= 0.5
        assert isinstance(result.hmm_state, int)


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


class _FakeDetector:
    """Detector stub with a controlled state sequence and transition matrix."""

    def __init__(self, states, confidence, regime=MarketRegime.BULL_TRENDING):
        from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector

        self._real = HiddenMarkovRegimeDetector(n_states=3)
        self._real.fitted = True
        self._real.transition_matrix = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.25, 0.25, 0.5]]
        )
        self.states = np.array(states)
        self.confidence = confidence
        self.regime = regime
        self.calls = 0

    def predict_with_states(self, df):
        self.calls += 1
        return self.regime, self.states, self.confidence

    def calculate_regime_persistence(self, states, lookback=20):
        return self._real.calculate_regime_persistence(states, lookback)

    def get_transition_probability(self, a, b):
        return self._real.get_transition_probability(a, b)


class TestPersistenceAndTransitionRegression:
    """Regression for review item P0-11: persistence was always 0, risk always High."""

    def _analyze(self, detector):
        a = _build_analyzer(n=300)
        a.indicators["1D"] = a._calculate_technical_indicators(a.data["1D"])
        a.hmm_models["1D"] = detector
        return a.analyze_current_regime("1D")

    def test_state_sequence_predicted_once(self):
        det = _FakeDetector([0] * 30, 0.9)
        self._analyze(det)
        assert det.calls == 1

    def test_persistence_from_sequence_tail(self):
        res = self._analyze(_FakeDetector([1] * 10 + [0] * 15, 0.9))
        assert res.regime_persistence == pytest.approx(15 / 20)
        assert res.hmm_state == 0

    def test_transition_uses_previous_state(self):
        res = self._analyze(_FakeDetector([0] * 10 + [1, 2], 0.9))
        # previous state 1 -> current state 2
        assert res.transition_probability == pytest.approx(0.1)
        res = self._analyze(_FakeDetector([2] * 10, 0.9))
        assert res.transition_probability == pytest.approx(0.5)

    def test_risk_level_varies_with_inputs(self):
        stable = self._analyze(_FakeDetector([0] * 30, 0.95))
        medium = self._analyze(_FakeDetector([1] * 8 + [0] * 12, 0.7))
        choppy = self._analyze(_FakeDetector([0, 1] * 15, 0.95))
        assert stable.risk_level == "Low"
        assert medium.risk_level == "Medium"
        assert choppy.risk_level == "High"

    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_persistence_positive_on_synthetic_data(self, seed):
        from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector

        a = _build_analyzer(df=_make_ohlcv(300, seed=seed))
        df = a.data["1D"]
        a.indicators["1D"] = a._calculate_technical_indicators(df)
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        hmm.fit(df)
        a.hmm_models["1D"] = hmm
        result = a.analyze_current_regime("1D")
        assert result.regime_persistence > 0
        assert result.transition_probability > 0
        _, states, _ = hmm.predict_with_states(df)
        assert result.transition_probability == pytest.approx(
            hmm.get_transition_probability(int(states[-2]), int(states[-1]))
        )


class TestAnnualization:
    def test_volatility_annualized_by_timeframe(self):
        a = _build_analyzer()
        df = _make_ohlcv(300)
        daily = a._calculate_technical_indicators(df, "1D")["volatility"]
        hourly = a._calculate_technical_indicators(df, "1H")["volatility"]
        m15 = a._calculate_technical_indicators(df, "15m")["volatility"]
        ratio_h = (hourly / daily).dropna()
        ratio_m = (m15 / daily).dropna()
        np.testing.assert_allclose(ratio_h, np.sqrt(6.5))
        np.testing.assert_allclose(ratio_m, np.sqrt(26))

    def test_vol_rank_defined_for_intraday(self):
        a = _build_analyzer()
        ind = a._calculate_technical_indicators(_make_ohlcv(600), "15m")
        assert ind["vol_rank"].notna().any()
        valid = ind["vol_rank"].dropna()
        assert ((valid > 0) & (valid <= 1)).all()

    def test_unknown_timeframe_raises(self):
        a = _build_analyzer()
        with pytest.raises(ValueError, match="Unknown timeframe"):
            a._calculate_technical_indicators(_make_ohlcv(100), "3W")


class TestPeriodsPerYear:
    def test_extra_intervals_supported(self):
        from mra_lib.config.regime_tables import periods_per_year

        assert periods_per_year("1wk") == 52
        assert periods_per_year("5M") == 252 * 78
        assert periods_per_year("1h") == periods_per_year("1H")


class TestRegimeTablesSingleSource:
    def test_analyzer_uses_canonical_tables(self):
        a = _build_analyzer()
        for regime in MarketRegime:
            assert a._get_trading_strategy(regime) == REGIME_STRATEGIES[regime]

    def test_init_copies_canonical_multipliers(self):
        with patch("mra_lib.analyzer.MarketDataProvider") as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.fetch.return_value = _make_ohlcv(300)
            mock_provider_cls.create_provider.return_value = mock_provider
            analyzer = MarketRegimeAnalyzer("TEST", periods={"1D": "2y"})
        assert analyzer.regime_multipliers == dict(REGIME_MULTIPLIERS)
        # A copy, so instance overrides cannot mutate the shared table
        analyzer.regime_multipliers[MarketRegime.BULL_TRENDING] = 9.9
        assert REGIME_MULTIPLIERS[MarketRegime.BULL_TRENDING] == 1.3

    def test_risk_calculator_uses_canonical_multipliers(self):
        from mra_lib.risk.risk_calculator import SimonsRiskCalculator

        sizes = {
            r: SimonsRiskCalculator.calculate_regime_adjusted_size(0.1, r, 1.0, 1.0)
            for r in MarketRegime
        }
        for regime, mult in REGIME_MULTIPLIERS.items():
            assert sizes[regime] == pytest.approx(0.1 * mult if mult > 0 else 0.0)


@pytest.fixture(scope="module")
def chart_analyzer():
    return MarketRegimeAnalyzer("TEST", periods={"1D": "2y"}, provider_flag="mock")


class TestRenderRegimeChart:
    @pytest.fixture
    def analyzer(self, chart_analyzer):
        return chart_analyzer

    def test_render_regime_chart_standalone_figure(self, analyzer):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        before = set(plt.get_fignums())
        fig = analyzer.render_regime_chart("1D", 60)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 5
        # Standalone Agg figure: nothing registered with pyplot, nothing to leak
        assert set(plt.get_fignums()) == before

    def test_render_regime_chart_draws_on_given_figure(self, analyzer):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 20))
        try:
            assert analyzer.render_regime_chart("1D", 60, figure=fig) is fig
            assert len(fig.axes) == 5
        finally:
            plt.close(fig)

    def test_render_regime_chart_insufficient_data_raises(self, analyzer):
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.render_regime_chart("1D", 5)

    def test_render_regime_chart_png(self, analyzer):
        assert analyzer.render_regime_chart_png("1D", 60).startswith(b"\x89PNG")
