"""Unit tests for RegimeStrategy."""

import pandas as pd
import pytest

from market_regime_analysis.backtester.strategy import RegimeStrategy
from market_regime_analysis.enums import MarketRegime, TradingStrategy


class TestGetSignal:
    """Tests for RegimeStrategy.get_signal()."""

    def test_bull_trending_returns_long(self):
        strategy = RegimeStrategy()
        direction, mult = strategy.get_signal(MarketRegime.BULL_TRENDING, confidence=0.8)
        assert direction == "LONG"
        assert mult > 0

    def test_high_volatility_returns_no_trade(self):
        strategy = RegimeStrategy()
        direction, mult = strategy.get_signal(MarketRegime.HIGH_VOLATILITY, confidence=0.9)
        assert direction is None
        assert mult == 0.0

    def test_unknown_regime_returns_no_trade(self):
        strategy = RegimeStrategy()
        direction, mult = strategy.get_signal(MarketRegime.UNKNOWN, confidence=1.0)
        assert direction is None
        assert mult == 0.0

    def test_low_confidence_blocks_trade(self):
        strategy = RegimeStrategy(min_confidence=0.5)
        direction, mult = strategy.get_signal(MarketRegime.BULL_TRENDING, confidence=0.3)
        assert direction is None
        assert mult == 0.0

    def test_confidence_scaling_affects_multiplier(self):
        strategy = RegimeStrategy(confidence_scaling=True, base_position_fraction=0.10)
        _, mult_high = strategy.get_signal(MarketRegime.BULL_TRENDING, confidence=1.0)
        _, mult_low = strategy.get_signal(MarketRegime.BULL_TRENDING, confidence=0.5)
        assert mult_high > mult_low

    def test_no_confidence_scaling(self):
        strategy = RegimeStrategy(confidence_scaling=False, base_position_fraction=0.10)
        _, mult_high = strategy.get_signal(MarketRegime.BULL_TRENDING, confidence=1.0)
        _, mult_low = strategy.get_signal(MarketRegime.BULL_TRENDING, confidence=0.5)
        assert mult_high == mult_low

    def test_base_position_fraction_scales_output(self):
        s1 = RegimeStrategy(base_position_fraction=0.10, confidence_scaling=False)
        s2 = RegimeStrategy(base_position_fraction=0.20, confidence_scaling=False)
        _, m1 = s1.get_signal(MarketRegime.BULL_TRENDING, confidence=0.8)
        _, m2 = s2.get_signal(MarketRegime.BULL_TRENDING, confidence=0.8)
        assert m2 == pytest.approx(m1 * 2.0)

    def test_bear_short_direction(self):
        strategy = RegimeStrategy(
            regime_directions={
                MarketRegime.BEAR_TRENDING: "SHORT",
                MarketRegime.BULL_TRENDING: "LONG",
                MarketRegime.MEAN_REVERTING: "LONG",
                MarketRegime.LOW_VOLATILITY: "LONG",
                MarketRegime.HIGH_VOLATILITY: None,
                MarketRegime.BREAKOUT: "LONG",
                MarketRegime.UNKNOWN: None,
            }
        )
        direction, _ = strategy.get_signal(MarketRegime.BEAR_TRENDING, confidence=0.8)
        assert direction == "SHORT"


class TestFromParamVector:
    """Tests for RegimeStrategy.from_param_vector()."""

    def test_creates_strategy_with_defaults(self):
        strategy = RegimeStrategy.from_param_vector({})
        assert isinstance(strategy, RegimeStrategy)

    def test_bull_mult_applied(self):
        strategy = RegimeStrategy.from_param_vector({"bull_mult": 2.0})
        assert strategy.regime_multipliers[MarketRegime.BULL_TRENDING] == 2.0

    def test_bear_short_true(self):
        strategy = RegimeStrategy.from_param_vector({"bear_short": 1})
        assert strategy.regime_directions[MarketRegime.BEAR_TRENDING] == "SHORT"

    def test_bear_short_false(self):
        strategy = RegimeStrategy.from_param_vector({"bear_short": 0})
        assert strategy.regime_directions[MarketRegime.BEAR_TRENDING] is None

    def test_base_fraction_stored(self):
        strategy = RegimeStrategy.from_param_vector({"base_fraction": 0.15})
        assert strategy.base_position_fraction == 0.15

    def test_stop_loss_stored(self):
        strategy = RegimeStrategy.from_param_vector({"stop_loss": 0.03})
        assert strategy.stop_loss_pct == 0.03


class TestGenerateSignals:
    """Tests for RegimeStrategy.generate_signals()."""

    def test_output_lengths_match_input(self):
        strategy = RegimeStrategy()
        regimes = pd.Series(
            [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING, MarketRegime.UNKNOWN]
        )
        confidences = pd.Series([0.8, 0.7, 0.5])

        strategies, directions, sizes = strategy.generate_signals(regimes, confidences)

        assert len(strategies) == 3
        assert len(directions) == 3
        assert len(sizes) == 3

    def test_unknown_regime_produces_avoid(self):
        strategy = RegimeStrategy()
        regimes = pd.Series([MarketRegime.UNKNOWN])
        confidences = pd.Series([0.5])

        strategies, directions, sizes = strategy.generate_signals(regimes, confidences)

        assert strategies.iloc[0] == TradingStrategy.AVOID
        assert directions.iloc[0] is None
        assert sizes.iloc[0] == 0.0

    def test_bull_regime_produces_nonzero_size(self):
        strategy = RegimeStrategy(base_position_fraction=0.10)
        regimes = pd.Series([MarketRegime.BULL_TRENDING])
        confidences = pd.Series([0.8])

        _, directions, sizes = strategy.generate_signals(regimes, confidences)

        assert directions.iloc[0] == "LONG"
        assert sizes.iloc[0] > 0

    def test_position_sizes_reflect_base_fraction(self):
        s1 = RegimeStrategy(base_position_fraction=0.10, confidence_scaling=False)
        s2 = RegimeStrategy(base_position_fraction=0.20, confidence_scaling=False)

        regimes = pd.Series([MarketRegime.BULL_TRENDING])
        confidences = pd.Series([0.8])

        _, _, sizes1 = s1.generate_signals(regimes, confidences)
        _, _, sizes2 = s2.generate_signals(regimes, confidences)

        assert sizes2.iloc[0] == pytest.approx(sizes1.iloc[0] * 2.0)
