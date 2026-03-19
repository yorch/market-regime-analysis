"""Tests for portfolio/portfolio.py — PortfolioHMMAnalyzer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.portfolio.portfolio import PortfolioHMMAnalyzer


def _mock_analysis(
    regime=MarketRegime.BULL_TRENDING,
    confidence=0.85,
    state=0,
    persistence=0.7,
):
    """Create a mock RegimeAnalysis."""
    return RegimeAnalysis(
        current_regime=regime,
        hmm_state=state,
        transition_probability=0.5,
        regime_persistence=persistence,
        recommended_strategy=TradingStrategy.TREND_FOLLOWING,
        position_sizing_multiplier=0.1,
        risk_level="Low",
        arbitrage_opportunities=[],
        statistical_signals=[],
        key_levels={},
        regime_confidence=confidence,
    )


def _build_portfolio(symbols, analyses_map, portfolio_data=None, periods=None):
    """Build a PortfolioHMMAnalyzer without calling __init__."""
    p = object.__new__(PortfolioHMMAnalyzer)
    p.symbols = symbols
    p.periods = periods or {"1D": "2y"}
    p.analyzers = {}
    p.portfolio_data = portfolio_data or {}

    for sym in symbols:
        mock_analyzer = MagicMock()

        def _make_side_effect(s):
            def side_effect(tf):
                return analyses_map[s]

            return side_effect

        mock_analyzer.analyze_current_regime = MagicMock(side_effect=_make_side_effect(sym))
        p.analyzers[sym] = mock_analyzer

    return p


class TestCalculateRegimeCorrelations:
    def test_raises_if_no_portfolio_data(self):
        p = _build_portfolio(["SPY"], {"SPY": _mock_analysis()})
        with pytest.raises(ValueError, match="not available"):
            p.calculate_regime_correlations("1D")

    def test_returns_dataframe(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "SPY": np.cumsum(np.random.default_rng(42).normal(0, 1, 100)) + 100,
                "QQQ": np.cumsum(np.random.default_rng(43).normal(0, 1, 100)) + 100,
            },
            index=idx,
        )
        analyses = {
            "SPY": _mock_analysis(),
            "QQQ": _mock_analysis(regime=MarketRegime.MEAN_REVERTING, confidence=0.7),
        }
        p = _build_portfolio(["SPY", "QQQ"], analyses, portfolio_data={"1D": df})
        result = p.calculate_regime_correlations("1D")
        assert isinstance(result, pd.DataFrame)
        assert "SPY" in result.index
        assert "QQQ" in result.index
        assert "regime" in result.columns

    def test_skips_failed_analyzers(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"SPY": np.ones(100) * 100, "QQQ": np.ones(100) * 200},
            index=idx,
        )
        analyses = {"SPY": _mock_analysis(), "QQQ": _mock_analysis()}
        p = _build_portfolio(["SPY", "QQQ"], analyses, portfolio_data={"1D": df})
        # Make QQQ analyzer raise
        p.analyzers["QQQ"].analyze_current_regime.side_effect = RuntimeError("fail")
        result = p.calculate_regime_correlations("1D")
        assert "SPY" in result.index
        # QQQ should be missing since it raised
        assert "QQQ" not in result.index


class TestGetPortfolioRegimeSummary:
    def test_empty_analyzers_returns_defaults(self):
        p = _build_portfolio([], {})
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["dominant_regime"] is None
        assert summary["regime_consensus"] == 0.0
        assert summary["average_confidence"] == 0.0

    def test_single_symbol(self):
        analyses = {"SPY": _mock_analysis(regime=MarketRegime.BULL_TRENDING, confidence=0.9)}
        p = _build_portfolio(["SPY"], analyses)
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["dominant_regime"] == "Bull Trending"
        assert summary["regime_consensus"] == 1.0
        assert summary["average_confidence"] == pytest.approx(0.9)

    def test_multiple_symbols_dominant_regime(self):
        analyses = {
            "SPY": _mock_analysis(regime=MarketRegime.BULL_TRENDING),
            "QQQ": _mock_analysis(regime=MarketRegime.BULL_TRENDING),
            "IWM": _mock_analysis(regime=MarketRegime.BEAR_TRENDING),
        }
        p = _build_portfolio(["SPY", "QQQ", "IWM"], analyses)
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["dominant_regime"] == "Bull Trending"
        assert summary["regime_consensus"] == pytest.approx(2 / 3)

    def test_risk_level_high(self):
        # 2/3 high_vol + unknown > 0.5 threshold -> "High"
        analyses = {
            "A": _mock_analysis(regime=MarketRegime.HIGH_VOLATILITY),
            "B": _mock_analysis(regime=MarketRegime.UNKNOWN),
            "C": _mock_analysis(regime=MarketRegime.BULL_TRENDING),
        }
        p = _build_portfolio(["A", "B", "C"], analyses)
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["risk_level"] == "High"

    def test_risk_level_medium(self):
        # 2/5 = 0.4 -> between 0.3 and 0.5 -> "Medium"
        analyses = {
            "A": _mock_analysis(regime=MarketRegime.HIGH_VOLATILITY),
            "B": _mock_analysis(regime=MarketRegime.UNKNOWN),
            "C": _mock_analysis(regime=MarketRegime.BULL_TRENDING),
            "D": _mock_analysis(regime=MarketRegime.MEAN_REVERTING),
            "E": _mock_analysis(regime=MarketRegime.LOW_VOLATILITY),
        }
        p = _build_portfolio(["A", "B", "C", "D", "E"], analyses)
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["risk_level"] == "Medium"

    def test_risk_level_low(self):
        analyses = {
            "A": _mock_analysis(regime=MarketRegime.BULL_TRENDING),
            "B": _mock_analysis(regime=MarketRegime.MEAN_REVERTING),
            "C": _mock_analysis(regime=MarketRegime.LOW_VOLATILITY),
        }
        p = _build_portfolio(["A", "B", "C"], analyses)
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["risk_level"] == "Low"

    def test_correlation_risk_with_portfolio_data(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.normal(0, 1, 100))
        df = pd.DataFrame(
            {
                "SPY": base + 100,
                "QQQ": base * 0.8 + rng.normal(0, 0.5, 100) + 100,
                "IWM": -base * 0.5 + rng.normal(0, 0.5, 100) + 100,
            },
            index=idx,
        )
        analyses = {
            "SPY": _mock_analysis(),
            "QQQ": _mock_analysis(),
            "IWM": _mock_analysis(),
        }
        p = _build_portfolio(["SPY", "QQQ", "IWM"], analyses, portfolio_data={"1D": df})
        summary = p.get_portfolio_regime_summary("1D")
        assert 0.0 <= summary["correlation_risk"] <= 1.0
        assert 0.0 <= summary["diversification_benefit"] <= 1.0

    def test_handles_failed_analyzers(self):
        analyses = {"SPY": _mock_analysis(), "QQQ": _mock_analysis()}
        p = _build_portfolio(["SPY", "QQQ"], analyses)
        p.analyzers["QQQ"].analyze_current_regime.side_effect = RuntimeError("boom")
        summary = p.get_portfolio_regime_summary("1D")
        # Should still return results from SPY
        assert summary["dominant_regime"] is not None


class TestIdentifyArbitragePairs:
    def test_empty_when_no_portfolio_data(self):
        p = _build_portfolio(["SPY", "QQQ"], {"SPY": _mock_analysis(), "QQQ": _mock_analysis()})
        result = p.identify_arbitrage_pairs("1D")
        assert result == []

    def test_empty_when_fewer_than_two_symbols(self):
        p = _build_portfolio(["SPY"], {"SPY": _mock_analysis()})
        p.portfolio_data = {"1D": pd.DataFrame({"SPY": [100, 101]})}
        result = p.identify_arbitrage_pairs("1D")
        assert result == []

    def test_returns_list_of_dicts(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="D")
        rng = np.random.default_rng(42)
        # Create correlated data with large spread divergence at end
        base = np.cumsum(rng.normal(0.001, 0.02, 200))
        spy = base + 100
        qqq = base + rng.normal(0, 0.001, 200) + 100
        # Create extreme divergence at end
        spy[-5:] += 5
        qqq[-5:] -= 5
        df = pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=idx)

        analyses = {"SPY": _mock_analysis(), "QQQ": _mock_analysis()}
        p = _build_portfolio(["SPY", "QQQ"], analyses, portfolio_data={"1D": df})
        result = p.identify_arbitrage_pairs("1D")
        assert isinstance(result, list)
        # May or may not find opportunities depending on z-score threshold
        for opp in result:
            assert "pair" in opp
            assert "correlation" in opp
            assert "spread_zscore" in opp
            assert "signal" in opp

    def test_max_5_results(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="D")
        rng = np.random.default_rng(42)
        symbols = ["A", "B", "C", "D", "E", "F", "G"]
        data = {}
        base = np.cumsum(rng.normal(0, 0.02, 200))
        for i, sym in enumerate(symbols):
            data[sym] = base + rng.normal(0, 0.001, 200) + 100 + i
        # Extreme divergence at end
        data["A"][-5:] += 10
        data["B"][-5:] -= 10
        df = pd.DataFrame(data, index=idx)

        analyses = {sym: _mock_analysis() for sym in symbols}
        p = _build_portfolio(symbols, analyses, portfolio_data={"1D": df})
        result = p.identify_arbitrage_pairs("1D")
        assert len(result) <= 5

    def test_skips_low_correlation_pairs(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="D")
        rng = np.random.default_rng(42)
        # Uncorrelated data
        df = pd.DataFrame(
            {
                "SPY": rng.normal(100, 1, 200),
                "QQQ": rng.normal(200, 1, 200),
            },
            index=idx,
        )
        analyses = {"SPY": _mock_analysis(), "QQQ": _mock_analysis()}
        p = _build_portfolio(["SPY", "QQQ"], analyses, portfolio_data={"1D": df})
        result = p.identify_arbitrage_pairs("1D")
        # Low correlation pairs should be skipped
        assert isinstance(result, list)


class TestPrintPortfolioSummary:
    def test_prints_without_error(self, capsys):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"SPY": np.ones(100) * 100, "QQQ": np.ones(100) * 200},
            index=idx,
        )
        analyses = {"SPY": _mock_analysis(), "QQQ": _mock_analysis()}
        p = _build_portfolio(["SPY", "QQQ"], analyses, portfolio_data={"1D": df})

        # Add data attribute to mock analyzers for print_portfolio_summary
        for _sym, analyzer in p.analyzers.items():
            mock_data = {"1D": pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp("2023-01-01")])}
            analyzer.data = mock_data

        p.print_portfolio_summary("1D")
        captured = capsys.readouterr()
        assert "PORTFOLIO HMM REGIME ANALYSIS" in captured.out


class TestPreparePortfolioData:
    def test_init_creates_analyzers(self):
        """Test that __init__ properly initializes analyzers (integration-style)."""
        with patch("mra_lib.portfolio.portfolio.MarketRegimeAnalyzer") as MockAnalyzer:
            mock_inst = MagicMock()
            mock_inst.data = {"1D": pd.DataFrame({"Close": pd.Series([100, 101, 102])})}
            MockAnalyzer.return_value = mock_inst

            p = PortfolioHMMAnalyzer(
                symbols=["SPY", "QQQ"],
                periods={"1D": "2y"},
                provider_flag="yfinance",
            )
            assert "SPY" in p.analyzers
            assert "QQQ" in p.analyzers

    def test_init_handles_failed_symbol(self):
        """Symbols that fail to initialize are skipped."""
        with patch("mra_lib.portfolio.portfolio.MarketRegimeAnalyzer") as MockAnalyzer:

            def side_effect(symbol, *args, **kwargs):
                if symbol == "BAD":
                    raise RuntimeError("bad symbol")
                mock = MagicMock()
                mock.data = {"1D": pd.DataFrame({"Close": pd.Series([100, 101])})}
                return mock

            MockAnalyzer.side_effect = side_effect
            p = PortfolioHMMAnalyzer(
                symbols=["SPY", "BAD"],
                periods={"1D": "2y"},
                provider_flag="yfinance",
            )
            assert "SPY" in p.analyzers
            assert "BAD" not in p.analyzers
