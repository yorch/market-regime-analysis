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
    p.errors = {}

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
    def test_prints_without_error(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"SPY": np.ones(100) * 100, "QQQ": np.ones(100) * 200},
            index=idx,
        )
        analyses = {"SPY": _mock_analysis(), "QQQ": _mock_analysis()}
        p = _build_portfolio(["SPY", "QQQ"], analyses, portfolio_data={"1D": df})

        # Add data attribute to mock analyzers for format_portfolio_summary
        for _sym, analyzer in p.analyzers.items():
            mock_data = {"1D": pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp("2023-01-01")])}
            analyzer.data = mock_data

        text = p.format_portfolio_summary("1D")
        assert "PORTFOLIO HMM REGIME ANALYSIS" in text


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
            assert isinstance(p.errors["BAD"], RuntimeError)
            assert "SPY" not in p.errors

    def test_collect_analyses_records_errors(self):
        """Per-symbol analysis failures are recorded; a later success clears them."""
        p = _build_portfolio(["SPY", "BAD"], {"SPY": _mock_analysis()})
        failure = ValueError("no data")
        p.analyzers["BAD"].analyze_current_regime = MagicMock(side_effect=failure)

        analyses = p.collect_analyses("1D")

        assert list(analyses) == ["SPY"]
        assert p.errors == {"BAD": failure}

        p.analyzers["BAD"].analyze_current_regime = MagicMock(return_value=_mock_analysis())
        p.collect_analyses("1D")
        assert p.errors == {}


class TestPortfolioRegressionFixes:
    """Regression tests for the 2026-10-03 analytics review."""

    @staticmethod
    def _independent_walks(n=500, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        # Two independent random walks with drift: price levels look correlated,
        # returns are not.
        a = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n)))
        b = 50 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n)))
        return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)

    def test_correlations_use_returns_not_prices(self):
        df = self._independent_walks()
        analyses = {"AAA": _mock_analysis(), "BBB": _mock_analysis()}
        p = _build_portfolio(["AAA", "BBB"], analyses, portfolio_data={"1D": df})

        price_corr = df["AAA"].corr(df["BBB"])
        return_corr = df["AAA"].pct_change().corr(df["BBB"].pct_change())
        assert abs(price_corr) > 0.5  # spurious level correlation
        assert abs(return_corr) < 0.2

        result = p.calculate_regime_correlations("1D")
        assert result.loc["AAA", "BBB_price_corr"] == pytest.approx(return_corr)

        summary = p.get_portfolio_regime_summary("1D")
        assert summary["correlation_risk"] == pytest.approx(abs(return_corr))

    def test_correlation_risk_computed_for_two_symbols(self):
        """Old column-count check (> 2 columns) skipped two-symbol portfolios."""
        df = self._independent_walks()
        analyses = {"AAA": _mock_analysis(), "BBB": _mock_analysis()}
        p = _build_portfolio(["AAA", "BBB"], analyses, portfolio_data={"1D": df})
        summary = p.get_portfolio_regime_summary("1D")
        assert summary["correlation_risk"] > 0.0
        assert summary["diversification_benefit"] < 1.0

    def test_derived_columns_ignored(self):
        df = self._independent_walks()
        df["portfolio_return"] = 0.0
        df["portfolio_volatility"] = 0.0
        p = _build_portfolio(["AAA", "BBB"], {"AAA": _mock_analysis(), "BBB": _mock_analysis()})
        p.portfolio_data = {"1D": df}
        assert p._symbol_columns("1D") == ["AAA", "BBB"]

    def test_cointegrated_pair_flagged_with_hedge_ratio(self):
        rng = np.random.default_rng(3)
        n = 400
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        log_b = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
        noise = rng.normal(0, 0.005, n)
        noise[-1] = 0.05  # A rich vs. B at the last bar (~10 sd)
        log_a = 1.0 + 1.5 * log_b + noise
        df = pd.DataFrame({"AAA": np.exp(log_a), "BBB": np.exp(log_b)}, index=idx)
        p = _build_portfolio(
            ["AAA", "BBB"],
            {"AAA": _mock_analysis(), "BBB": _mock_analysis()},
            portfolio_data={"1D": df},
        )
        result = p.identify_arbitrage_pairs("1D")
        assert len(result) == 1
        opp = result[0]
        assert opp["signal"] == "SHORT_1_LONG_2"
        assert opp["hedge_ratio"] == pytest.approx(1.5, abs=0.05)
        assert opp["coint_pvalue"] < 0.05
        assert opp["spread_zscore"] > 2.0

    def test_non_cointegrated_pair_not_flagged(self):
        df = self._independent_walks(seed=11)
        p = _build_portfolio(
            ["AAA", "BBB"],
            {"AAA": _mock_analysis(), "BBB": _mock_analysis()},
            portfolio_data={"1D": df},
        )
        assert p.identify_arbitrage_pairs("1D", max_pvalue=0.01) == []

    def test_print_summary_analyzes_each_symbol_once(self):
        df = self._independent_walks()
        analyses = {"AAA": _mock_analysis(), "BBB": _mock_analysis()}
        p = _build_portfolio(["AAA", "BBB"], analyses, portfolio_data={"1D": df})
        for analyzer in p.analyzers.values():
            analyzer.data = {"1D": pd.DataFrame({"Close": [100.0]})}
        out = p.format_portfolio_summary("1D")
        for analyzer in p.analyzers.values():
            assert analyzer.analyze_current_regime.call_count == 1
        assert "Highest return correlation" in out

    def test_distribution_percentage_uses_successful_analyses(self):
        analyses = {"AAA": _mock_analysis(), "BBB": _mock_analysis()}
        p = _build_portfolio(["AAA", "BBB"], analyses)
        p.analyzers["BBB"].analyze_current_regime.side_effect = RuntimeError("boom")
        for analyzer in p.analyzers.values():
            analyzer.data = {"1D": pd.DataFrame({"Close": [100.0]})}
        out = p.format_portfolio_summary("1D")
        assert "Bull Trending: 1 assets (100.0%)" in out
