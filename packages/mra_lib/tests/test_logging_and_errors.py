"""The library logs instead of printing, raises typed errors, and keeps print_* shims."""

import logging
import math
from unittest.mock import patch

import pandas as pd
import pytest

import mra_lib
from mra_lib import MarketRegimeAnalyzer, PortfolioHMMAnalyzer
from mra_lib.backtesting import BacktestEngine, PerformanceMetrics, StrategyOptimizer
from mra_lib.backtesting.calibrator import CalibrationResult, RegimeTradeStats
from mra_lib.config.enums import MarketRegime
from mra_lib.data_providers import (
    AuthError,
    InvalidSymbolError,
    MockDataProvider,
    ProviderError,
    RateLimitError,
)
from mra_lib.errors import (
    DataLoadError,
    InsufficientDataError,
    ModelNotFittedError,
    MRAError,
)

PERIODS = {"1D": "2y"}


@pytest.fixture(scope="module")
def analyzer() -> MarketRegimeAnalyzer:
    return MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        ("cls", "builtin"),
        [
            (DataLoadError, ValueError),
            (InsufficientDataError, ValueError),
            (ModelNotFittedError, ValueError),
            (InvalidSymbolError, ValueError),
            (AuthError, ConnectionError),
            (RateLimitError, ConnectionError),
        ],
    )
    def test_library_errors_keep_builtin_bases(self, cls, builtin):
        assert issubclass(cls, MRAError)
        assert issubclass(cls, builtin)

    def test_provider_errors_are_one_hierarchy(self):
        for cls in (InvalidSymbolError, AuthError, RateLimitError):
            assert issubclass(cls, ProviderError)
        assert issubclass(ProviderError, MRAError)
        assert mra_lib.ProviderError is ProviderError
        assert mra_lib.MRAError is MRAError


class TestLoadDataErrors:
    @pytest.mark.parametrize(
        "cls", [InvalidSymbolError, AuthError, RateLimitError, ConnectionError]
    )
    def test_provider_errors_propagate_unchanged(self, cls):
        error = cls("upstream said no")
        with (
            patch.object(MockDataProvider, "fetch", side_effect=error),
            pytest.raises(cls) as info,
        ):
            MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")
        assert info.value is error

    def test_timeout_is_raised_as_connection_error(self):
        with (
            patch.object(MockDataProvider, "fetch", side_effect=TimeoutError("slow")),
            pytest.raises(ConnectionError, match="timed out") as info,
        ):
            MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")
        assert isinstance(info.value.__cause__, TimeoutError)

    def test_unexpected_provider_failure_is_data_load_error(self):
        with (
            patch.object(MockDataProvider, "fetch", side_effect=KeyError("Close")),
            pytest.raises(DataLoadError, match="Data loading failed for 1D") as info,
        ):
            MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")
        assert isinstance(info.value.__cause__, KeyError)

    def test_empty_frame_is_data_load_error(self):
        with (
            patch.object(MockDataProvider, "fetch", return_value=pd.DataFrame()),
            pytest.raises(DataLoadError, match="No data available"),
        ):
            MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")

    def test_missing_columns_is_data_load_error(self):
        frame = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
        with (
            patch.object(MockDataProvider, "fetch", return_value=frame),
            pytest.raises(DataLoadError, match="Missing columns"),
        ):
            MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")

    def test_missing_model_is_model_not_fitted_error(self, analyzer):
        model = analyzer.hmm_models.pop("1D")
        try:
            with pytest.raises(ModelNotFittedError):
                analyzer.analyze_current_regime("1D")
        finally:
            analyzer.hmm_models["1D"] = model


class TestNoPrinting:
    def test_package_logger_has_null_handler(self):
        handlers = logging.getLogger("mra_lib").handlers
        assert any(isinstance(h, logging.NullHandler) for h in handlers)

    def test_analyzer_logs_progress_instead_of_printing(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="mra_lib"):
            MarketRegimeAnalyzer("SPY", periods=PERIODS, provider_flag="mock")
        assert capsys.readouterr().out == ""
        messages = [r.getMessage() for r in caplog.records if r.name == "mra_lib.analyzer"]
        assert "Loading data for SPY..." in messages
        assert any(m.startswith("✓ Loaded") and m.endswith("bars for 1D") for m in messages)
        assert "✓ Trained HMM for 1D" in messages

    def test_portfolio_records_failed_symbols(self, capsys, caplog):
        def fetch(self, symbol, period, interval):
            if symbol == "BAD":
                raise InvalidSymbolError("no such symbol")
            return original(self, symbol, period, interval)

        original = MockDataProvider.fetch
        with (
            patch.object(MockDataProvider, "fetch", fetch),
            caplog.at_level(logging.INFO, logger="mra_lib"),
        ):
            portfolio = PortfolioHMMAnalyzer(["SPY", "BAD"], periods=PERIODS, provider_flag="mock")
        assert capsys.readouterr().out == ""
        assert list(portfolio.analyzers) == ["SPY"]
        assert isinstance(portfolio.failed_symbols["BAD"], InvalidSymbolError)
        assert any("Failed to initialize BAD" in r.getMessage() for r in caplog.records)

    def test_portfolio_all_failed_reraises_shared_root_cause(self):
        with (
            patch.object(MockDataProvider, "fetch", side_effect=AuthError("bad key")),
            pytest.raises(AuthError, match="bad key"),
        ):
            PortfolioHMMAnalyzer(["AAA", "BBB"], periods=PERIODS, provider_flag="mock")

    @pytest.mark.parametrize(
        ("errors", "expected"),
        [
            ((RateLimitError("slow down"), ConnectionError("down")), RateLimitError),
            ((InvalidSymbolError("AAA"), AuthError("bad key")), AuthError),
            ((InvalidSymbolError("AAA"), ConnectionError("down")), ConnectionError),
            ((InvalidSymbolError("AAA"), InvalidSymbolError("BBB")), InvalidSymbolError),
        ],
    )
    def test_portfolio_all_failed_raises_most_actionable_cause(self, errors, expected):
        by_symbol = dict(zip(["AAA", "BBB"], errors, strict=True))

        def fetch(self, symbol, period, interval):
            raise by_symbol[symbol]

        with patch.object(MockDataProvider, "fetch", fetch), pytest.raises(expected):
            PortfolioHMMAnalyzer(["AAA", "BBB"], periods=PERIODS, provider_flag="mock")

    def test_portfolio_all_failed_unclassified_is_data_load_error(self):
        def fetch(self, symbol, period, interval):
            raise InvalidSymbolError(symbol) if symbol == "AAA" else KeyError("Close")

        with (
            patch.object(MockDataProvider, "fetch", fetch),
            pytest.raises(DataLoadError, match="AAA: AAA") as info,
        ):
            PortfolioHMMAnalyzer(["AAA", "BBB"], periods=PERIODS, provider_flag="mock")
        assert isinstance(info.value.__cause__, InvalidSymbolError)

    def test_portfolio_records_analysis_failures(self):
        portfolio = PortfolioHMMAnalyzer(["SPY"], periods=PERIODS, provider_flag="mock")
        portfolio.analyzers["SPY"].hmm_models.clear()
        assert portfolio.collect_analyses("1D") == {}
        assert isinstance(portfolio.analysis_failures["SPY"], ModelNotFittedError)

    def test_monitoring_default_logs_report(self, analyzer, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="mra_lib"):
            successes = analyzer.run_continuous_monitoring(1, max_iterations=1)
        assert successes == 1
        assert capsys.readouterr().out == ""
        assert any(
            "HMM MARKET REGIME ANALYSIS - SPY (1D)" in r.getMessage() for r in caplog.records
        )


class TestReports:
    def test_format_analysis_report(self, analyzer):
        text = analyzer.format_analysis_report("1D")
        assert text.startswith("\n" + "=" * 80)
        assert "HMM MARKET REGIME ANALYSIS - SPY (1D)" in text
        assert "📊 REGIME CLASSIFICATION:" in text
        assert text.endswith("=" * 80)

    def test_format_analysis_report_raises_instead_of_printing(self, analyzer):
        with pytest.raises(ValueError, match="not available"):
            analyzer.format_analysis_report("4H")

    def test_export_returns_path(self, analyzer, tmp_path):
        target = tmp_path / "out.csv"
        assert analyzer.export_analysis_to_csv(str(target)) == str(target)
        assert len(pd.read_csv(target)) == 1

    def test_export_with_nothing_to_export_raises(self, analyzer, tmp_path):
        target = tmp_path / "out.csv"
        with (
            patch.object(
                MarketRegimeAnalyzer, "build_export_dataframe", return_value=pd.DataFrame()
            ),
            pytest.raises(InsufficientDataError),
        ):
            analyzer.export_analysis_to_csv(str(target))
        assert not target.exists()

    def test_render_chart_with_too_little_data_raises(self, analyzer):
        with pytest.raises(InsufficientDataError):
            analyzer.render_regime_chart("1D", days=5)

    def _calibration_result(self, holdout: dict | None) -> CalibrationResult:
        regime = MarketRegime.BULL_TRENDING
        stats = RegimeTradeStats(
            regime=regime,
            n_trades=4,
            win_rate=0.5,
            avg_win=10.0,
            avg_loss=5.0,
            profit_factor=2.0,
            sharpe=1.1,
            kelly_fraction=0.25,
        )
        return CalibrationResult(
            multipliers={regime: 1.5},
            regime_stats={regime: stats},
            method="sharpe_weighted",
            total_trades=4,
            trades_per_regime={regime: 4},
            baseline_sharpe=0.3,
            raw_scores={regime: 1.1},
            in_sample=holdout is None,
            holdout_metrics=holdout,
        )

    def test_calibration_report_in_sample(self):
        text = self._calibration_result(None).format_report()
        assert "REGIME MULTIPLIER CALIBRATION RESULTS" in text
        assert "Method: sharpe_weighted" in text
        assert "Bull Trending" in text and "###############" in text
        assert "NOTE: IN-SAMPLE calibration" in text

    def test_calibration_report_holdout(self):
        holdout = {
            "compounded_strategy_return": 0.05,
            "compounded_bh_return": 0.07,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.04,
            "total_trades": 6,
            "profit_factor": math.nan,
        }
        text = self._calibration_result(holdout).format_report()
        assert "HOLDOUT (out-of-sample) evaluation" in text
        assert "Profit Factor: n/a" in text
        assert "IN-SAMPLE calibration" not in text


class TestDeprecatedPrintShims:
    def _assert_shim(self, capsys, call, expected: str) -> None:
        with pytest.warns(DeprecationWarning, match="is deprecated"):
            call()
        out = capsys.readouterr().out
        assert out == expected + "\n"

    def test_print_analysis_report(self, analyzer, capsys):
        with patch("mra_lib.analyzer.datetime") as clock:
            clock.now.return_value.strftime.return_value = "NOW"
            expected = analyzer.format_analysis_report("1D")
            self._assert_shim(capsys, lambda: analyzer.print_analysis_report("1D"), expected)

    def test_print_portfolio_summary(self, capsys):
        portfolio = PortfolioHMMAnalyzer(["SPY"], periods=PERIODS, provider_flag="mock")
        expected = portfolio.format_portfolio_summary("1D")
        self._assert_shim(capsys, lambda: portfolio.print_portfolio_summary("1D"), expected)

    def test_print_results_and_summary(self, capsys):
        equity = pd.Series(
            [100_000.0, 101_000.0, 102_000.0], index=pd.bdate_range("2020-01-01", periods=3)
        )
        trades = [
            {
                "entry_date": pd.Timestamp("2020-01-01"),
                "exit_date": pd.Timestamp("2020-01-03"),
                "entry_price": 100.0,
                "exit_price": 110.0,
                "shares": 100,
                "direction": "LONG",
                "pnl": 1000.0,
            }
        ] * 2
        metrics = PerformanceMetrics(trades, equity)
        engine = BacktestEngine(initial_capital=100_000)
        results = {
            "final_capital": 102_000.0,
            "total_return": 0.02,
            "trades": trades,
            "performance": metrics,
        }
        text = engine.format_results(results)
        assert metrics.format_summary() in text
        self._assert_shim(capsys, lambda: engine.print_results(results), text)
        self._assert_shim(capsys, metrics.print_summary, metrics.format_summary())

    def test_print_top_results(self, capsys):
        optimizer = StrategyOptimizer(pd.DataFrame({"Close": [1.0]}))
        expected = optimizer.format_top_results()
        assert "No results to display." in expected
        self._assert_shim(capsys, optimizer.print_top_results, expected)
