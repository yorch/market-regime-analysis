"""End-to-end CLI tests using the offline ``mock`` provider (no network, no keys)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mra_cli.main import cli

KEY_VARS = (
    "ALPHA_VANTAGE_API_KEY",
    "ALPHAVANTAGE_API_KEY",
    "POLYGON_API_KEY",
    "TIINGO_API_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "DEFAULT_PROVIDER",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestNoKeyCommands:
    def test_cli_list_providers_needs_no_key(self, runner):
        result = runner.invoke(cli, ["list-providers"])
        assert result.exit_code == 0, result.output
        assert "MOCK" in result.output

    def test_cli_subcommand_help_needs_no_key(self, runner):
        result = runner.invoke(cli, ["--provider", "alphavantage", "detailed-analysis", "--help"])
        assert result.exit_code == 0, result.output
        assert "--provider" in result.output

    def test_cli_position_sizing_needs_no_key(self, runner):
        result = runner.invoke(cli, ["position-sizing"])
        assert result.exit_code == 0, result.output

    def test_cli_missing_key_fails_lazily(self, runner):
        result = runner.invoke(cli, ["detailed-analysis", "--provider", "polygon"])
        assert result.exit_code == 1
        assert "POLYGON_API_KEY" in result.output

    def test_cli_provider_choices_from_registry(self, runner):
        result = runner.invoke(cli, ["detailed-analysis", "--provider", "nope"])
        assert result.exit_code == 2
        assert "mock" in result.output


class TestProviderResolution:
    def test_cli_default_provider_env(self, runner, monkeypatch):
        monkeypatch.setenv("DEFAULT_PROVIDER", "mock")
        result = runner.invoke(cli, ["detailed-analysis"])
        assert result.exit_code == 0, result.output

    def test_cli_unknown_default_provider_env(self, runner, monkeypatch):
        monkeypatch.setenv("DEFAULT_PROVIDER", "bogus")
        result = runner.invoke(cli, ["detailed-analysis"])
        assert result.exit_code == 1
        assert "Unknown provider 'bogus'" in result.output

    def test_cli_group_level_provider_still_works(self, runner):
        result = runner.invoke(cli, ["--provider", "mock", "detailed-analysis"])
        assert result.exit_code == 0, result.output

    def test_cli_default_is_yfinance(self, runner):
        with patch("mra_cli.main._analyze_single_timeframe", side_effect=ValueError("x")) as f:
            runner.invoke(cli, ["detailed-analysis"])
        assert f.call_args.args[2] == "yfinance"

    def test_cli_subcommand_provider_overrides_group(self, runner):
        with patch("mra_cli.main._analyze_single_timeframe", side_effect=ValueError("x")) as f:
            runner.invoke(cli, ["--provider", "polygon", "detailed-analysis", "--provider", "mock"])
        assert f.call_args.args[2] == "mock"


class TestAnalysisCommands:
    def test_cli_detailed_analysis_mock(self, runner):
        result = runner.invoke(
            cli, ["detailed-analysis", "--provider", "mock", "--timeframe", "15m"]
        )
        assert result.exit_code == 0, result.output
        assert "HMM MARKET REGIME ANALYSIS - SPY (15m)" in result.output
        assert "DETAILED METRICS" in result.output

    def test_cli_detailed_analysis_failure_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["detailed-analysis", "--provider", "mock", "--symbol", " "])
        assert result.exit_code == 1

    def test_cli_current_analysis_mock_all_timeframes(self, runner):
        result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        for tf in ("1D", "1H", "15m"):
            assert f"SPY ({tf})" in result.output

    def test_cli_current_analysis_analyzes_once_per_timeframe(self, runner):
        from mra_lib import MarketRegimeAnalyzer

        original = MarketRegimeAnalyzer.analyze_current_regime
        with patch.object(
            MarketRegimeAnalyzer, "analyze_current_regime", autospec=True, side_effect=original
        ) as spy:
            result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        assert spy.call_count == 3

    def test_cli_current_analysis_partial_failure_exits_zero(self, runner):
        from mra_cli import main

        real = main._analyze_single_timeframe

        def flaky(symbol, tf, provider, key):
            if tf == "15m":
                raise ConnectionError("boom")
            return real(symbol, tf, provider, key)

        with patch("mra_cli.main._analyze_single_timeframe", side_effect=flaky):
            result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        assert "Error analyzing 15m" in result.output

    def test_cli_current_analysis_all_fail_exits_nonzero(self, runner):
        with patch("mra_cli.main._analyze_single_timeframe", side_effect=ConnectionError("down")):
            result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 1
        assert "failed for every timeframe" in result.output

    def test_cli_debug_reraises(self, runner):
        with patch("mra_cli.main._analyze_single_timeframe", side_effect=RuntimeError("kaboom")):
            result = runner.invoke(cli, ["--debug", "detailed-analysis", "--provider", "mock"])
        assert isinstance(result.exception, RuntimeError)

    def test_cli_regime_forecast_mock(self, runner):
        result = runner.invoke(
            cli, ["regime-forecast", "--provider", "mock", "--steps", "2", "--n-states", "3"]
        )
        assert result.exit_code == 0, result.output
        assert "REGIME FORECAST (2-step ahead)" in result.output

    @pytest.mark.parametrize("args", [["--steps", "0"], ["--n-states", "1"], ["--n-states", "50"]])
    def test_cli_regime_forecast_rejects_bad_ranges(self, runner, args):
        result = runner.invoke(cli, ["regime-forecast", "--provider", "mock", *args])
        assert result.exit_code == 2

    def test_cli_multi_symbol_analysis_mock(self, runner):
        result = runner.invoke(
            cli, ["multi-symbol-analysis", "--provider", "mock", "--symbols", "SPY,QQQ"]
        )
        assert result.exit_code == 0, result.output


class TestOutputCommands:
    def test_cli_generate_charts_output(self, runner, tmp_path: Path):
        out = tmp_path / "chart.png"
        result = runner.invoke(
            cli,
            ["generate-charts", "--provider", "mock", "--days", "30", "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 0

    def test_cli_generate_charts_library_failure_exits_nonzero(self, runner, tmp_path: Path):
        from mra_lib import MarketRegimeAnalyzer

        # The library swallows errors and returns None without drawing anything
        with patch.object(MarketRegimeAnalyzer, "plot_regime_analysis", return_value=None):
            result = runner.invoke(
                cli,
                ["generate-charts", "--provider", "mock", "-o", str(tmp_path / "c.png")],
            )
        assert result.exit_code == 1
        assert "Chart generation failed" in result.output

    def test_cli_generate_charts_accepts_png_bytes(self, runner, tmp_path: Path):
        from mra_lib import MarketRegimeAnalyzer

        out = tmp_path / "c.png"
        with patch.object(MarketRegimeAnalyzer, "plot_regime_analysis", return_value=b"PNG"):
            result = runner.invoke(cli, ["generate-charts", "--provider", "mock", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.read_bytes() == b"PNG"

    def test_cli_export_csv(self, runner, tmp_path: Path):
        out = tmp_path / "a.csv"
        result = runner.invoke(cli, ["export-csv", "--provider", "mock", "--filename", str(out)])
        assert result.exit_code == 0, result.output
        assert "timeframe" in out.read_text()

    def test_cli_export_csv_library_failure_exits_nonzero(self, runner, tmp_path: Path):
        from mra_lib import MarketRegimeAnalyzer

        out = tmp_path / "a.csv"
        with patch.object(MarketRegimeAnalyzer, "export_analysis_to_csv", return_value=None):
            result = runner.invoke(
                cli, ["export-csv", "--provider", "mock", "--filename", str(out)]
            )
        assert result.exit_code == 1
        assert "was not written" in result.output

    def test_cli_export_csv_writes_returned_dataframe(self, runner, tmp_path: Path):
        import pandas as pd

        from mra_lib import MarketRegimeAnalyzer

        out = tmp_path / "a.csv"
        frame = pd.DataFrame([{"timeframe": "1D", "regime": "Bull Trending"}])
        with patch.object(MarketRegimeAnalyzer, "export_analysis_to_csv", return_value=frame):
            result = runner.invoke(
                cli, ["export-csv", "--provider", "mock", "--filename", str(out)]
            )
        assert result.exit_code == 0, result.output
        assert "Bull Trending" in out.read_text()

    def test_cli_export_csv_library_raises_exits_nonzero(self, runner, tmp_path: Path):
        from mra_lib import MarketRegimeAnalyzer

        with patch.object(
            MarketRegimeAnalyzer, "export_analysis_to_csv", side_effect=OSError("disk full")
        ):
            result = runner.invoke(
                cli,
                ["export-csv", "--provider", "mock", "--filename", str(tmp_path / "a.csv")],
            )
        assert result.exit_code == 1


class TestMonitoringAndServer:
    def test_cli_continuous_monitoring_once(self, runner):
        result = runner.invoke(cli, ["continuous-monitoring", "--provider", "mock", "--once"])
        assert result.exit_code == 0, result.output
        assert "SPY (15m)" in result.output

    def test_cli_continuous_monitoring_no_success_exits_nonzero(self, runner):
        from mra_lib import MarketRegimeAnalyzer

        with patch.object(MarketRegimeAnalyzer, "run_continuous_monitoring", return_value=0):
            result = runner.invoke(cli, ["continuous-monitoring", "--provider", "mock", "--once"])
        assert result.exit_code == 1

    def test_cli_start_api_defaults_to_localhost(self, runner):
        with patch("uvicorn.run") as run:
            result = runner.invoke(cli, ["start-api"])
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["host"] == "127.0.0.1"
