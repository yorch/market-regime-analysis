"""End-to-end CLI tests using the offline ``mock`` provider (no network, no keys)."""

import os
from datetime import datetime
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

    def test_cli_current_analysis_prints_confirmation(self, runner):
        result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        assert "MULTI-TIMEFRAME CONFIRMATION - SPY" in result.output
        assert "Direction:" in result.output
        assert "Agreement:" in result.output
        # The section comes after every per-timeframe report
        assert result.output.index("MULTI-TIMEFRAME") > result.output.index("SPY (15m)")

    def test_cli_current_analysis_confirmation_needs_two_timeframes(self, runner):
        from mra_cli import main

        real = main._analyze_single_timeframe

        def only_daily(symbol, tf, provider, key):
            if tf != "1D":
                raise ConnectionError("boom")
            return real(symbol, tf, provider, key)

        with patch("mra_cli.main._analyze_single_timeframe", side_effect=only_daily):
            result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        assert "MULTI-TIMEFRAME CONFIRMATION" not in result.output

    def test_cli_current_analysis_confirmation_with_partial_failure(self, runner):
        from mra_cli import main

        real = main._analyze_single_timeframe

        def flaky(symbol, tf, provider, key):
            if tf == "1H":
                raise ConnectionError("boom")
            return real(symbol, tf, provider, key)

        with patch("mra_cli.main._analyze_single_timeframe", side_effect=flaky):
            result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        assert "MULTI-TIMEFRAME CONFIRMATION" in result.output
        assert "Unavailable: 1H" in result.output

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

        out = tmp_path / "c.png"
        with patch.object(
            MarketRegimeAnalyzer,
            "render_regime_chart",
            side_effect=ValueError("Insufficient data for plotting"),
        ):
            result = runner.invoke(cli, ["generate-charts", "--provider", "mock", "-o", str(out)])
        assert result.exit_code == 1
        assert "Chart generation failed" in result.output
        assert "Insufficient data" in result.output
        assert not out.exists()

    def test_cli_generate_charts_format_from_extension(self, runner, tmp_path: Path):
        out = tmp_path / "chart.svg"
        result = runner.invoke(
            cli, ["generate-charts", "--provider", "mock", "--days", "30", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert out.read_text().lstrip().startswith("<?xml")

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

    def test_cli_start_api_defaults_to_localhost(self, runner, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("JWT_SECRET", "cli-test-secret-0123456789abcdefghijklmnop")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        monkeypatch.delenv("API_HOST", raising=False)
        monkeypatch.delenv("API_PORT", raising=False)
        with patch("uvicorn.run") as run:
            result = runner.invoke(cli, ["start-api"])
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["host"] == "127.0.0.1"
        assert run.call_args.kwargs["port"] == 8000
        assert run.call_args.kwargs["reload"] is False

    def test_cli_start_api_honours_env(self, runner, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("JWT_SECRET", "cli-test-secret-0123456789abcdefghijklmnop")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("API_PORT", "9001")
        with patch("uvicorn.run") as run:
            result = runner.invoke(cli, ["start-api"])
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["port"] == 9001
        assert run.call_args.kwargs["log_level"] == "warning"
        assert os.environ["LOG_LEVEL"] == "WARNING"

    def test_cli_start_api_dev_sets_development_environment(self, runner, monkeypatch):
        # --dev must start without JWT_SECRET, like `mra-api --dev`
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        with patch("uvicorn.run") as run:
            result = runner.invoke(cli, ["start-api", "--dev"])
        assert result.exit_code == 0, result.output
        assert os.environ["ENVIRONMENT"] == "development"
        assert run.call_args.kwargs["reload"] is True
        assert run.call_args.kwargs["log_level"] == "debug"

    def test_cli_start_api_refuses_production_without_secret(self, runner, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        with patch("uvicorn.run") as run:
            result = runner.invoke(cli, ["start-api"])
        assert result.exit_code == 2
        run.assert_not_called()


class TestReviewFixes:
    @pytest.mark.parametrize(
        ("error", "label"),
        [
            ("AuthError", "Authentication error"),
            ("RateLimitError", "Rate limited"),
            ("InvalidSymbolError", "Unknown symbol / no data"),
            ("ConnectionError", "Network error"),
        ],
    )
    def test_cli_labels_provider_errors_raised_by_analyzer(self, runner, error, label):
        # The analyzer lets provider errors propagate unchanged, so the CLI labels them
        # by type (no cause-chain walking)
        from mra_lib import data_providers
        from mra_lib.data_providers import MockDataProvider

        cls = getattr(data_providers, error, None) or ConnectionError
        with patch.object(MockDataProvider, "fetch", side_effect=cls("boom")):
            result = runner.invoke(cli, ["detailed-analysis", "--provider", "mock"])
        assert result.exit_code == 1
        assert label in result.output
        assert "boom" in result.output

    def test_cli_group_key_not_reused_for_other_provider(self, runner):
        result = runner.invoke(
            cli,
            [
                "--provider",
                "tiingo",
                "--api-key",
                "T",
                "detailed-analysis",
                "--provider",
                "polygon",
            ],
        )
        assert result.exit_code == 1
        assert "POLYGON_API_KEY" in result.output

    def test_cli_generate_charts_interactive_shows_and_closes(self, runner, monkeypatch):
        import matplotlib
        import matplotlib.pyplot as plt

        monkeypatch.setattr(matplotlib, "get_backend", lambda: "macosx")
        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(plt.get_fignums()))
        before = set(plt.get_fignums())

        result = runner.invoke(cli, ["generate-charts", "--provider", "mock", "--days", "30"])

        assert result.exit_code == 0, result.output
        assert len(shown) == 1 and shown[0]  # a figure was open when show() ran
        assert set(plt.get_fignums()) == before  # and closed afterwards

    def test_cli_monitoring_retries_failed_startup(self, runner):
        from mra_lib import MarketRegimeAnalyzer

        real_init = MarketRegimeAnalyzer.__init__
        calls = {"n": 0}

        def flaky_init(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("network down")
            real_init(self, *args, **kwargs)

        with (
            patch.object(MarketRegimeAnalyzer, "__init__", flaky_init),
            patch("mra_cli.main.time.sleep") as sleep,
        ):
            result = runner.invoke(
                cli,
                ["continuous-monitoring", "--provider", "mock", "--max-iterations", "2"],
            )
        assert result.exit_code == 0, result.output
        assert "Startup failed" in result.output
        sleep.assert_called_once_with(300)


def test_json_safe_replaces_non_finite_floats():
    import json

    import numpy as np

    from mra_cli.main import _json_safe

    data = {"pf": float("inf"), "nan": np.float64("nan"), "ok": [1.5, np.int64(2)], "s": "x"}
    safe = _json_safe(data)
    assert safe == {"pf": None, "nan": None, "ok": [1.5, 2], "s": "x"}
    json.dumps(safe, allow_nan=False)


def test_cli_calibrate_multipliers_writes_valid_json(runner, tmp_path: Path):
    import json
    from types import SimpleNamespace

    from mra_lib.config.enums import MarketRegime

    stats = SimpleNamespace(
        n_trades=3,
        win_rate=1.0,
        avg_pnl=0.01,
        sharpe=float("nan"),
        profit_factor=float("inf"),
        kelly_fraction=0.1,
    )
    regime = MarketRegime.BULL_TRENDING
    fake = SimpleNamespace(
        baseline_sharpe=0.5,
        total_trades=3,
        multipliers={regime: 1.2},
        trades_per_regime={regime: 3},
        raw_scores={regime: float("inf")},
        regime_stats={regime: stats},
        format_report=lambda: "CALIBRATION REPORT",
    )
    out = tmp_path / "cal.json"
    with patch("mra_cli.main.RegimeMultiplierCalibrator") as calibrator:
        calibrator.return_value.calibrate_with_details.return_value = fake
        result = runner.invoke(
            cli, ["calibrate-multipliers", "--provider", "mock", "--output", str(out)]
        )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["regime_stats"]["Bull Trending"]["profit_factor"] is None
    assert data["raw_scores"]["Bull Trending"] is None


def test_cli_shows_library_progress_from_logging(runner):
    # mra_lib logs (never prints); the CLI renders INFO records as plain progress lines
    result = runner.invoke(cli, ["detailed-analysis", "--provider", "mock"])
    assert result.exit_code == 0, result.output
    assert "Loading data for SPY..." in result.stderr
    assert "✓ Trained HMM for 1D" in result.stderr
    assert "INFO" not in result.output
    assert "HMM MARKET REGIME ANALYSIS - SPY (1D)" in result.stdout


def test_cli_export_csv_with_no_data_fails(runner, tmp_path):
    import pandas as pd

    from mra_lib import MarketRegimeAnalyzer

    with patch.object(
        MarketRegimeAnalyzer,
        "build_export_dataframe",
        return_value=pd.DataFrame(),
    ):
        result = runner.invoke(
            cli, ["export-csv", "--provider", "mock", "--filename", str(tmp_path / "x.csv")]
        )
    assert result.exit_code == 1
    assert "No analysis data to export" in result.output


class TestRegimeHistory:
    """``current-analysis --record`` and ``history`` against a temp database."""

    @pytest.fixture(autouse=True)
    def db_path(self, tmp_path, monkeypatch) -> Path:
        path = tmp_path / "mra" / "regimes.db"
        monkeypatch.setenv("MRA_DB_PATH", str(path))
        return path

    def test_cli_current_analysis_without_record_writes_nothing(self, runner, db_path):
        result = runner.invoke(cli, ["current-analysis", "--provider", "mock"])
        assert result.exit_code == 0, result.output
        assert not db_path.exists()

    def test_cli_current_analysis_record_saves_each_timeframe(self, runner, db_path):
        from mra_lib.config.timeframes import TIMEFRAMES
        from mra_lib.storage import SQLiteRegimeStore

        result = runner.invoke(
            cli, ["current-analysis", "--provider", "mock", "--symbol", "spy", "--record"]
        )
        assert result.exit_code == 0, result.output
        assert f"Recorded {len(TIMEFRAMES)} analysis record(s)" in result.output

        store = SQLiteRegimeStore(db_path)
        records = store.history("SPY")
        assert sorted(r.timeframe for r in records) == sorted(TIMEFRAMES)
        assert {r.provider for r in records} == {"mock"}
        assert store.symbols() == ["SPY"]

        # Re-running upserts on (timeframe, bar_time) instead of duplicating
        result = runner.invoke(
            cli, ["current-analysis", "--provider", "mock", "--symbol", "SPY", "--record"]
        )
        assert result.exit_code == 0, result.output
        records = store.history("SPY")
        assert len({(r.timeframe, r.bar_time) for r in records}) == len(records)

    def test_cli_record_failure_does_not_fail_analysis(self, runner):
        from mra_lib.errors import StorageError
        from mra_lib.storage import SQLiteRegimeStore

        with patch.object(SQLiteRegimeStore, "save", side_effect=StorageError("disk full")):
            result = runner.invoke(cli, ["current-analysis", "--provider", "mock", "--record"])
        assert result.exit_code == 0, result.output
        assert "Could not record 1D analysis" in result.stderr
        assert "Recorded" not in result.output

    def test_cli_history_empty(self, runner):
        result = runner.invoke(cli, ["history", "--symbol", "SPY"])
        assert result.exit_code == 0, result.output
        assert "No recorded regime history for SPY" in result.output

    def test_cli_history_table_and_json(self, runner):
        import json

        from mra_lib.storage import RegimeRecord, default_store

        store = default_store()
        for day in (1, 2, 3):
            store.save(
                RegimeRecord(
                    symbol="SPY",
                    timeframe="1D",
                    bar_time=datetime(2026, 1, day),
                    regime="Bull Trending",
                    confidence=0.75,
                    persistence=0.5,
                    transition_probability=0.9,
                    recommended_strategy="Trend Following",
                    provider="mock",
                    close=100.0 + day,
                )
            )

        result = runner.invoke(cli, ["history", "--symbol", "spy", "--timeframe", "1D"])
        assert result.exit_code == 0, result.output
        assert "Regime history for SPY (1D)" in result.output
        lines = [line for line in result.output.splitlines() if line.startswith("2026-")]
        assert [line[:10] for line in lines] == ["2026-01-03", "2026-01-02", "2026-01-01"]
        assert "Bull Trending" in lines[0]
        assert "103.00" in lines[0]

        result = runner.invoke(cli, ["history", "--symbol", "SPY", "--limit", "2", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert [d["bar_time"] for d in data] == ["2026-01-03T00:00:00", "2026-01-02T00:00:00"]
        assert data[0]["regime"] == "Bull Trending"
        assert data[0]["close"] == 103.0
        assert data[0]["recorded_at"].endswith("+00:00")

    def test_cli_history_validates_options(self, runner):
        assert runner.invoke(cli, ["history", "--limit", "0"]).exit_code == 2
        assert runner.invoke(cli, ["history", "--timeframe", "4H"]).exit_code == 2
        assert runner.invoke(cli, ["history", "--symbol", " "]).exit_code == 2

    def test_cli_history_storage_error_exits_nonzero(self, runner, tmp_path, monkeypatch):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        monkeypatch.setenv("MRA_DB_PATH", str(blocker / "regimes.db"))
        result = runner.invoke(cli, ["history"])
        assert result.exit_code == 1
        assert "Storage error" in result.output
