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

    def test_cli_start_api_defaults_to_localhost(self, runner):
        with patch("uvicorn.run") as run:
            result = runner.invoke(cli, ["start-api"])
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["host"] == "127.0.0.1"


class TestReviewFixes:
    def test_cli_labels_wrapped_provider_errors_by_cause(self, runner):
        from mra_lib.data_providers import AuthError

        def wrapped(*_args):
            try:
                raise AuthError("bad key")
            except AuthError as e:
                raise ValueError("Data loading failed for 1D: bad key") from e

        with patch("mra_cli.main._analyze_single_timeframe", side_effect=wrapped):
            result = runner.invoke(cli, ["detailed-analysis", "--provider", "mock"])
        assert result.exit_code == 1
        assert "Authentication error" in result.output

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
