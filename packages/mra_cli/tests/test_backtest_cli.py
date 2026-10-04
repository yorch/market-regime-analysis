"""End-to-end tests for ``mra backtest`` with the offline ``mock`` provider."""

import argparse
import csv
import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mra_cli import optimization
from mra_cli.main import cli

# 1y of mock daily data = 252 bars; small windows and 2 states keep this fast
FAST_WF = [
    "--provider",
    "mock",
    "--period",
    "1y",
    "--train-bars",
    "120",
    "--test-bars",
    "40",
    "--n-states",
    "2",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DEFAULT_PROVIDER", raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, *args: str):
    return runner.invoke(cli, ["backtest", "--symbol", "SPY", *args])


def _optimizer_file(tmp_path, best_params: dict, search_end: str | None = None):
    """Write a params file through mra-optimize's own ``build_output``."""
    summary = {"n_trials": 3, "search_bars": 200}
    if search_end is not None:
        summary["search_end"] = search_end
    best = SimpleNamespace(
        params=best_params,
        score=1.0,
        sharpe=1.2,
        total_return=0.1,
        excess_return=0.02,
        max_drawdown=-0.05,
        profit_factor=float("inf"),
        total_trades=12,
    )
    optimizer = SimpleNamespace(results=[best], search_summary=lambda: summary)
    args = argparse.Namespace(symbol="SPY", provider="mock", mode="random", seed=1)
    output = optimization.build_output(args, optimizer, None)  # type: ignore[arg-type]
    path = tmp_path / "optimization_results.json"
    path.write_text(json.dumps(output, indent=2, default=str))
    return path


class TestBacktestModes:
    def test_backtest_walk_forward_text(self, runner, tmp_path):
        result = _invoke(runner, *FAST_WF)
        assert result.exit_code == 0, result.output
        out = result.stdout
        assert "walk-forward | OUT-OF-SAMPLE" in out
        assert "BUY & HOLD" in out
        for label in (
            "Total Return",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max Drawdown",
            "Max DD Duration",
            "Win Rate",
            "Profit Factor",
            "Trades",
            "Time in Market",
        ):
            assert label in out
        assert "PER-WINDOW SUMMARY" in out
        assert "refit failures: 0" in out
        assert "IN-SAMPLE" not in out

    def test_backtest_simple_labelled_in_sample(self, runner):
        result = _invoke(runner, "--provider", "mock", "--period", "1y", "--mode", "simple")
        assert result.exit_code == 0, result.output
        assert "simple | IN-SAMPLE" in result.stdout
        assert "NOT an out-of-sample estimate" in result.stdout
        assert "PER-WINDOW SUMMARY" not in result.stdout

    def test_backtest_group_level_provider(self, runner):
        result = runner.invoke(
            cli,
            ["--provider", "mock", "backtest", "--period", "1y", "--mode", "simple", "--json"],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["mode"] == "simple"


class TestBacktestOutput:
    def test_backtest_json_shape(self, runner):
        result = _invoke(runner, *FAST_WF, "--json")
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)  # stdout holds only the JSON object
        assert data["symbol"] == "SPY"
        assert data["mode"] == "walk-forward"
        assert data["sample"] == "out-of-sample" and data["in_sample"] is False
        assert data["params_out_of_sample"] is True
        assert "asset_return" in data
        assert data["params_source"] == "defaults"
        assert data["trades_file"] is None
        expected = {
            "total_return",
            "cagr",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "max_drawdown_duration",
            "win_rate",
            "profit_factor",
            "n_trades",
            "time_in_market",
            "avg_exposure",
        }
        assert set(data["strategy"]) == expected
        assert set(data["buy_hold"]) == expected
        assert data["settings"]["train_bars"] == 120
        assert data["windows"] and {"test_start", "strategy_return", "buy_hold_return"} <= set(
            data["windows"][0]
        )
        assert data["period"]["bars"] == sum(w["bars"] for w in data["windows"])

    def test_backtest_trades_csv(self, runner, tmp_path):
        out = tmp_path / "trades.csv"
        result = _invoke(runner, *FAST_WF, "--json", "--output", str(out))
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["trades_file"] == str(out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == data["strategy"]["n_trades"]
        assert rows, "mock data should produce trades"
        assert {"window", "entry_date", "exit_date", "direction", "pnl"} <= set(rows[0])

    def test_backtest_trades_csv_text_mode(self, runner, tmp_path):
        out = tmp_path / "trades.csv"
        result = _invoke(
            runner, "--provider", "mock", "--period", "1y", "--mode", "simple", "-o", str(out)
        )
        assert result.exit_code == 0, result.output
        assert f"trades written to {out}" in result.stdout
        header = out.read_text().splitlines()[0].split(",")
        assert "window" not in header and header[0] == "entry_date"


class TestBacktestParams:
    def test_backtest_params_from_optimizer_output(self, runner, tmp_path):
        best = {"bull_mult": 1.5, "bear_mult": 0.0, "bear_short": 0, "base_fraction": 0.08}
        path = _optimizer_file(tmp_path, best)
        result = _invoke(runner, *FAST_WF, "--params", str(path), "--json")
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["params"] == best
        assert data["params_source"] == "mra-optimize output"

    def test_backtest_params_overlap_warning(self, runner, tmp_path):
        path = _optimizer_file(tmp_path, {"bull_mult": 1.5}, search_end="2100-01-01 00:00:00")
        result = _invoke(runner, *FAST_WF, "--params", str(path))
        assert result.exit_code == 0, result.output
        assert "NOT for the parameters" in result.stdout
        assert "OUT-OF-SAMPLE (regime model only)" in result.stdout

    def test_backtest_params_flat_file(self, runner, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"stop_loss": 0.03, "take_profit": None}))
        result = _invoke(
            runner,
            "--provider",
            "mock",
            "--period",
            "1y",
            "--mode",
            "simple",
            "--params",
            str(path),
        )
        assert result.exit_code == 0, result.output
        assert "stop_loss=0.03" in result.stdout

    def test_backtest_params_unknown_key(self, runner, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"best_params": {"stoploss": 0.05}}))
        result = _invoke(runner, *FAST_WF, "--params", str(path))
        assert result.exit_code == 1
        assert "stoploss" in result.output
        assert "Valid" in result.output or "valid" in result.output

    def test_backtest_params_bad_type(self, runner, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"bull_mult": "high"}))
        result = _invoke(runner, *FAST_WF, "--params", str(path))
        assert result.exit_code == 1
        assert "bull_mult must be a number" in result.output

    def test_backtest_params_not_json(self, runner, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("bull_mult: 1.5")
        result = _invoke(runner, *FAST_WF, "--params", str(path))
        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_backtest_params_missing_file(self, runner, tmp_path):
        result = _invoke(runner, *FAST_WF, "--params", str(tmp_path / "nope.json"))
        assert result.exit_code == 2


class TestBacktestErrors:
    def test_backtest_insufficient_data_walk_forward(self, runner):
        # 6mo of mock data (~124 bars) < default 252 + 63 bar windows
        result = _invoke(runner, "--provider", "mock", "--period", "6mo")
        assert result.exit_code == 1
        assert "Insufficient data for SPY" in result.output
        assert "--period" in result.output
        assert "Traceback" not in result.output

    def test_backtest_insufficient_data_simple(self, runner):
        result = _invoke(runner, "--provider", "mock", "--period", "1mo", "--mode", "simple")
        assert result.exit_code == 1
        assert "Insufficient data for SPY" in result.output

    def test_backtest_train_bars_too_small(self, runner):
        result = _invoke(runner, "--provider", "mock", "--period", "1y", "--train-bars", "30")
        assert result.exit_code == 1
        assert "too small" in result.output

    def test_backtest_missing_key_fails(self, runner, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        result = _invoke(runner, "--provider", "polygon")
        assert result.exit_code == 1
        assert "POLYGON_API_KEY" in result.output
