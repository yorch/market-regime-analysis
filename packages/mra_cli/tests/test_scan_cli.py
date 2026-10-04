"""CLI tests for ``mra scan`` (offline mock provider; tmp database; no network)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mra_cli.main import cli
from mra_lib.data_providers.mock_provider import MockDataProvider
from mra_lib.storage import SQLiteRegimeStore

ALERT_VARS = ("ALERT_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DISCORD_WEBHOOK_URL")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    for var in (*ALERT_VARS, "DEFAULT_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MRA_DB_PATH", str(tmp_path / "regimes.db"))
    return tmp_path / "regimes.db"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_scan_once_dry_run_records_and_summarizes(runner, isolated_env):
    args = ["scan", "--provider", "mock", "--once", "--dry-run", "--symbols", "SPY,QQQ"]
    args += ["--timeframes", "1D", "--no-confirmation"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Alerts: log (dry run)" in result.output
    assert "Scan #1" in result.output
    assert "2 symbol(s) ok, 0 failed | 2 record(s) saved | 0 change(s)" in result.output
    assert "SPY: 1D" in result.output and "(baseline)" in result.output
    assert "Scanner finished: 1 scan(s) (1 ok, 0 failed" in result.output

    store = SQLiteRegimeStore(isolated_env)
    assert store.symbols() == ["QQQ", "SPY"]

    # Same bars again: recorded, but never a change
    again = runner.invoke(cli, args)
    assert again.exit_code == 0, again.output
    assert "(same bar)" in again.output and "0 change(s)" in again.output


def test_dry_run_ignores_alert_env(runner, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://insecure.example.com/hook")
    result = runner.invoke(
        cli,
        [
            "scan",
            "--provider",
            "mock",
            "--once",
            "--dry-run",
            "--timeframes",
            "1D",
            "--no-confirmation",
        ],
    )
    assert result.exit_code == 0, result.output


def test_invalid_alert_env_fails_without_echoing_secret(runner, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://insecure.example.com/s3cret")
    result = runner.invoke(cli, ["scan", "--provider", "mock", "--once"])
    assert result.exit_code != 0
    assert "ALERT_WEBHOOK_URL" in result.output
    assert "s3cret" not in result.output


def test_env_channels_listed_safely(runner, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    result = runner.invoke(
        cli, ["scan", "--provider", "mock", "--once", "--timeframes", "1D", "--no-confirmation"]
    )
    assert result.exit_code == 0, result.output
    assert "Alerts: log, telegram (chat 42)" in result.output
    assert "AAHdq" not in result.output


def test_once_exits_nonzero_when_every_symbol_fails(runner):
    def fail(self, symbol, period, interval):
        raise ConnectionError("network down")

    with patch.object(MockDataProvider, "fetch", fail):
        result = runner.invoke(
            cli, ["scan", "--provider", "mock", "--once", "--dry-run", "--symbols", "SPY,QQQ"]
        )
    assert result.exit_code != 0
    assert "0 symbol(s) ok, 2 failed" in result.output
    assert "Every symbol failed" in result.output


def test_partial_failure_still_succeeds(runner):
    real_fetch = MockDataProvider.fetch

    def fetch(self, symbol, period, interval):
        if symbol == "BAD":
            raise ConnectionError("network down")
        return real_fetch(self, symbol, period, interval)

    with patch.object(MockDataProvider, "fetch", fetch):
        result = runner.invoke(
            cli,
            [
                "scan",
                "--provider",
                "mock",
                "--once",
                "--dry-run",
                "--symbols",
                "SPY,BAD",
                "--timeframes",
                "1D",
                "--no-confirmation",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "1 symbol(s) ok, 1 failed" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--timeframes", "4H"], "unknown timeframe"),
        (["--symbols", " , "], "at least one value"),
        (["--timeframes", "1H,15m"], "not in --timeframes"),
        (["--min-confidence", "2"], "between 0.0 and 1.0"),
    ],
)
def test_bad_options(runner, args, message):
    result = runner.invoke(cli, ["scan", "--provider", "mock", "--once", *args])
    assert result.exit_code == 2
    assert message in result.output


def test_max_iterations_uses_interval(runner):
    with patch("mra_lib.scheduling.interruptible_sleep") as sleep:
        result = runner.invoke(
            cli,
            [
                "scan",
                "--provider",
                "mock",
                "--dry-run",
                "--timeframes",
                "1D",
                "--no-confirmation",
                "--max-iterations",
                "2",
                "--interval",
                "5",
            ],
        )
    assert result.exit_code == 0, result.output
    assert sleep.call_count == 1
    assert "Scan #2" in result.output
    assert "2 scan(s) (2 ok, 0 failed" in result.output
