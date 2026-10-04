"""The ``mra`` and ``mra-optimize`` entry points load ``.env`` before reading config."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mra_cli import optimization
from mra_cli.main import cli

ENV_VARS = ("DEFAULT_PROVIDER", "MRA_ENV_FILE", "MRA_NO_DOTENV", "POLYGON_API_KEY")


@pytest.fixture(autouse=True)
def dotenv_project(tmp_path: Path, monkeypatch) -> Path:
    """Run in a temp project dir with loading enabled; restore os.environ afterwards."""
    saved = dict(os.environ)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))  # never walk into the real home dir
    monkeypatch.chdir(project)
    yield project
    os.environ.clear()
    os.environ.update(saved)


def _provider_used(args: list[str]) -> str:
    with patch("mra_cli.main._analyze_single_timeframe", side_effect=ValueError("x")) as f:
        CliRunner().invoke(cli, args)
    return f.call_args.args[2]


def test_cli_dotenv_default_provider_picked_up(dotenv_project):
    (dotenv_project / ".env").write_text("DEFAULT_PROVIDER=mock\n")
    assert _provider_used(["detailed-analysis"]) == "mock"


def test_cli_dotenv_bogus_provider_reaches_resolution(dotenv_project):
    (dotenv_project / ".env").write_text("DEFAULT_PROVIDER=bogus\n")
    result = CliRunner().invoke(cli, ["list-providers"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(cli, ["detailed-analysis"])
    assert result.exit_code == 1
    assert "Unknown provider 'bogus'" in result.output


def test_cli_dotenv_process_env_wins(dotenv_project, monkeypatch):
    (dotenv_project / ".env").write_text("DEFAULT_PROVIDER=bogus\n")
    monkeypatch.setenv("DEFAULT_PROVIDER", "mock")
    assert _provider_used(["detailed-analysis"]) == "mock"


def test_cli_dotenv_disabled(dotenv_project, monkeypatch):
    (dotenv_project / ".env").write_text("DEFAULT_PROVIDER=mock\n")
    monkeypatch.setenv("MRA_NO_DOTENV", "1")
    assert _provider_used(["detailed-analysis"]) == "yfinance"
    assert "DEFAULT_PROVIDER" not in os.environ


def test_cli_dotenv_api_key_picked_up(dotenv_project):
    (dotenv_project / ".env").write_text("POLYGON_API_KEY=pk-from-dotenv\n")
    with patch("mra_cli.main._analyze_single_timeframe", side_effect=ValueError("x")) as f:
        CliRunner().invoke(cli, ["detailed-analysis", "--provider", "polygon"])
    assert f.call_args.args[3] == "pk-from-dotenv"


def test_cli_explicit_env_file(dotenv_project, tmp_path, monkeypatch):
    custom = tmp_path / "custom.env"
    custom.write_text("DEFAULT_PROVIDER=mock\n")
    monkeypatch.setenv("MRA_ENV_FILE", str(custom))
    assert _provider_used(["detailed-analysis"]) == "mock"


def test_cli_missing_explicit_env_file_fails(dotenv_project, monkeypatch):
    monkeypatch.setenv("MRA_ENV_FILE", str(dotenv_project / "nope.env"))
    result = CliRunner().invoke(cli, ["list-providers"])
    assert result.exit_code == 1
    assert "MRA_ENV_FILE" in result.output


def test_optimize_main_loads_dotenv_first(dotenv_project, monkeypatch):
    (dotenv_project / ".env").write_text("POLYGON_API_KEY=pk-from-dotenv\n")
    seen: dict[str, str | None] = {}

    def fake_parse_args(self, *a, **k):
        seen["key"] = os.getenv("POLYGON_API_KEY")
        raise SystemExit(0)

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", fake_parse_args)
    with pytest.raises(SystemExit):
        optimization.main()
    assert seen["key"] == "pk-from-dotenv"


def test_optimize_main_missing_explicit_env_file(dotenv_project, monkeypatch):
    monkeypatch.setenv("MRA_ENV_FILE", "nope.env")
    with pytest.raises(SystemExit) as exc:
        optimization.main()
    assert "MRA_ENV_FILE" in str(exc.value.code)
