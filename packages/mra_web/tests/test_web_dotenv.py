"""``mra-api``, ``mra-token`` and ``mra_web.app`` load ``.env`` before reading config."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from jose import jwt

from mra_web import auth, server

DOTENV_SECRET = "dotenv-jwt-secret-abcdefghijklmnopqrstuvwxyz0123"
CLEARED = ("JWT_SECRET", "API_KEYS", "ENVIRONMENT", "API_PORT", "MRA_ENV_FILE", "MRA_NO_DOTENV")


@pytest.fixture
def dotenv_project(tmp_path: Path, monkeypatch) -> Path:
    """A temp project with a ``.env``; loading enabled and the conftest secrets cleared."""
    saved = dict(os.environ)
    for var in CLEARED:
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    (project / ".env").write_text(
        f"ENVIRONMENT=production\nJWT_SECRET={DOTENV_SECRET}\nAPI_PORT=9123\n"
    )
    yield project
    os.environ.clear()
    os.environ.update(saved)


def test_mra_api_main_sees_dotenv_config(dotenv_project, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mra-api"])
    with patch("mra_web.server.uvicorn.run") as run:
        server.main()  # would sys.exit(2) in production without JWT_SECRET
    run.assert_called_once()
    assert run.call_args.kwargs["port"] == 9123  # argparse default read after loading
    assert os.environ["JWT_SECRET"] == DOTENV_SECRET


def test_mra_api_process_env_wins(dotenv_project, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mra-api"])
    monkeypatch.setenv("API_PORT", "9555")
    with patch("mra_web.server.uvicorn.run") as run:
        server.main()
    assert run.call_args.kwargs["port"] == 9555


def test_mra_api_without_dotenv_refuses_to_start(dotenv_project, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mra-api"])
    monkeypatch.setenv("MRA_NO_DOTENV", "1")
    with patch("mra_web.server.uvicorn.run") as run, pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2
    run.assert_not_called()


def test_mra_api_missing_explicit_env_file(dotenv_project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mra-api"])
    monkeypatch.setenv("MRA_ENV_FILE", "nope.env")
    with patch("mra_web.server.uvicorn.run") as run, pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2
    assert "MRA_ENV_FILE" in capsys.readouterr().err
    run.assert_not_called()


def test_mra_token_signs_with_dotenv_secret(dotenv_project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mra-token", "--sub", "alice"])
    with pytest.raises(SystemExit) as exc:
        auth.main()
    assert exc.value.code == 0
    token = capsys.readouterr().out.strip()
    assert jwt.decode(token, DOTENV_SECRET, algorithms=["HS256"])["sub"] == "alice"


def test_app_module_loads_dotenv_at_import(dotenv_project):
    env = {k: v for k, v in os.environ.items() if k not in CLEARED}
    code = (
        "import mra_web.app as a\n"
        f"print(a.app.state.config.jwt_secret == {DOTENV_SECRET!r}, a.config.port)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=dotenv_project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["True", "9123"]
