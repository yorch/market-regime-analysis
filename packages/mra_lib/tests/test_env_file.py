"""Tests for the explicit ``.env`` loader used by the app entry points."""

import os
from pathlib import Path

import pytest

from mra_lib.config.env_file import (
    ENV_FILE_VAR,
    NO_DOTENV_VAR,
    EnvFileError,
    find_env_file,
    load_env_file,
)


@pytest.fixture(autouse=True)
def restore_environ(monkeypatch):
    """Snapshot os.environ (the loader writes to it directly) and enable loading."""
    saved = dict(os.environ)
    monkeypatch.delenv(NO_DOTENV_VAR, raising=False)
    monkeypatch.delenv(ENV_FILE_VAR, raising=False)
    for key in [k for k in os.environ if k.startswith("MRA_T_")]:
        monkeypatch.delenv(key)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture
def project(home: Path, monkeypatch) -> Path:
    path = home / "code" / "repo"
    path.mkdir(parents=True)
    monkeypatch.chdir(path)
    return path


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestDiscovery:
    def test_load_env_file_loads_from_cwd(self, project, home):
        env = write(project / ".env", "MRA_T_A=one\n")
        assert load_env_file(home=home) == env
        assert os.environ["MRA_T_A"] == "one"

    def test_load_env_file_no_file_is_noop(self, project, home):
        before = dict(os.environ)
        assert load_env_file(home=home) is None
        assert dict(os.environ) == before

    def test_find_env_file_walks_up_to_nearest(self, project, home):
        sub = project / "packages" / "pkg"
        sub.mkdir(parents=True)
        root_env = write(project / ".env", "MRA_T_A=root\n")
        write(home / "code" / ".env", "MRA_T_A=higher\n")
        assert find_env_file(sub, home=home) == root_env

    def test_load_env_file_from_subdirectory(self, project, home, monkeypatch):
        sub = project / "packages" / "pkg"
        sub.mkdir(parents=True)
        write(project / ".env", "MRA_T_A=root\n")
        monkeypatch.chdir(sub)
        load_env_file(home=home)
        assert os.environ["MRA_T_A"] == "root"

    def test_find_env_file_does_not_use_home_env_from_below(self, project, home):
        write(home / ".env", "MRA_T_A=home\n")
        assert find_env_file(project, home=home) is None

    def test_find_env_file_uses_home_env_when_cwd_is_home(self, home):
        env = write(home / ".env", "MRA_T_A=home\n")
        assert find_env_file(home, home=home) == env

    def test_find_env_file_outside_home_checks_only_cwd(self, tmp_path, home):
        outside = tmp_path / "srv" / "app"
        outside.mkdir(parents=True)
        write(tmp_path / "srv" / ".env", "MRA_T_A=parent\n")
        assert find_env_file(outside, home=home) is None
        env = write(outside / ".env", "MRA_T_A=here\n")
        assert find_env_file(outside, home=home) == env

    def test_find_env_file_stops_at_git_root(self, project, home):
        write(home / "code" / ".env", "MRA_T_A=other-project\n")
        (project / ".git").mkdir()
        sub = project / "src"
        sub.mkdir()
        assert find_env_file(sub, home=home) is None
        env = write(project / ".env", "MRA_T_A=mine\n")
        assert find_env_file(sub, home=home) == env

    def test_find_env_file_git_worktree_file_is_a_root(self, project, home):
        write(home / "code" / ".env", "MRA_T_A=main-checkout\n")
        write(project / ".git", "gitdir: /elsewhere\n")
        assert find_env_file(project, home=home) is None

    def test_load_env_file_deleted_cwd_is_noop(self, project, home, monkeypatch):
        def gone():
            raise FileNotFoundError("cwd removed")

        monkeypatch.setattr(Path, "cwd", staticmethod(gone))
        assert load_env_file(home=home) is None

    def test_find_env_file_ignores_env_directory(self, project, home):
        (project / ".env").mkdir()
        assert find_env_file(project, home=home) is None


class TestPrecedence:
    def test_load_env_file_does_not_override_existing(self, project, home, monkeypatch):
        monkeypatch.setenv("MRA_T_A", "from-process")
        write(project / ".env", "MRA_T_A=from-file\nMRA_T_B=two\n")
        load_env_file(home=home)
        assert os.environ["MRA_T_A"] == "from-process"
        assert os.environ["MRA_T_B"] == "two"

    def test_load_env_file_empty_process_value_still_wins(self, project, home, monkeypatch):
        monkeypatch.setenv("MRA_T_A", "")
        write(project / ".env", "MRA_T_A=from-file\n")
        load_env_file(home=home)
        assert os.environ["MRA_T_A"] == ""

    def test_load_env_file_interpolation_prefers_process_env(self, project, home, monkeypatch):
        monkeypatch.setenv("MRA_T_BASE", "proc")
        write(project / ".env", "MRA_T_BASE=file\nMRA_T_URL=${MRA_T_BASE}/x\n")
        load_env_file(home=home)
        assert os.environ["MRA_T_URL"] == "proc/x"


class TestOverrides:
    def test_no_dotenv_disables(self, project, home, monkeypatch):
        write(project / ".env", "MRA_T_A=one\n")
        monkeypatch.setenv(NO_DOTENV_VAR, "1")
        assert load_env_file(home=home) is None
        assert "MRA_T_A" not in os.environ

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_no_dotenv_falsy_values_keep_loading(self, project, home, monkeypatch, value):
        write(project / ".env", "MRA_T_A=one\n")
        monkeypatch.setenv(NO_DOTENV_VAR, value)
        assert load_env_file(home=home) is not None
        assert os.environ["MRA_T_A"] == "one"

    def test_no_dotenv_wins_over_explicit_file(self, project, home, monkeypatch):
        monkeypatch.setenv(ENV_FILE_VAR, str(project / "missing.env"))
        monkeypatch.setenv(NO_DOTENV_VAR, "true")
        assert load_env_file(home=home) is None

    def test_explicit_file_absolute(self, project, home, tmp_path, monkeypatch):
        write(project / ".env", "MRA_T_A=discovered\n")
        explicit = write(tmp_path / "custom.env", "MRA_T_A=explicit\n")
        monkeypatch.setenv(ENV_FILE_VAR, str(explicit))
        assert load_env_file(home=home) == explicit
        assert os.environ["MRA_T_A"] == "explicit"

    def test_explicit_file_relative_to_cwd(self, project, home, monkeypatch):
        explicit = write(project / "dev.env", "MRA_T_A=dev\n")
        monkeypatch.setenv(ENV_FILE_VAR, "dev.env")
        assert load_env_file(home=home) == explicit
        assert os.environ["MRA_T_A"] == "dev"

    def test_explicit_file_missing_is_error(self, project, home, monkeypatch):
        write(project / ".env", "MRA_T_A=discovered\n")
        monkeypatch.setenv(ENV_FILE_VAR, str(project / "missing.env"))
        with pytest.raises(EnvFileError, match=r"missing\.env"):
            load_env_file(home=home)
        assert "MRA_T_A" not in os.environ

    def test_explicit_file_directory_is_error(self, project, home, monkeypatch):
        monkeypatch.setenv(ENV_FILE_VAR, str(project))
        with pytest.raises(EnvFileError):
            load_env_file(home=home)


class TestParsing:
    def test_python_dotenv_syntax(self, project, home):
        write(
            project / ".env",
            "# comment\n"
            "\n"
            "export MRA_T_EXPORTED=yes\n"
            "MRA_T_SPACED = padded \n"
            "MRA_T_SINGLE='a # not comment'\n"
            'MRA_T_DOUBLE="line1\\nline2"\n'
            "MRA_T_INLINE=value # trailing comment\n"
            "MRA_T_EMPTY=\n"
            'MRA_T_MULTI="first\nsecond"\n'
            "MRA_T_BARE\n",
        )
        load_env_file(home=home)
        assert os.environ["MRA_T_EXPORTED"] == "yes"
        assert os.environ["MRA_T_SPACED"] == "padded"
        assert os.environ["MRA_T_SINGLE"] == "a # not comment"
        assert os.environ["MRA_T_DOUBLE"] == "line1\nline2"
        assert os.environ["MRA_T_INLINE"] == "value"
        assert os.environ["MRA_T_EMPTY"] == ""
        assert os.environ["MRA_T_MULTI"] == "first\nsecond"
        assert "MRA_T_BARE" not in os.environ


class TestLogging:
    def test_values_never_logged(self, project, home, caplog):
        write(project / ".env", "MRA_T_SECRET=s3cr3t-value\n")
        with caplog.at_level("DEBUG", logger="mra_lib.config.env_file"):
            load_env_file(home=home)
        assert "s3cr3t-value" not in caplog.text
        assert "Loaded 1 variable(s)" in caplog.text
        assert all(r.levelname == "DEBUG" for r in caplog.records)


def test_importing_mra_lib_does_not_load_env(tmp_path):
    """mra_lib is a library: importing it must never read .env."""
    import subprocess
    import sys

    write(tmp_path / ".env", "MRA_T_IMPORT=leaked\n")
    env = {k: v for k, v in os.environ.items() if k != NO_DOTENV_VAR}
    code = "import os, mra_lib, mra_lib.config.env_file; print(os.getenv('MRA_T_IMPORT'))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "None"
