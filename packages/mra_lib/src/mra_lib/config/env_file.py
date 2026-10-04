"""Explicit ``.env`` loading for the application entry points.

``mra_lib`` is a library, so importing it never touches the environment. The apps
(``mra``, ``mra-optimize``, ``mra-api``, ``mra-token``, ``mra_web.app``) call
:func:`load_env_file` once, before they read any configuration.

Rules:

* Variables already set in the process environment always win; ``.env`` only
  fills in what is missing. Docker, compose and CI keep working unchanged.
* ``MRA_NO_DOTENV=1`` (or ``true``/``yes``/``on``) disables loading entirely.
* ``MRA_ENV_FILE=/path/to/file`` loads exactly that file (relative paths resolve
  against the current directory); a missing file is an error.
* Otherwise the first ``.env`` found walking up from the current directory is
  loaded. The walk stops below the user's home directory: ``~/.env`` itself is
  only used when the current directory *is* home, and a current directory outside
  home (e.g. ``/app`` in the Docker image) is the only place checked.

Parsing is delegated to ``python-dotenv`` (quotes, ``export`` prefixes, comments,
multiline values and ``${VAR}`` interpolation). Values are never logged; only the
file path and the number of variables set are, at DEBUG.
"""

import logging
import os
from pathlib import Path

from mra_lib.errors import MRAError

logger = logging.getLogger(__name__)

ENV_FILE_VAR = "MRA_ENV_FILE"
NO_DOTENV_VAR = "MRA_NO_DOTENV"
ENV_FILE_NAME = ".env"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class EnvFileError(MRAError):
    """Raised when an explicitly requested or discovered ``.env`` cannot be read."""


def dotenv_disabled() -> bool:
    """Return True when ``MRA_NO_DOTENV`` is set to a truthy value."""
    return os.getenv(NO_DOTENV_VAR, "").strip().lower() in _TRUTHY


def _home_dir() -> Path | None:
    try:
        return Path.home().resolve()
    except (RuntimeError, OSError):
        return None


def find_env_file(start: Path | None = None, *, home: Path | None = None) -> Path | None:
    """Find the ``.env`` file to load by walking up from ``start``.

    Args:
        start: Directory to start from (default: the current working directory).
        home: Directory the walk must not go above or into (default: the user's
            home directory). It is only checked when ``start`` is ``home`` itself.

    Returns:
        Path of the first ``.env`` file found, or ``None``.
    """
    current = (start if start is not None else Path.cwd()).resolve()
    boundary = home.resolve() if home is not None else _home_dir()

    candidates = [current]
    if boundary is not None and current != boundary and current.is_relative_to(boundary):
        for parent in current.parents:
            if parent == boundary:
                break
            candidates.append(parent)

    for directory in candidates:
        path = directory / ENV_FILE_NAME
        if path.is_file():
            return path
    return None


def load_env_file(start: Path | None = None, *, home: Path | None = None) -> Path | None:
    """Load a ``.env`` file into ``os.environ`` without overriding existing variables.

    Args:
        start: Directory to start discovery from (default: the current directory).
        home: Discovery boundary (default: the user's home directory).

    Returns:
        The file that was loaded, or ``None`` when loading is disabled or no file
        was found.

    Raises:
        EnvFileError: If ``MRA_ENV_FILE`` names a missing or non-regular file, or
            the selected file cannot be read.
    """
    if dotenv_disabled():
        logger.debug("%s is set; not loading a .env file", NO_DOTENV_VAR)
        return None

    explicit = os.getenv(ENV_FILE_VAR, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (start if start is not None else Path.cwd()) / path
        if not path.is_file():
            raise EnvFileError(f"{ENV_FILE_VAR} points to {path}, which is not a readable file")
    else:
        found = find_env_file(start, home=home)
        if found is None:
            return None
        path = found

    from dotenv import load_dotenv  # noqa: PLC0415 - only the apps need the parser

    before = set(os.environ)
    try:
        load_dotenv(path, override=False)
    except (OSError, UnicodeDecodeError) as e:
        raise EnvFileError(f"Could not read env file {path}: {e.__class__.__name__}") from e
    added = len(set(os.environ) - before)
    logger.debug("Loaded %d variable(s) from %s", added, path)
    return path
