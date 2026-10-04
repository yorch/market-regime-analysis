"""Repo-wide pytest setup, loaded before every package's tests and conftest.

The app entry points load a ``.env`` file (see ``mra_lib.config.env_file``), and
``mra_web.app`` does so at import time. Disable that for the test session so a
developer's real ``.env`` never leaks into the suite; the ``.env`` tests opt back
in with ``monkeypatch.delenv("MRA_NO_DOTENV")``.
"""

import os

os.environ["MRA_NO_DOTENV"] = "1"
os.environ.pop("MRA_ENV_FILE", None)
