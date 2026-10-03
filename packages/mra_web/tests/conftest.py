"""Shared fixtures for mra_web tests.

The web config is loaded from the environment at import time and refuses to start
without a strong JWT secret, so set a deterministic test environment *before* any
``mra_web`` module is imported.
"""

import os

TEST_JWT_SECRET = "test-jwt-secret-0123456789abcdefghijklmnop"
TEST_API_KEY = "test-api-key-0123456789abcdef"

os.environ["ENVIRONMENT"] = "production"
os.environ["JWT_SECRET"] = TEST_JWT_SECRET
os.environ["API_KEYS"] = TEST_API_KEY
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ.pop("CORS_ORIGINS", None)
os.environ.pop("ENABLE_DOCS", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mra_web.app import app, create_app  # noqa: E402
from mra_web.auth import create_user_token  # noqa: E402
from mra_web.config import APIConfig  # noqa: E402


def make_config(**overrides) -> APIConfig:
    """Build a production test config with optional overrides."""
    values = {
        "jwt_secret": TEST_JWT_SECRET,
        "api_keys": [TEST_API_KEY],
        "environment": "production",
        "rate_limit_per_minute": 100000,
    }
    values.update(overrides)
    return APIConfig(**values)


@pytest.fixture
def client() -> TestClient:
    """Unauthenticated client for the default (production) app."""
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Bearer headers with a valid JWT."""
    return {"Authorization": f"Bearer {create_user_token('tester')}"}


@pytest.fixture
def api_key_headers() -> dict[str, str]:
    """Headers with a valid static API key."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def app_factory():
    """Build an app from config overrides."""

    def _factory(**overrides):
        return create_app(make_config(**overrides))

    return _factory
