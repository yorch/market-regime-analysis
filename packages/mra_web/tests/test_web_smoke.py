"""Smoke tests for mra_web package."""

from mra_web.app import app


def test_app_exists():
    """Verify the FastAPI app can be imported."""
    assert app is not None
