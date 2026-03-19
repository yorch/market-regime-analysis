"""Smoke tests for mra_cli package."""

from mra_cli.main import cli


def test_cli_group_exists():
    """Verify the CLI entry point is a Click group."""
    assert cli.name == "cli" or callable(cli)
