"""Tests for mra_cli package — CLI commands and helpers."""

import pytest
from click.testing import CliRunner

from mra_cli.main import (
    cli,
    handle_exceptions,
    validate_api_key,
    validate_correlation,
    validate_percentage,
    validate_positive_int,
)


class TestValidateApiKey:
    def test_yfinance_no_key_needed(self):
        result = validate_api_key("yfinance", None)
        assert result == ""

    def test_yfinance_key_passthrough(self):
        result = validate_api_key("yfinance", "mykey")
        assert result == "mykey"

    def test_alphavantage_with_key(self):
        result = validate_api_key("alphavantage", "mykey")
        assert result == "mykey"

    def test_alphavantage_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        import click

        with pytest.raises(click.ClickException):
            validate_api_key("alphavantage", None)

    def test_alphavantage_env_var(self, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "env_key")
        result = validate_api_key("alphavantage", None)
        assert result == "env_key"

    def test_polygon_with_key(self):
        result = validate_api_key("polygon", "mykey")
        assert result == "mykey"

    def test_polygon_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        import click

        with pytest.raises(click.ClickException):
            validate_api_key("polygon", None)

    def test_polygon_env_var(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "poly_key")
        result = validate_api_key("polygon", None)
        assert result == "poly_key"


class TestValidatePercentage:
    def test_valid(self):
        assert validate_percentage(None, None, 0.5) == 0.5

    def test_zero(self):
        assert validate_percentage(None, None, 0.0) == 0.0

    def test_one(self):
        assert validate_percentage(None, None, 1.0) == 1.0

    def test_negative_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            validate_percentage(None, None, -0.1)

    def test_above_one_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            validate_percentage(None, None, 1.1)


class TestValidateCorrelation:
    def test_valid(self):
        assert validate_correlation(None, None, 0.5) == 0.5

    def test_negative(self):
        assert validate_correlation(None, None, -0.5) == -0.5

    def test_below_neg_one_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            validate_correlation(None, None, -1.1)

    def test_above_one_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            validate_correlation(None, None, 1.1)


class TestValidatePositiveInt:
    def test_valid(self):
        assert validate_positive_int(None, None, 5) == 5

    def test_zero_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            validate_positive_int(None, None, 0)

    def test_negative_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            validate_positive_int(None, None, -1)


class TestHandleExceptions:
    def test_normal_return(self):
        @handle_exceptions
        def good_func():
            return 42

        assert good_func() == 42

    def test_value_error_aborts(self):
        import click

        @handle_exceptions
        def bad_func():
            raise ValueError("bad input")

        with pytest.raises(click.Abort):
            bad_func()

    def test_connection_error_aborts(self):
        import click

        @handle_exceptions
        def net_func():
            raise ConnectionError("no internet")

        with pytest.raises(click.Abort):
            net_func()

    def test_file_not_found_aborts(self):
        import click

        @handle_exceptions
        def file_func():
            raise FileNotFoundError("missing")

        with pytest.raises(click.Abort):
            file_func()

    def test_permission_error_aborts(self):
        import click

        @handle_exceptions
        def perm_func():
            raise PermissionError("denied")

        with pytest.raises(click.Abort):
            perm_func()

    def test_generic_exception_aborts(self):
        import click

        @handle_exceptions
        def generic_func():
            raise RuntimeError("unexpected")

        with pytest.raises(click.Abort):
            generic_func()

    def test_click_exception_passthrough(self):
        import click

        @handle_exceptions
        def click_func():
            raise click.ClickException("click error")

        with pytest.raises(click.ClickException):
            click_func()


class TestCliGroup:
    def test_cli_exists(self):
        assert callable(cli)

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Market Regime Analysis" in result.output

    def test_list_providers(self):
        runner = CliRunner()
        # list-providers doesn't use the provider context, but the group
        # validates --api-key for the default alphavantage provider.
        # Use --provider yfinance to skip API key validation.
        result = runner.invoke(cli, ["--provider", "yfinance", "list-providers"])
        assert result.exit_code == 0
        assert "yfinance" in result.output.lower()

    def test_position_sizing(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--provider",
                "yfinance",
                "position-sizing",
                "--base-size",
                "0.02",
                "--regime",
                "Bull Trending",
                "--confidence",
                "0.8",
                "--persistence",
                "0.7",
                "--correlation",
                "0.0",
            ],
        )
        assert result.exit_code == 0
        assert "POSITION SIZING RESULTS" in result.output

    def test_position_sizing_invalid_base_size(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--provider",
                "yfinance",
                "position-sizing",
                "--base-size",
                "1.5",
            ],
        )
        assert result.exit_code != 0
