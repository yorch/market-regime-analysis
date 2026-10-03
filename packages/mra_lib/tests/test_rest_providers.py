"""Tests for the REST-based providers (Alpaca, Tiingo) and credential resolution."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from mra_lib.data_providers import (
    AlpacaProvider,
    InvalidSymbolError,
    MarketDataProvider,
    ProviderConfig,
    TiingoProvider,
    list_available_providers,
    required_env_vars,
    requires_credentials,
    resolve_api_key,
)
from mra_lib.data_providers._http import get_json
from mra_lib.data_providers.base import period_to_start

ALPACA_ENV = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_DATA_FEED")


def _response(status: int = 200, body: Any = None, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = body
    resp.text = str(body)
    resp.headers = headers or {}
    return resp


def _bar(t: str, price: float, volume: int = 1000) -> dict[str, Any]:
    return {"t": t, "o": price, "h": price + 1, "l": price - 1, "c": price + 0.5, "v": volume}


@pytest.fixture
def no_alpaca_env(monkeypatch):
    for var in ALPACA_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_sleep():
    with patch("mra_lib.data_providers._http.time.sleep") as sleep:
        yield sleep


class TestPeriodToStart:
    def test_period_to_start_days(self):
        end = datetime(2026, 6, 30, tzinfo=UTC)
        assert period_to_start("1y", end) == datetime(2025, 6, 30, tzinfo=UTC)

    def test_period_to_start_ytd(self):
        end = datetime(2026, 6, 30, 15, 30, tzinfo=UTC)
        assert period_to_start("ytd", end) == datetime(2026, 1, 1, tzinfo=UTC)

    def test_period_to_start_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported period"):
            period_to_start("7y")


class TestGetJson:
    def test_get_json_returns_body(self):
        with patch("requests.get", return_value=_response(body={"ok": True})):
            assert get_json("https://x", provider="P", config=ProviderConfig()) == {"ok": True}

    def test_get_json_retries_on_429(self, no_sleep):
        responses = [_response(429, headers={"Retry-After": "2"}), _response(body=[1])]
        with patch("requests.get", side_effect=responses) as get:
            assert get_json("https://x", provider="P", config=ProviderConfig(retries=2)) == [1]
        assert get.call_count == 2
        no_sleep.assert_called_once_with(2.0)

    def test_get_json_long_retry_after_fails_fast(self, no_sleep):
        with (
            patch("requests.get", return_value=_response(429, headers={"Retry-After": "3600"})),
            pytest.raises(ConnectionError, match="rate limiting"),
        ):
            get_json("https://x", provider="P", config=ProviderConfig(retries=2))
        no_sleep.assert_not_called()

    def test_get_json_auth_failure_raises(self):
        with (
            patch("requests.get", return_value=_response(403)),
            pytest.raises(ConnectionError, match="rejected the credentials"),
        ):
            get_json("https://x", provider="P", config=ProviderConfig())

    def test_get_json_exhausted_retries_raises(self, no_sleep):
        with (
            patch("requests.get", return_value=_response(503, body="down")),
            pytest.raises(ConnectionError, match="HTTP 503"),
        ):
            get_json("https://x", provider="P", config=ProviderConfig(retries=1))

    def test_get_json_network_error_raises(self, no_sleep):
        with (
            patch("requests.get", side_effect=requests.ConnectionError("boom")),
            pytest.raises(ConnectionError, match="request failed"),
        ):
            get_json("https://x", provider="P", config=ProviderConfig(retries=1))


class TestCredentials:
    def test_resolve_api_key_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("TIINGO_API_KEY", "env")
        assert resolve_api_key("tiingo", "explicit") == "explicit"

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TIINGO_API_KEY", "env")
        assert resolve_api_key("tiingo", None) == "env"

    def test_resolve_api_key_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)
        assert resolve_api_key("tiingo", None) is None

    def test_resolve_api_key_free_provider(self):
        assert resolve_api_key("yfinance", None) == ""
        assert not requires_credentials("yfinance")

    def test_resolve_api_key_alpaca_pair(self, monkeypatch):
        monkeypatch.setenv("APCA_API_KEY_ID", "kid")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "sec")
        assert resolve_api_key("alpaca", None) == "kid:sec"

    def test_resolve_api_key_alpaca_partial_env(self, no_alpaca_env, monkeypatch):
        monkeypatch.setenv("APCA_API_KEY_ID", "kid")
        assert resolve_api_key("alpaca", None) is None

    def test_required_env_vars(self):
        assert required_env_vars("alpaca") == ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
        assert required_env_vars("polygon") == ["POLYGON_API_KEY"]
        assert required_env_vars("yfinance") == []


class TestAlpacaProvider:
    def test_alpaca_registered(self):
        assert "alpaca" in list_available_providers()

    def test_alpaca_requires_credentials(self, no_alpaca_env):
        with pytest.raises(ValueError, match="APCA_API_KEY_ID"):
            AlpacaProvider()

    def test_alpaca_combined_key(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="kid:sec"))
        assert provider._headers == {"APCA-API-KEY-ID": "kid", "APCA-API-SECRET-KEY": "sec"}

    def test_alpaca_explicit_key_not_paired_with_env_secret(self, no_alpaca_env, monkeypatch):
        monkeypatch.setenv("APCA_API_SECRET_KEY", "server-secret")
        with pytest.raises(ValueError, match="requires an API key ID and secret"):
            AlpacaProvider(ProviderConfig(api_key="client-key-id"))

    def test_alpaca_explicit_secret(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="kid", api_secret="sec"))
        assert provider._headers["APCA-API-SECRET-KEY"] == "sec"

    def test_alpaca_env_credentials(self, no_alpaca_env, monkeypatch):
        monkeypatch.setenv("APCA_API_KEY_ID", "kid")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "sec")
        provider = MarketDataProvider.create_provider("alpaca")
        assert provider._headers["APCA-API-KEY-ID"] == "kid"
        assert provider.feed == "iex"

    def test_alpaca_feed_from_env(self, no_alpaca_env, monkeypatch):
        monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
        assert AlpacaProvider(ProviderConfig(api_key="k:s")).feed == "sip"

    def test_alpaca_invalid_feed(self, no_alpaca_env):
        with pytest.raises(ValueError, match="Unsupported Alpaca feed"):
            AlpacaProvider(ProviderConfig(api_key="k:s", feed="nasdaq"))

    def test_alpaca_fetch_paginates(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        pages = [
            _response(
                body={
                    "bars": [_bar("2026-01-02T05:00:00Z", 100), _bar("2026-01-05T05:00:00Z", 101)],
                    "next_page_token": "abc",
                }
            ),
            _response(body={"bars": [_bar("2026-01-06T05:00:00Z", 102)], "next_page_token": None}),
        ]
        with patch("requests.get", side_effect=pages) as get:
            df = provider.fetch("spy", "1y", "1D")

        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 3
        assert df.index.tz is None
        # Daily bars are labeled by session date (midnight), per the base contract
        assert df.index[0] == pd.Timestamp("2026-01-02")

        first_url = get.call_args_list[0].args[0]
        first_params = get.call_args_list[0].kwargs["params"]
        assert first_url.endswith("/SPY/bars")
        assert first_params["timeframe"] == "1Day"
        assert first_params["adjustment"] == "all"
        assert first_params["feed"] == "iex"
        assert get.call_args_list[1].kwargs["params"]["page_token"] == "abc"
        assert get.call_args_list[0].kwargs["headers"]["APCA-API-KEY-ID"] == "k"

    def test_alpaca_fetch_intraday_timeframes(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        body = {"bars": [_bar("2026-01-02T15:00:00Z", 100)], "next_page_token": None}
        for interval, expected in [("15m", "15Min"), ("1h", "1Hour"), ("1wk", "1Week")]:
            with patch("requests.get", return_value=_response(body=body)) as get:
                provider.fetch("SPY", "2mo", interval)
            assert get.call_args.kwargs["params"]["timeframe"] == expected

    def test_alpaca_intraday_filters_extended_hours(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        # 2026-01-05 is a Monday; EST is UTC-5
        times = ["13:00", "14:00", "14:30", "20:45", "21:00", "23:00"]
        body = {"bars": [_bar(f"2026-01-05T{t}:00Z", 100) for t in times]}
        with patch("requests.get", return_value=_response(body=body)):
            df = provider.fetch("SPY", "2mo", "15m")
        # 08:00 and 09:00 ET pre-market dropped; 09:30-15:45 kept; 16:00+ dropped
        assert [ts.strftime("%H:%M") for ts in df.index] == ["14:30", "20:45"]

    def test_alpaca_hourly_keeps_opening_bar(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        body = {"bars": [_bar(f"2026-01-05T{t}:00Z", 100) for t in ["13:00", "14:00", "21:00"]]}
        with patch("requests.get", return_value=_response(body=body)):
            df = provider.fetch("SPY", "6mo", "1h")
        # The 09:00 ET bar overlaps the 09:30 open; 08:00 and 16:00 ET do not
        assert [ts.strftime("%H:%M") for ts in df.index] == ["14:00"]

    def test_alpaca_extended_hours_opt_in(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s", extended_hours=True))
        body = {"bars": [_bar(f"2026-01-05T{t}:00Z", 100) for t in ["13:00", "23:00"]]}
        with patch("requests.get", return_value=_response(body=body)):
            assert len(provider.fetch("SPY", "2mo", "15m")) == 2

    def test_alpaca_symbol_is_escaped(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        body = {"bars": [_bar("2026-01-05T05:00:00Z", 100)]}
        with patch("requests.get", return_value=_response(body=body)) as get:
            provider.fetch("spy?feed=sip", "1y", "1d")
        assert get.call_args.args[0].endswith("/SPY%3FFEED%3DSIP/bars")

    def test_alpaca_sip_end_is_delayed(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s", feed="sip"))
        body = {"bars": [_bar("2026-01-02T15:00:00Z", 100)]}
        with patch("requests.get", return_value=_response(body=body)) as get:
            provider.fetch("SPY", "1mo", "1d")
        end = datetime.fromisoformat(get.call_args.kwargs["params"]["end"])
        assert (datetime.now(UTC) - end).total_seconds() >= 15 * 60

    def test_alpaca_fetch_empty_raises(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        with (
            patch("requests.get", return_value=_response(body={"bars": None})),
            pytest.raises(ValueError, match="No data returned"),
        ):
            provider.fetch("SPY", "1y", "1d")

    def test_alpaca_fetch_malformed_raises(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        with (
            patch("requests.get", return_value=_response(body={"bars": [{"t": "x"}]})),
            pytest.raises(ValueError, match="Malformed"),
        ):
            provider.fetch("SPY", "1y", "1d")

    def test_alpaca_unsupported_interval(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        with pytest.raises(ValueError, match="not supported"):
            provider.fetch("SPY", "1y", "4h")

    def test_alpaca_runaway_pagination_raises(self, no_alpaca_env):
        provider = AlpacaProvider(ProviderConfig(api_key="k:s"))
        provider._MAX_PAGES = 2
        body = {"bars": [_bar("2026-01-02T15:00:00Z", 100)], "next_page_token": "again"}
        with (
            patch("requests.get", return_value=_response(body=body)),
            pytest.raises(ConnectionError, match="pagination"),
        ):
            provider.fetch("SPY", "1y", "1d")


class TestTiingoProvider:
    @staticmethod
    def _daily_row(date: str, price: float) -> dict[str, Any]:
        return {
            "date": date,
            "open": price * 2,
            "adjOpen": price,
            "adjHigh": price + 1,
            "adjLow": price - 1,
            "adjClose": price + 0.5,
            "adjVolume": 5000,
        }

    def test_tiingo_registered(self):
        assert "tiingo" in list_available_providers()

    def test_tiingo_requires_key(self, monkeypatch):
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)
        with pytest.raises(ValueError, match="requires an API key"):
            TiingoProvider()

    def test_tiingo_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TIINGO_API_KEY", "tk")
        assert MarketDataProvider.create_provider("tiingo").config.api_key == "tk"

    def test_tiingo_fetch_daily_uses_adjusted(self):
        provider = TiingoProvider(ProviderConfig(api_key="tk"))
        rows = [
            self._daily_row("2026-01-02T00:00:00.000Z", 100),
            self._daily_row("2026-01-05T00:00:00.000Z", 101),
        ]
        with patch("requests.get", return_value=_response(body=rows)) as get:
            df = provider.fetch("SPY", "2y", "1d")

        assert df["Open"].iloc[0] == 100  # adjOpen, not raw open
        assert df["Volume"].iloc[0] == 5000
        assert df.index.tz is None
        assert get.call_args.args[0] == "https://api.tiingo.com/tiingo/daily/spy/prices"
        assert get.call_args.kwargs["params"]["resampleFreq"] == "daily"
        assert get.call_args.kwargs["headers"] == {"Authorization": "Token tk"}
        assert "token" not in get.call_args.kwargs["params"]

    def test_tiingo_fetch_weekly(self):
        provider = TiingoProvider(ProviderConfig(api_key="tk"))
        rows = [self._daily_row("2026-01-02T00:00:00.000Z", 100)]
        with patch("requests.get", return_value=_response(body=rows)) as get:
            provider.fetch("SPY", "5y", "1wk")
        assert get.call_args.kwargs["params"]["resampleFreq"] == "weekly"

    def test_tiingo_fetch_intraday(self):
        provider = TiingoProvider(ProviderConfig(api_key="tk"))
        rows = [
            {
                "date": "2026-01-02T14:30:00.000Z",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": None,
            }
        ]
        with patch("requests.get", return_value=_response(body=rows)) as get:
            df = provider.fetch("SPY", "2mo", "15m")

        assert get.call_args.args[0] == "https://api.tiingo.com/iex/spy/prices"
        params = get.call_args.kwargs["params"]
        assert params["resampleFreq"] == "15min"
        assert "volume" in params["columns"]
        assert df["Volume"].iloc[0] == 0

    def test_tiingo_fetch_empty_raises(self):
        provider = TiingoProvider(ProviderConfig(api_key="tk"))
        with (
            patch("requests.get", return_value=_response(body=[])),
            pytest.raises(ValueError, match="No data returned"),
        ):
            provider.fetch("SPY", "1y", "1d")

    def test_tiingo_fetch_error_object_raises(self):
        provider = TiingoProvider(ProviderConfig(api_key="tk"))
        with (
            patch("requests.get", return_value=_response(body={"detail": "Not found."})),
            pytest.raises(InvalidSymbolError, match="Not found"),
        ):
            provider.fetch("NOPE", "1y", "1d")

    def test_tiingo_symbol_is_escaped(self):
        provider = TiingoProvider(ProviderConfig(api_key="tk"))
        rows = [self._daily_row("2026-01-02T00:00:00.000Z", 100)]
        with patch("requests.get", return_value=_response(body=rows)) as get:
            provider.fetch("a/b?x=1", "1y", "1d")
        assert get.call_args.args[0].endswith("/a%2Fb%3Fx%3D1/prices")
