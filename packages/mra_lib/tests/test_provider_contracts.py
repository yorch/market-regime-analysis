"""Tests for the provider contract: errors, timeouts, tz handling, and per-provider fixes.

All HTTP and client libraries are mocked; nothing touches the network.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mra_lib.data_providers import (
    AlphaVantageProvider,
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    MockDataProvider,
    PolygonProvider,
    ProviderConfig,
    RateLimitError,
    YFinanceProvider,
)
from mra_lib.data_providers._http import get_json, redact
from mra_lib.data_providers.base import TokenBucket, drop_in_progress_bar


def _response(status: int = 200, body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = body
    resp.text = str(body)
    resp.headers = {}
    return resp


def _ohlcv(index: Any) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "Open": np.arange(n, dtype=int) + 1,
            "High": np.arange(n, dtype=int) + 2,
            "Low": np.arange(n, dtype=int),
            "Close": np.arange(n, dtype=int) + 1,
            "Volume": np.arange(n, dtype=int) * 100,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Error hierarchy / HTTP helper
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_errors_keep_backward_compatible_bases(self):
        assert issubclass(InvalidSymbolError, ValueError)
        assert issubclass(AuthError, ConnectionError)
        assert issubclass(RateLimitError, ConnectionError)

    def test_get_json_classifies_status_codes(self):
        cases = {401: AuthError, 403: AuthError, 404: InvalidSymbolError, 429: RateLimitError}
        for status, exc in cases.items():
            with (
                patch("requests.get", return_value=_response(status, body="x")),
                pytest.raises(exc),
            ):
                get_json("https://x", provider="P", config=ProviderConfig(retries=0))

    def test_get_json_passes_timeout_and_throttles(self):
        throttle = MagicMock()
        with patch("requests.get", return_value=_response(body={})) as get:
            get_json("https://x", provider="P", config=ProviderConfig(timeout=7), throttle=throttle)
        assert get.call_args.kwargs["timeout"] == 7
        throttle.assert_called_once()

    def test_get_json_redacts_api_key_in_network_errors(self):
        import requests

        err = requests.ConnectionError("failed for https://x/query?symbol=SPY&apikey=SECRET123")
        with (
            patch("requests.get", side_effect=err),
            pytest.raises(ConnectionError) as info,
        ):
            get_json("https://x", provider="P", config=ProviderConfig(retries=0))
        assert "SECRET123" not in str(info.value)
        assert info.value.__cause__ is None

    def test_redact_masks_credentials(self):
        assert redact("a?apikey=abc&token=def&x=1") == "a?apikey=***&token=***&x=1"


# ---------------------------------------------------------------------------
# standardize_dataframe / in-progress bar / rate limiter
# ---------------------------------------------------------------------------


class TestStandardizeContract:
    def setup_method(self):
        self.provider = MockDataProvider()

    def test_standardize_does_not_mutate_input(self):
        raw = _ohlcv(["2024-01-02", "2024-01-01"])
        original_index = raw.index.copy()
        self.provider.standardize_dataframe(raw)
        pd.testing.assert_index_equal(raw.index, original_index)

    def test_standardize_dedupes_and_casts_to_float(self):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"])
        out = self.provider.standardize_dataframe(_ohlcv(idx))
        assert len(out) == 2
        assert out.loc["2024-01-02", "Close"] == 3.0  # keep="last"
        assert (out.dtypes == "float64").all()

    def test_standardize_intraday_tz_aware_to_naive_utc(self):
        idx = pd.DatetimeIndex(["2024-01-02 09:30"], tz="America/New_York")
        out = self.provider.standardize_dataframe(_ohlcv(idx), "15m")
        assert out.index.tz is None
        assert out.index[0] == pd.Timestamp("2024-01-02 14:30")

    def test_standardize_daily_keeps_session_date(self):
        idx = pd.DatetimeIndex(["2024-01-02 00:00"], tz="America/New_York")
        out = self.provider.standardize_dataframe(_ohlcv(idx), "1d")
        assert out.index[0] == pd.Timestamp("2024-01-02")

    def test_standardize_daily_utc_timestamps_normalized(self):
        idx = pd.to_datetime(["2024-01-02 05:00"])  # Alpaca/Polygon style
        out = self.provider.standardize_dataframe(_ohlcv(idx), "1d")
        assert out.index[0] == pd.Timestamp("2024-01-02")

    def test_drop_in_progress_intraday(self):
        now = datetime(2024, 1, 2, 15, 10, tzinfo=UTC)
        df = _ohlcv(pd.to_datetime(["2024-01-02 14:45", "2024-01-02 15:00"]))
        assert len(drop_in_progress_bar(df, "15m", now)) == 1
        later = now + timedelta(minutes=10)
        assert len(drop_in_progress_bar(df, "15m", later)) == 2

    def test_drop_in_progress_daily(self):
        df = _ohlcv(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        during_session = datetime(2024, 1, 2, 18, 0, tzinfo=UTC)  # 13:00 New York
        after_close = datetime(2024, 1, 2, 22, 0, tzinfo=UTC)  # 17:00 New York
        assert len(drop_in_progress_bar(df, "1d", during_session)) == 1
        assert len(drop_in_progress_bar(df, "1d", after_close)) == 2
        assert len(drop_in_progress_bar(df, "1wk", during_session)) == 2

    def test_drop_incomplete_bar_is_opt_in(self):
        provider = MockDataProvider(ProviderConfig(drop_incomplete_bar=True))
        default = MockDataProvider()
        assert (
            len(provider.fetch("SPY", "1mo", "15m")) == len(default.fetch("SPY", "1mo", "15m")) - 1
        )

    def test_token_bucket_waits_after_burst(self):
        bucket = TokenBucket(rate_per_minute=60, capacity=2)
        with patch("mra_lib.data_providers.base.time.sleep") as sleep:
            assert bucket.acquire() == 0
            assert bucket.acquire() == 0
            waited = bucket.acquire()
        assert 0.9 < waited <= 1.0
        sleep.assert_called_once()

    def test_rate_limit_config_overrides_and_disables(self):
        assert YFinanceProvider()._limiter is not None
        assert YFinanceProvider(ProviderConfig(rate_limit=-1))._limiter is None
        assert MockDataProvider()._limiter is None


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_mock_registered(self):
        assert "mock" in MarketDataProvider.get_available_providers()

    def test_mock_distinct_per_symbol_and_reproducible(self):
        provider = MockDataProvider()
        spy1 = provider.fetch("SPY", "1y", "1d")["Close"].to_numpy()
        spy2 = provider.fetch("SPY", "1y", "1d")["Close"].to_numpy()
        qqq = provider.fetch("QQQ", "1y", "1d")["Close"].to_numpy()
        np.testing.assert_array_equal(spy1, spy2)
        assert not np.allclose(spy1, qqq)

    def test_mock_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        MockDataProvider().fetch("SPY", "1y", "1d")
        assert np.random.random() == expected

    def test_mock_supports_default_periods(self):
        provider = MockDataProvider()
        for period, interval in (("2y", "1d"), ("6mo", "1h"), ("1mo", "15m")):
            df = provider.fetch("SPY", period, interval)
            assert len(df) > 100
            assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
            assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()


# ---------------------------------------------------------------------------
# Default periods work with every provider
# ---------------------------------------------------------------------------


def test_default_periods_supported_by_every_provider():
    from mra_lib.config.timeframes import DEFAULT_PERIODS, TIMEFRAME_INTERVALS

    for name, info in MarketDataProvider.get_available_providers().items():
        for tf, period in DEFAULT_PERIODS.items():
            assert period in info["supported_periods"], (name, tf, period)
            assert TIMEFRAME_INTERVALS[tf] in info["supported_intervals"], (name, tf)


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------


class TestYFinance:
    def test_yfinance_rejects_intraday_beyond_limit(self):
        with pytest.raises(ValueError, match="last 60 days"):
            YFinanceProvider().validate_parameters("SPY", "3mo", "15m")
        YFinanceProvider().validate_parameters("SPY", "1mo", "15m")

    def test_yfinance_passes_timeout_and_standardizes(self):
        idx = pd.DatetimeIndex(["2024-01-02 09:30", "2024-01-02 09:45"], tz="America/New_York")
        ticker = MagicMock()
        ticker.history.return_value = _ohlcv(idx)
        with patch("yfinance.Ticker", return_value=ticker):
            df = YFinanceProvider(ProviderConfig(timeout=5)).fetch("SPY", "1mo", "15m")
        assert ticker.history.call_args.kwargs["timeout"] == 5
        assert df.index.tz is None
        assert df.index[0] == pd.Timestamp("2024-01-02 14:30")

    def test_yfinance_empty_is_invalid_symbol(self):
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        with (
            patch("yfinance.Ticker", return_value=ticker),
            pytest.raises(InvalidSymbolError),
        ):
            YFinanceProvider().fetch("NOPE", "1mo", "1d")

    def test_yfinance_rate_limit_classified(self):
        from yfinance.exceptions import YFRateLimitError

        ticker = MagicMock()
        ticker.history.side_effect = YFRateLimitError()
        with (
            patch("yfinance.Ticker", return_value=ticker),
            pytest.raises(RateLimitError),
        ):
            YFinanceProvider().fetch("SPY", "1mo", "1d")

    def test_yfinance_network_error_is_connection_error(self):
        ticker = MagicMock()
        ticker.history.side_effect = OSError("socket closed")
        with (
            patch("yfinance.Ticker", return_value=ticker),
            pytest.raises(ConnectionError),
        ):
            YFinanceProvider().fetch("SPY", "1mo", "1d")


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------


def _av_daily(dates: list[str], adjusted: bool = False) -> dict[str, Any]:
    series = {}
    for i, d in enumerate(dates):
        row = {"1. open": "100", "2. high": "110", "3. low": "90", "4. close": "100"}
        if adjusted:
            row.update({"5. adjusted close": "50", "6. volume": str(1000 + i)})
        else:
            row["5. volume"] = str(1000 + i)
        series[d] = row
    return {"Meta Data": {}, "Time Series (Daily)": series}


@pytest.fixture
def av() -> AlphaVantageProvider:
    return AlphaVantageProvider(ProviderConfig(api_key="KEY", retries=0, premium=False))


class TestAlphaVantage:
    def test_av_intervals_consistent(self):
        for interval in ("30m", "1w", "1wk", "1h", "15m"):
            assert interval in AlphaVantageProvider.supported_intervals
            assert interval in AlphaVantageProvider._INTERVAL_MAP

    def test_av_daily_trims_to_period(self, av):
        today = datetime.now(UTC).date()
        dates = [(today - timedelta(days=d)).isoformat() for d in (1, 10, 100, 400)]
        with patch("requests.get", return_value=_response(body=_av_daily(dates))) as get:
            df = av.fetch("spy", "3mo", "1d")
        assert len(df) == 2  # 100 and 400 days ago are outside 3mo
        params = get.call_args.kwargs["params"]
        assert params["function"] == "TIME_SERIES_DAILY"
        assert params["symbol"] == "SPY"
        assert params["outputsize"] == "compact"
        assert get.call_args.kwargs["timeout"] == av.config.timeout

    def test_av_premium_uses_adjusted_prices(self):
        provider = AlphaVantageProvider(ProviderConfig(api_key="KEY", premium=True))
        today = datetime.now(UTC).date().isoformat()
        body = _av_daily([today], adjusted=True)
        with patch("requests.get", return_value=_response(body=body)) as get:
            df = provider.fetch("SPY", "1y", "1d")
        assert get.call_args.kwargs["params"]["function"] == "TIME_SERIES_DAILY_ADJUSTED"
        # Adjustment factor 0.5 applied to the whole bar
        assert df["Close"].iloc[-1] == 50
        assert df["High"].iloc[-1] == 55

    def test_av_full_history_premium_falls_back_to_compact(self):
        # e.g. a key whose plan lacks full history; free keys request compact directly
        av = AlphaVantageProvider(ProviderConfig(api_key="KEY", retries=0, premium=True))
        today = datetime.now(UTC).date().isoformat()
        responses = [
            _response(body={"Information": "outputsize=full is a premium feature"}),
            _response(body=_av_daily([today], adjusted=True)),
        ]
        with patch("requests.get", side_effect=responses) as get:
            df = av.fetch("SPY", "2y", "1d")
        assert len(df) == 1
        assert get.call_args_list[1].kwargs["params"]["outputsize"] == "compact"

    def test_av_intraday_converted_to_utc(self, av):
        now_et = pd.Timestamp.now(tz="America/New_York").floor("15min") - pd.Timedelta(hours=1)
        body = {
            "Meta Data": {"6. Time Zone": "US/Eastern"},
            "Time Series (15min)": {
                now_et.strftime("%Y-%m-%d %H:%M:%S"): {
                    "1. open": "1",
                    "2. high": "2",
                    "3. low": "0.5",
                    "4. close": "1.5",
                    "5. volume": "10",
                }
            },
        }
        with patch("requests.get", return_value=_response(body=body)) as get:
            df = av.fetch("SPY", "1mo", "15m")
        assert df.index[0] == now_et.tz_convert("UTC").tz_localize(None)
        params = get.call_args.kwargs["params"]
        assert params["function"] == "TIME_SERIES_INTRADAY"
        assert params["adjusted"] == "true"
        assert params["extended_hours"] == "false"

    @pytest.mark.parametrize(
        ("body", "exc"),
        [
            ({"Error Message": "Invalid API call. Please retry"}, InvalidSymbolError),
            ({"Error Message": "the parameter apikey is invalid or missing"}, AuthError),
            (
                {"Information": "Our standard API rate limit is 25 requests per day. premium"},
                RateLimitError,
            ),
            ({"Note": "Thank you! Our standard API call frequency is 5 calls"}, RateLimitError),
        ],
    )
    def test_av_error_classification(self, av, body, exc):
        with patch("requests.get", return_value=_response(body=body)), pytest.raises(exc):
            av.fetch("SPY", "1mo", "1d")


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------


def _agg(ts: datetime, price: float, **overrides: Any) -> SimpleNamespace:
    fields = {
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 100,
        "timestamp": int(ts.timestamp() * 1000),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def polygon() -> PolygonProvider:
    provider = PolygonProvider(ProviderConfig(api_key="KEY", timeout=9, retries=1))
    provider.client = MagicMock()
    return provider


class TestPolygon:
    def test_polygon_client_gets_timeout_and_retries(self):
        with patch("polygon.RESTClient") as client:
            PolygonProvider(ProviderConfig(api_key="KEY", timeout=9, retries=1))
        kwargs = client.call_args.kwargs
        assert kwargs["connect_timeout"] == 9
        assert kwargs["read_timeout"] == 9
        assert kwargs["retries"] == 1

    def test_polygon_paginates_and_drops_bad_bars(self, polygon):
        day = datetime(2024, 1, 2, 5, tzinfo=UTC)
        polygon.client.list_aggs.return_value = iter(
            [
                _agg(day, 100),
                _agg(day + timedelta(days=1), 101, high=50),  # High < Low: dropped
                _agg(day + timedelta(days=2), 102, close=None),  # Malformed: dropped
                _agg(day + timedelta(days=3), 103),
            ]
        )
        df = polygon.fetch("spy", "1y", "1d")
        assert len(df) == 2
        assert df.index[0] == pd.Timestamp("2024-01-02")
        kwargs = polygon.client.list_aggs.call_args.kwargs
        assert kwargs["ticker"] == "SPY"
        assert kwargs["adjusted"] is True

    def test_polygon_supports_advertised_intervals(self, polygon):
        for interval in PolygonProvider.supported_intervals:
            polygon._parse_interval(interval)
        assert polygon._parse_interval("quarter") == (1, "quarter")
        assert polygon._parse_interval("15min") == (15, "minute")

    def test_polygon_parse_interval_fullmatch(self, polygon):
        with pytest.raises(ValueError):
            polygon._parse_interval("15minutesXYZ")

    def test_polygon_empty_is_invalid_symbol(self, polygon):
        polygon.client.list_aggs.return_value = iter([])
        with pytest.raises(InvalidSymbolError):
            polygon.fetch("NOPE", "1mo", "1d")

    @pytest.mark.parametrize(
        ("error", "exc"),
        [
            (Exception('{"status":"NOT_AUTHORIZED","message":"Unknown API Key"}'), AuthError),
            (Exception("You've exceeded the maximum requests per minute"), RateLimitError),
            (Exception("socket timeout"), ConnectionError),
        ],
    )
    def test_polygon_error_classification(self, polygon, error, exc):
        polygon.client.list_aggs.side_effect = error
        with pytest.raises(exc):
            polygon.fetch("SPY", "1mo", "1d")

    def test_polygon_rate_limit_metadata(self):
        assert PolygonProvider.rate_limit_per_minute == 5


class TestReviewFixes:
    def test_av_key_never_in_error_messages(self):
        provider = AlphaVantageProvider(ProviderConfig(api_key="SECRETKEY9", retries=0))
        body = {
            "Information": "We have detected your API key as SECRETKEY9 and our standard API "
            "rate limit is 25 requests per day."
        }
        with (
            patch("requests.get", return_value=_response(body=body)),
            pytest.raises(RateLimitError) as info,
        ):
            provider.fetch("SPY", "1mo", "1d")
        assert "SECRETKEY9" not in str(info.value)

    def test_av_free_daily_requests_compact_once(self):
        provider = AlphaVantageProvider(ProviderConfig(api_key="K", premium=False))
        today = datetime.now(UTC).date().isoformat()
        with patch("requests.get", return_value=_response(body=_av_daily([today]))) as get:
            provider.fetch("SPY", "2y", "1d")
        assert get.call_count == 1
        assert get.call_args.kwargs["params"]["outputsize"] == "compact"

    def test_polygon_max_retries_is_connection_error(self, polygon):
        polygon.client.list_aggs.side_effect = Exception(
            "HTTPSConnectionPool: Max retries exceeded with url (NameResolutionError)"
        )
        with pytest.raises(ConnectionError) as info:
            polygon.fetch("SPY", "1mo", "1d")
        assert not isinstance(info.value, RateLimitError)

    def test_yfinance_requests_raised_errors(self):
        ticker = MagicMock()
        ticker.history.return_value = _ohlcv(pd.to_datetime(["2024-01-02"]))
        with patch("yfinance.Ticker", return_value=ticker):
            YFinanceProvider().fetch("SPY", "1mo", "1d")
        assert ticker.history.call_args.kwargs["raise_errors"] is True

    def test_rate_limiter_shared_across_instances(self):
        a = PolygonProvider(ProviderConfig(api_key="K"))
        b = PolygonProvider(ProviderConfig(api_key="K"))
        c = PolygonProvider(ProviderConfig(api_key="OTHER"))
        assert a._limiter is b._limiter
        assert a._limiter is not c._limiter
