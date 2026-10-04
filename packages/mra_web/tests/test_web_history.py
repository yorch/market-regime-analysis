"""Tests for ``GET /api/v1/regimes/{symbol}/history``."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from mra_lib.errors import StorageError
from mra_lib.storage import MAX_HISTORY_LIMIT, RegimeRecord, SQLiteRegimeStore

URL = "/api/v1/regimes/{}/history"
BASE = datetime(2026, 1, 5)


def _record(symbol: str = "SPY", timeframe: str = "1D", bar_time: datetime = BASE, **kw):
    values = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_time": bar_time,
        "regime": "Bull Trending",
        "confidence": 0.8,
        "persistence": 0.6,
        "transition_probability": 0.9,
        "recommended_strategy": "Trend Following",
        "provider": "mock",
        "close": 100.0,
    }
    values.update(kw)
    return RegimeRecord(**values)


def _envelope(resp, code: int, error_code: str) -> dict:
    assert resp.status_code == code, resp.text
    body = resp.json()
    assert set(body) == {"error_code", "message", "details", "timestamp"}
    assert body["error_code"] == error_code
    return body


@pytest.fixture
def db_store(tmp_path, monkeypatch) -> SQLiteRegimeStore:
    path = tmp_path / "data" / "regimes.db"
    monkeypatch.setenv("MRA_DB_PATH", str(path))
    return SQLiteRegimeStore(path)


class TestHistoryAuth:
    def test_history_requires_auth(self, client, db_store):
        resp = client.get(URL.format("SPY"))
        _envelope(resp, 401, "UNAUTHORIZED")

    def test_history_rejects_bad_token(self, client, db_store):
        resp = client.get(URL.format("SPY"), headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401

    def test_history_accepts_jwt_and_api_key(self, client, db_store, auth_headers, api_key_headers):
        assert client.get(URL.format("SPY"), headers=auth_headers).status_code == 200
        assert client.get(URL.format("SPY"), headers=api_key_headers).status_code == 200


class TestHistoryValidation:
    @pytest.mark.parametrize("symbol", ["-SPY", "TOOLONGSYMBOL12345", "SP%20Y", "%3Dcmd"])
    def test_history_invalid_symbol(self, client, auth_headers, db_store, symbol):
        resp = client.get(URL.format(symbol), headers=auth_headers)
        body = _envelope(resp, 422, "VALIDATION_ERROR")
        assert body["details"]["errors"][0]["loc"] == ["path", "symbol"]

    def test_history_invalid_timeframe(self, client, auth_headers, db_store):
        resp = client.get(URL.format("SPY"), params={"timeframe": "4H"}, headers=auth_headers)
        body = _envelope(resp, 422, "VALIDATION_ERROR")
        assert body["details"]["errors"][0]["loc"] == ["query", "timeframe"]

    @pytest.mark.parametrize("limit", ["0", "-5", "abc"])
    def test_history_invalid_limit(self, client, auth_headers, db_store, limit):
        resp = client.get(URL.format("SPY"), params={"limit": limit}, headers=auth_headers)
        _envelope(resp, 422, "VALIDATION_ERROR")

    def test_history_invalid_dates(self, client, auth_headers, db_store):
        resp = client.get(URL.format("SPY"), params={"since": "yesterday"}, headers=auth_headers)
        _envelope(resp, 422, "VALIDATION_ERROR")
        resp = client.get(
            URL.format("SPY"),
            params={"since": "2026-02-01", "until": "2026-01-01"},
            headers=auth_headers,
        )
        _envelope(resp, 422, "VALIDATION_ERROR")

    @pytest.mark.parametrize(
        "params",
        [
            {"since": "0001-01-01T00:00:00+05:00"},
            {"until": "9999-12-31T23:59:59-05:00"},
            {"since": "0001-01-01T00:00:00+05:00", "until": "2026-01-01T00:00:00"},
        ],
    )
    def test_history_out_of_range_dates(self, client, auth_headers, db_store, params):
        resp = client.get(URL.format("SPY"), params=params, headers=auth_headers)
        _envelope(resp, 422, "VALIDATION_ERROR")

    def test_history_validation_does_not_echo_input(self, client, auth_headers, db_store):
        resp = client.get(URL.format("SPY"), params={"timeframe": "<script>"}, headers=auth_headers)
        assert "<script>" not in resp.text


class TestHistoryResults:
    def test_history_empty_for_unknown_symbol(self, client, auth_headers, db_store):
        resp = client.get(URL.format("nope"), headers=auth_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "symbol": "NOPE",
            "timeframe": None,
            "limit": 100,
            "count": 0,
            "records": [],
        }

    def test_history_populated(self, client, auth_headers, db_store):
        for days in range(3):
            db_store.save(_record(bar_time=BASE + timedelta(days=days), close=100.0 + days))
        db_store.save(_record(timeframe="1H", bar_time=BASE + timedelta(days=10)))
        db_store.save(_record(symbol="QQQ"))

        resp = client.get(URL.format("spy"), params={"timeframe": "1D"}, headers=auth_headers)
        assert resp.status_code == 200, resp.text
        body = json.loads(resp.text)
        assert body["symbol"] == "SPY"
        assert body["timeframe"] == "1D"
        assert body["count"] == 3
        records = body["records"]
        assert [r["bar_time"] for r in records] == [
            "2026-01-07T00:00:00",
            "2026-01-06T00:00:00",
            "2026-01-05T00:00:00",
        ]
        first = records[0]
        assert set(first) == {
            "symbol",
            "timeframe",
            "bar_time",
            "recorded_at",
            "regime",
            "confidence",
            "persistence",
            "transition_probability",
            "recommended_strategy",
            "close",
            "provider",
        }
        assert first["regime"] == "Bull Trending"
        assert first["close"] == 102.0
        assert first["provider"] == "mock"

        # All timeframes, newest first
        resp = client.get(URL.format("SPY"), headers=auth_headers)
        assert [r["timeframe"] for r in resp.json()["records"]] == ["1H", "1D", "1D", "1D"]

    def test_history_since_until(self, client, auth_headers, db_store):
        for days in range(5):
            db_store.save(_record(bar_time=BASE + timedelta(days=days)))
        resp = client.get(
            URL.format("SPY"),
            params={"since": "2026-01-06T00:00:00", "until": "2026-01-08T00:00:00Z"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert [r["bar_time"][:10] for r in resp.json()["records"]] == [
            "2026-01-08",
            "2026-01-07",
            "2026-01-06",
        ]

    def test_history_aware_bounds_are_converted_to_utc(self, client, auth_headers, db_store):
        for hour in range(10, 15):
            db_store.save(_record(timeframe="1H", bar_time=datetime(2026, 1, 5, hour)))
        # 13:00+02:00 == 11:00Z; 08:00-05:00 == 13:00Z
        resp = client.get(
            URL.format("SPY"),
            params={"since": "2026-01-05T13:00:00+02:00", "until": "2026-01-05T08:00:00-05:00"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert [r["bar_time"][11:13] for r in resp.json()["records"]] == ["13", "12", "11"]

    def test_history_does_not_use_analysis_slots(self, client, auth_headers, db_store):
        from mra_web import utils as utils_module

        slots = utils_module.analysis_slots()
        taken = 0
        while slots.acquire(blocking=False):
            taken += 1
        try:
            resp = client.get(URL.format("SPY"), headers=auth_headers)
        finally:
            for _ in range(taken):
                slots.release()
        assert resp.status_code == 200, resp.text

    def test_history_limit(self, client, auth_headers, db_store):
        for days in range(5):
            db_store.save(_record(bar_time=BASE + timedelta(days=days)))
        resp = client.get(URL.format("SPY"), params={"limit": 2}, headers=auth_headers)
        body = resp.json()
        assert body["limit"] == 2
        assert body["count"] == 2

    def test_history_limit_is_capped(self, client, auth_headers, db_store):
        with patch.object(SQLiteRegimeStore, "history", return_value=[]) as history:
            resp = client.get(
                URL.format("SPY"), params={"limit": MAX_HISTORY_LIMIT * 10}, headers=auth_headers
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["limit"] == MAX_HISTORY_LIMIT
        assert history.call_args.kwargs["limit"] == MAX_HISTORY_LIMIT

    def test_history_db_error_is_generic(self, client, auth_headers, db_store):
        secret = "/very/secret/path/regimes.db"
        with patch.object(SQLiteRegimeStore, "history", side_effect=StorageError(secret)):
            resp = client.get(URL.format("SPY"), headers=auth_headers)
        body = _envelope(resp, 503, "SERVICE_UNAVAILABLE")
        assert body["message"] == "Regime history unavailable"
        assert secret not in resp.text

    def test_history_unreadable_db_is_generic(self, client, auth_headers, tmp_path, monkeypatch):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        monkeypatch.setenv("MRA_DB_PATH", str(blocker / "regimes.db"))
        resp = client.get(URL.format("SPY"), headers=auth_headers)
        body = _envelope(resp, 503, "SERVICE_UNAVAILABLE")
        assert str(tmp_path) not in resp.text
        assert body["message"] == "Regime history unavailable"

    def test_analysis_endpoints_do_not_write_history(self, client, auth_headers, db_store):
        from mra_lib.data_providers import MarketDataProvider, MockDataProvider

        added = "mock" not in MarketDataProvider._providers
        MarketDataProvider.register(MockDataProvider)
        try:
            resp = client.post(
                "/api/v1/analysis/detailed",
                json={"symbol": "SPY", "timeframe": "1D", "provider": "mock"},
                headers=auth_headers,
            )
        finally:
            if added:
                MarketDataProvider._providers.pop("mock", None)
        assert resp.status_code == 200, resp.text
        assert db_store.symbols() == []
