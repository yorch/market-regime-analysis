"""API behaviour tests: /api/v1/* happy and error paths against the mock provider,
the uniform error envelope, strict JSON, thread-pool execution and the WebSocket
monitoring lifecycle."""

import asyncio
import io
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.data_providers import MarketDataProvider, MockDataProvider
from mra_web import utils as utils_module, websocket as ws_module
from mra_web.utils import NumpyJSONResponse, run_in_thread, to_jsonable


def _strict_loads(text: str):
    def reject(value):
        raise ValueError(f"non-standard JSON constant {value}")

    return json.loads(text, parse_constant=reject)


def _assert_envelope(resp, code: int, error_code: str | None = None) -> dict:
    assert resp.status_code == code
    body = _strict_loads(resp.text)
    assert set(body) == {"error_code", "message", "details", "timestamp"}
    datetime.fromisoformat(body["timestamp"])
    if error_code:
        assert body["error_code"] == error_code
    return body


@pytest.fixture
def mock_provider():
    added = "mock" not in MarketDataProvider._providers
    MarketDataProvider.register(MockDataProvider)
    yield "mock"
    if added:
        MarketDataProvider._providers.pop("mock", None)


@pytest.fixture
def authed(client, auth_headers):
    """Client wrapper that always sends a valid JWT."""

    class _Client:
        def get(self, url, **kw):
            return client.get(url, headers=auth_headers, **kw)

        def post(self, url, **kw):
            return client.post(url, headers=auth_headers, **kw)

    return _Client()


# ── /api/v1 happy paths with the mock provider ──


class TestAnalysisEndpoints:
    def test_detailed(self, authed, mock_provider):
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "spy", "timeframe": "1D", "provider": "mock"},
        )
        assert resp.status_code == 200, resp.text
        body = _strict_loads(resp.text)
        assert body["symbol"] == "SPY"
        assert body["timeframe"] == "1D"
        assert body["current_regime"] in {r.value for r in MarketRegime}
        assert body["hmm_state"] >= 0

    def test_detailed_loads_only_requested_timeframe(self, authed, mock_provider, monkeypatch):
        seen = []
        original = MockDataProvider.fetch

        def spy(self, symbol, period, interval):
            seen.append(interval)
            return original(self, symbol, period, interval)

        monkeypatch.setattr(MockDataProvider, "fetch", spy)
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1D", "provider": "mock"},
        )
        assert resp.status_code == 200
        assert seen == ["1d"]

    def test_current_tolerates_failing_timeframes(self, authed, mock_provider):
        # The mock provider has no "6mo" period, so 1H fails; others still return.
        resp = authed.post("/api/v1/analysis/current", json={"symbol": "SPY", "provider": "mock"})
        assert resp.status_code == 200, resp.text
        timeframes = [a["timeframe"] for a in _strict_loads(resp.text)["analyses"]]
        assert "1D" in timeframes
        assert "1H" not in timeframes

    def test_multi_symbol(self, authed, mock_provider, monkeypatch):
        original = MockDataProvider.fetch

        def aligned(self, symbol, period, interval):
            # Mock timestamps derive from now(); align them so symbols overlap
            df = original(self, symbol, period, interval)
            df.index = df.index.normalize()
            return df

        monkeypatch.setattr(MockDataProvider, "fetch", aligned)
        resp = authed.post(
            "/api/v1/analysis/multi-symbol",
            json={"symbols": ["aaa", "BBB", " aaa ", ""], "timeframe": "1D", "provider": "mock"},
        )
        assert resp.status_code == 200, resp.text
        body = _strict_loads(resp.text)
        assert body["symbols"] == ["AAA", "BBB"]
        assert len(body["analyses"]) == 2
        metrics = body["portfolio_metrics"]
        assert metrics["analyzed_symbols"] == 2
        assert metrics["dominant_regime"] in {r.value for r in MarketRegime}
        assert set(body["correlations"]) == {"AAA", "BBB"}
        # Mock data is seeded identically per symbol: returns correlate perfectly
        assert body["correlations"]["AAA"]["BBB"] == pytest.approx(1.0)

    def test_multi_symbol_single_symbol(self, authed, mock_provider):
        resp = authed.post(
            "/api/v1/analysis/multi-symbol",
            json={"symbols": ["AAA"], "timeframe": "1D", "provider": "mock"},
        )
        assert resp.status_code == 200
        metrics = _strict_loads(resp.text)["portfolio_metrics"]
        assert metrics["correlation_risk"] is None

    def test_multi_symbol_analyzes_each_symbol_once(self, authed, mock_provider, monkeypatch):
        from mra_lib import MarketRegimeAnalyzer

        calls = []
        original = MarketRegimeAnalyzer.analyze_current_regime

        def counting(self, timeframe):
            calls.append(self.symbol)
            return original(self, timeframe)

        monkeypatch.setattr(MarketRegimeAnalyzer, "analyze_current_regime", counting)
        resp = authed.post(
            "/api/v1/analysis/multi-symbol",
            json={"symbols": ["AAA", "BBB"], "timeframe": "1D", "provider": "mock"},
        )
        assert resp.status_code == 200
        assert sorted(calls) == ["AAA", "BBB"]

    def test_position_sizing(self, authed):
        resp = authed.post(
            "/api/v1/position-sizing",
            json={
                "base_size": 0.02,
                "regime": "Bull Trending",
                "confidence": 0.8,
                "persistence": 0.7,
                "correlation": 0.1,
            },
        )
        assert resp.status_code == 200
        body = _strict_loads(resp.text)
        assert body["final_recommendation"] >= 0
        assert "kelly_criterion_applied" not in body["calculations"]
        assert "safety_caps_applied" not in body["calculations"]

    def test_position_sizing_zero_base(self, authed):
        resp = authed.post(
            "/api/v1/position-sizing",
            json={"base_size": 0.0, "regime": "Unknown", "confidence": 0.5, "persistence": 0.5},
        )
        assert resp.status_code == 200
        assert _strict_loads(resp.text)["calculations"]["regime_multiplier"] is None

    def test_providers(self, authed):
        resp = authed.get("/api/v1/providers")
        assert resp.status_code == 200
        assert "yfinance" in resp.json()["providers"]

    def test_csv_export(self, authed, mock_provider):
        resp = authed.post("/api/v1/export/csv", json={"symbol": "SPY", "provider": "mock"})
        assert resp.status_code == 200, resp.text
        df = pd.read_csv(io.StringIO(resp.text))
        assert int(resp.headers["x-record-count"]) == len(df) >= 1
        assert set(df["symbol"]) == {"SPY"}

    def test_chart(self, authed, mock_provider):
        resp = authed.post(
            "/api/v1/charts/generate",
            json={"symbol": "SPY", "timeframe": "1D", "days": 60, "provider": "mock"},
        )
        assert resp.status_code == 200
        assert resp.content.startswith(b"\x89PNG")

    def test_metrics_bounded_samples(self):
        metrics = utils_module.APIMetrics()
        for i in range(metrics.MAX_SAMPLES + 50):
            metrics.record_response_time("/x", float(i))
        assert len(metrics.response_times["/x"]) == metrics.MAX_SAMPLES


# ── Error envelope ──


class TestErrorEnvelope:
    def test_unknown_route(self, client):
        _assert_envelope(client.get("/nope"), 404, "NOT_FOUND")

    def test_method_not_allowed(self, client):
        _assert_envelope(client.delete("/health"), 405, "METHOD_NOT_ALLOWED")

    def test_unauthorized(self, client):
        resp = client.get("/api/v1/providers")
        _assert_envelope(resp, 401, "UNAUTHORIZED")
        assert resp.headers["www-authenticate"] == "Bearer"

    def test_validation_error_does_not_echo_input(self, authed):
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1W", "provider": "yfinance", "api_key": "SEKRIT"},
        )
        body = _assert_envelope(resp, 422, "VALIDATION_ERROR")
        assert body["details"]["errors"][0]["loc"] == ["body", "timeframe"]
        assert "SEKRIT" not in resp.text

    def test_unknown_provider_rejected(self, authed):
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1D", "provider": "nope"},
        )
        _assert_envelope(resp, 422, "VALIDATION_ERROR")

    def test_missing_provider_key_is_400_not_500(self, authed, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1D", "provider": "alphavantage"},
        )
        body = _assert_envelope(resp, 400, "API_KEY_REQUIRED")
        assert body["details"]["provider"] == "alphavantage"

    def test_no_data_is_400(self, authed, mock_provider, monkeypatch):
        def empty(self, symbol, period, interval):
            return pd.DataFrame()

        monkeypatch.setattr(MockDataProvider, "fetch", empty)
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1D", "provider": "mock"},
        )
        _assert_envelope(resp, 400, "BAD_REQUEST")

    def test_rate_limited(self, app_factory, auth_headers):
        c = TestClient(app_factory(rate_limit_per_minute=1))
        c.get("/api/v1/providers", headers=auth_headers)
        resp = c.get("/api/v1/providers", headers=auth_headers)
        _assert_envelope(resp, 429, "RATE_LIMITED")
        assert int(resp.headers["retry-after"]) >= 1

    def test_unhandled_exception(self, app_factory):
        app = app_factory()

        @app.get("/boom")
        async def boom():
            raise RuntimeError("secret apikey=LEAKED123456")

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/boom")
        body = _assert_envelope(resp, 500, "INTERNAL_SERVER_ERROR")
        assert "LEAKED" not in resp.text
        assert body["message"] == "An unexpected error occurred"

    def test_slowapi_rate_limit_handler(self):
        from mra_web.errors import rate_limit_exception_handler

        resp = asyncio.run(rate_limit_exception_handler(None, Exception()))  # type: ignore[arg-type]
        assert resp.status_code == 429


# ── Strict JSON ──


class TestStrictJSON:
    def test_render_non_finite_and_numpy(self):
        content = {
            "nan": float("nan"),
            "inf": np.float64("inf"),
            "neg": -np.inf,
            "flag": np.bool_(True),
            "int": np.int64(3),
            "arr": np.array([1.0, np.nan]),
            "nat": pd.NaT,
            "ts": pd.Timestamp("2024-01-02T03:04:05", tz="UTC"),
            "dt": datetime(2024, 1, 2, tzinfo=UTC),
            "enum": MarketRegime.BULL_TRENDING,
            "nested": [{"x": float("nan")}],
        }
        data = _strict_loads(NumpyJSONResponse(content).body.decode())
        assert data["nan"] is None and data["inf"] is None and data["neg"] is None
        assert data["flag"] is True
        assert data["int"] == 3
        assert data["arr"] == [1.0, None]
        assert data["nat"] is None
        assert data["ts"].startswith("2024-01-02T03:04:05")
        assert data["enum"] == "Bull Trending"
        assert data["nested"] == [{"x": None}]

    def test_nan_analysis_serializes(self, authed, monkeypatch):
        analysis = RegimeAnalysis(
            current_regime=MarketRegime.UNKNOWN,
            hmm_state=np.int64(0),
            transition_probability=float("nan"),
            regime_persistence=float("nan"),
            recommended_strategy=TradingStrategy.AVOID
            if hasattr(TradingStrategy, "AVOID")
            else next(iter(TradingStrategy)),
            position_sizing_multiplier=float("inf"),
            risk_level="High",
            arbitrage_opportunities=[],
            statistical_signals=[],
            key_levels={"support": float("nan")},
            regime_confidence=np.float64("nan"),
        )

        class Stub:
            def __init__(self, *a, **k):
                pass

            def analyze_current_regime(self, timeframe):
                return analysis

        from mra_web import endpoints

        monkeypatch.setattr(endpoints, "MarketRegimeAnalyzer", Stub)
        resp = authed.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"},
        )
        assert resp.status_code == 200, resp.text
        body = _strict_loads(resp.text)
        assert body["regime_confidence"] is None
        assert body["metrics"]["key_levels"]["support"] is None

    def test_to_jsonable_passthrough(self):
        assert to_jsonable("x") == "x"
        assert to_jsonable((1, 2)) == [1, 2]
        obj = object()
        assert to_jsonable(obj) is obj


# ── Thread pool helper ──


class TestRunInThread:
    def test_kwargs(self):
        def add(a, b=0):
            return a + b

        assert asyncio.run(run_in_thread(add, 1, b=2)) == 3

    def test_timeout(self):
        def slow():
            time.sleep(0.5)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(run_in_thread(slow, _timeout=0.05))
        assert exc.value.status_code == 504

    def test_does_not_block_event_loop(self):
        async def scenario():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            task = asyncio.create_task(ticker())
            await run_in_thread(time.sleep, 0.2)
            task.cancel()
            return ticks

        assert asyncio.run(scenario()) >= 5


# ── WebSocket lifecycle ──


def _analysis(regime: MarketRegime, confidence: float = 0.9) -> RegimeAnalysis:
    return RegimeAnalysis(
        current_regime=regime,
        hmm_state=1,
        transition_probability=0.1,
        regime_persistence=0.8,
        recommended_strategy=next(iter(TradingStrategy)),
        position_sizing_multiplier=1.0,
        risk_level="Low",
        arbitrage_opportunities=[],
        statistical_signals=[],
        key_levels={},
        regime_confidence=confidence,
    )


@pytest.fixture
def ws_analysis(monkeypatch):
    """Patch the blocking analysis; records calls and can be scripted."""
    state = {"calls": 0, "results": []}

    def fake(symbol, provider, api_key):
        state["calls"] += 1
        if state["results"]:
            result = state["results"].pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return _analysis(MarketRegime.BULL_TRENDING)

    monkeypatch.setattr(ws_module, "analyze_symbol", fake)
    return state


WS_BASE = "/ws/monitoring/SPY?provider=yfinance"


class TestWebSocketLifecycle:
    def test_connect_update_disconnect_terminates_loop(self, client, api_key_headers, ws_analysis):
        start = time.monotonic()
        with client.websocket_connect(f"{WS_BASE}&interval=3600", headers=api_key_headers) as ws:
            assert ws.receive_json()["message_type"] == "connection"
            update = ws.receive_json()
            assert update["message_type"] == "update"
            assert update["data"]["current_regime"] == "Bull Trending"
            assert ws_module.manager.get_connection_count() == 1
        # Exiting waits for the endpoint: with a 1h interval this only returns
        # promptly if the loop noticed the disconnect.
        assert time.monotonic() - start < 10
        assert ws_module.manager.get_connection_count() == 0
        assert ws_module.manager.reserved == 0
        assert ws_analysis["calls"] == 1

    def test_regime_change_alert(self, client, api_key_headers, ws_analysis, monkeypatch):
        real_wait = asyncio.wait

        async def fast_wait(aws, timeout=None, **kw):
            # Shrink the inter-tick sleep so the second tick happens immediately
            return await real_wait(aws, timeout=0.01 if timeout else None, **kw)

        monkeypatch.setattr(ws_module.asyncio, "wait", fast_wait)
        ws_analysis["results"] = [
            _analysis(MarketRegime.BULL_TRENDING),
            _analysis(MarketRegime.BEAR_TRENDING),
        ]
        with client.websocket_connect(f"{WS_BASE}&interval=60", headers=api_key_headers) as ws:
            ws.receive_json()  # connection
            assert ws.receive_json()["data"]["regime_change"] is False
            second = ws.receive_json()
            assert second["data"]["regime_change"] is True
            alert = ws.receive_json()
            assert alert["message_type"] == "alert"
            assert alert["data"]["new_regime"] == "Bear Trending"

    def test_errors_are_generic_and_close_1011(
        self, client, api_key_headers, ws_analysis, monkeypatch
    ):
        real_wait = asyncio.wait

        async def fast_wait(aws, timeout=None, **kw):
            return await real_wait(aws, timeout=0.01 if timeout else None, **kw)

        monkeypatch.setattr(ws_module.asyncio, "wait", fast_wait)
        ws_analysis["results"] = [
            ConnectionError("https://x?apikey=LEAKED123456")
        ] * ws_module.MAX_CONSECUTIVE_ERRORS
        with client.websocket_connect(f"{WS_BASE}&interval=60", headers=api_key_headers) as ws:
            ws.receive_json()
            for i in range(ws_module.MAX_CONSECUTIVE_ERRORS):
                msg = ws.receive_json()
                assert msg["message_type"] == "error"
                assert msg["data"]["error_count"] == i + 1
                assert "LEAKED" not in json.dumps(msg)
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1011

    @pytest.mark.parametrize(
        "query",
        ["interval=abc", "interval=10", "interval=99999", "provider=bogus"],
    )
    def test_invalid_query_rejected_1008(self, client, api_key_headers, query, ws_analysis):
        url = f"/ws/monitoring/SPY?{query}"
        if "provider" not in query:
            url += "&provider=yfinance"
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(url, headers=api_key_headers),
        ):
            pass
        assert exc.value.code == 1008
        assert ws_analysis["calls"] == 0

    def test_registered_provider_accepted(
        self, client, api_key_headers, ws_analysis, mock_provider
    ):
        url = "/ws/monitoring/SPY?provider=mock&interval=3600"
        with client.websocket_connect(url, headers=api_key_headers) as ws:
            assert ws.receive_json()["message_type"] == "connection"
