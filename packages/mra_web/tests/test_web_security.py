"""Security tests for mra_web: config hardening, auth matrix, input validation,
error-detail hygiene, CSV/chart export, CORS, docs, rate limits and WebSockets."""

import io
import logging
import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from jose import jwt
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from mra_lib.data_providers import MarketDataProvider, MockDataProvider
from mra_web import endpoints, websocket as ws_module
from mra_web.app import app
from mra_web.auth import authenticate_credentials, create_access_token, mint_token_main
from mra_web.config import APIConfig, ConfigError
from mra_web.models import (
    ExportCSVRequest,
    MultiSymbolAnalysisRequest,
    normalize_symbol,
)
from mra_web.security import SecretScrubFilter, scrub_secrets

JWT_SECRET = os.environ["JWT_SECRET"]
API_KEY = os.environ["API_KEYS"]
PROTECTED = "/api/v1/providers"


def _token(claims: dict, secret: str = JWT_SECRET, algorithm: str = "HS256") -> str:
    return jwt.encode(claims, secret, algorithm=algorithm)


def _future(hours: int = 1) -> int:
    return int((datetime.now(UTC) + timedelta(hours=hours)).timestamp())


# ── Config hardening ──


class TestConfigHardening:
    @pytest.mark.parametrize("secret", ["", "your-secret-key-change-in-production", "short-secret"])
    def test_weak_secret_refused_in_production(self, secret):
        with pytest.raises(ConfigError):
            APIConfig(jwt_secret=secret, environment="production")

    def test_from_env_defaults_to_production_and_refuses(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with pytest.raises(ConfigError):
            APIConfig.from_env()

    def test_development_generates_ephemeral_secret(self, caplog):
        with caplog.at_level(logging.WARNING):
            c = APIConfig(jwt_secret="", environment="development")
        assert len(c.jwt_secret) >= 32
        assert c.jwt_secret_ephemeral is True
        assert "JWT_SECRET" in caplog.text
        other = APIConfig(jwt_secret="", environment="development")
        assert other.jwt_secret != c.jwt_secret

    def test_short_api_key_refused_in_production(self):
        with pytest.raises(ConfigError):
            APIConfig(jwt_secret=JWT_SECRET, api_keys=["short"])

    def test_env_parsing(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
        monkeypatch.setenv("ENVIRONMENT", " Production ")
        monkeypatch.setenv("API_KEYS", f"{API_KEY}, ,{API_KEY}x")
        monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
        monkeypatch.setenv("LOG_LEVEL", "info")
        monkeypatch.setenv("ENABLE_DOCS", "true")
        c = APIConfig.from_env()
        assert c.environment == "production"
        assert c.api_keys == [API_KEY, f"{API_KEY}x"]
        assert c.cors_origins == ["https://a.example", "https://b.example"]
        assert c.log_level == "INFO"
        assert c.docs_enabled is True

    def test_secret_not_in_repr(self):
        c = APIConfig(jwt_secret=JWT_SECRET, api_keys=[API_KEY])
        assert JWT_SECRET not in repr(c)
        assert API_KEY not in repr(c)


# ── Auth matrix ──


class TestAuthMatrix:
    def test_no_credentials(self, client):
        resp = client.get(PROTECTED)
        assert resp.status_code == 401

    def test_valid_jwt(self, client, auth_headers):
        assert client.get(PROTECTED, headers=auth_headers).status_code == 200

    def test_valid_api_key(self, client, api_key_headers):
        assert client.get(PROTECTED, headers=api_key_headers).status_code == 200

    def test_invalid_api_key(self, client):
        resp = client.get(PROTECTED, headers={"X-API-Key": "wrong-key-0123456789"})
        assert resp.status_code == 401

    @pytest.mark.parametrize("key", ["demo-api-key-12345", "admin-api-key-67890"])
    def test_old_hardcoded_keys_rejected(self, client, key):
        assert client.get(PROTECTED, headers={"X-API-Key": key}).status_code == 401

    def test_api_key_query_param_not_accepted(self, client):
        assert client.get(f"{PROTECTED}?api_key={API_KEY}").status_code == 401

    def test_tampered_jwt(self, client):
        token = _token({"sub": "mallory", "exp": _future()})
        header, payload, sig = token.split(".")
        tampered = f"{header}.{payload}.{sig[::-1]}"
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {tampered}"})
        assert resp.status_code == 401

    def test_wrong_secret(self, client):
        token = _token({"sub": "mallory", "exp": _future()}, secret="z" * 40)
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_expired_jwt(self, client):
        token = _token({"sub": "alice", "exp": _future(-1)})
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_jwt_without_exp(self, client):
        token = _token({"sub": "alice"})
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_jwt_without_sub(self, client):
        token = _token({"exp": _future()})
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_jwt_other_algorithm_rejected(self, client):
        token = _token({"sub": "alice", "exp": _future()}, algorithm="HS512")
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_unsigned_jwt_rejected(self, client):
        import base64
        import json

        def b64(d: dict) -> str:
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        token = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64({'sub': 'x', 'exp': _future()})}."
        resp = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_create_access_token_requires_sub(self):
        with pytest.raises(ValueError):
            create_access_token({"user": "x"})

    def test_metrics_requires_auth(self, client, auth_headers):
        assert client.get("/metrics").status_code == 401
        assert client.get("/api/v1/metrics").status_code == 401
        assert client.get("/metrics", headers=auth_headers).status_code == 200

    def test_ws_status_requires_auth(self, client, api_key_headers):
        assert client.get("/ws/monitoring/status").status_code == 401
        assert client.get("/ws/monitoring/status", headers=api_key_headers).status_code == 200

    def test_public_probes(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/health").status_code == 200
        assert "environment" not in client.get("/health").json()

    def test_development_bypass_only_without_credentials(self, app_factory):
        dev = TestClient(app_factory(environment="development"))
        assert dev.get(PROTECTED).status_code == 200
        bad = dev.get(PROTECTED, headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401

    def test_authenticate_credentials_production(self):
        c = APIConfig(jwt_secret=JWT_SECRET, api_keys=[API_KEY])
        assert authenticate_credentials(c, api_key=API_KEY).username.startswith("apikey-")


# ── Docs, debug, CORS, rate limits ──


class TestAppHardening:
    def test_docs_disabled_in_production(self, client):
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404

    def test_docs_can_be_enabled(self, app_factory):
        c = TestClient(app_factory(enable_docs=True))
        assert c.get("/openapi.json").status_code == 200

    def test_debug_config_development_only(self, client, app_factory):
        assert client.get("/debug/config").status_code in (401, 404)
        dev = TestClient(app_factory(environment="development"))
        resp = dev.get("/debug/config")
        assert resp.status_code == 200
        assert JWT_SECRET not in resp.text

    def test_cors_default_denies(self, client, auth_headers):
        resp = client.get(PROTECTED, headers={**auth_headers, "Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_configured_origin(self, app_factory, auth_headers):
        c = TestClient(app_factory(cors_origins=["https://app.example"]))
        ok = c.get(PROTECTED, headers={**auth_headers, "Origin": "https://app.example"})
        assert ok.headers["access-control-allow-origin"] == "https://app.example"
        assert ok.headers["access-control-allow-credentials"] == "true"
        evil = c.get(PROTECTED, headers={**auth_headers, "Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in evil.headers

    def test_cors_wildcard_never_allows_credentials(self, app_factory, auth_headers):
        c = TestClient(app_factory(cors_origins=["*"]))
        resp = c.get(PROTECTED, headers={**auth_headers, "Origin": "https://evil.example"})
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in resp.headers

    def test_rate_limit_applies_to_api_routes(self, app_factory, auth_headers):
        c = TestClient(app_factory(rate_limit_per_minute=2))
        codes = [c.get(PROTECTED, headers=auth_headers).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        # Health probes are exempt
        assert all(c.get("/health").status_code == 200 for _ in range(5))


# ── Input validation ──


class TestInputValidation:
    @pytest.mark.parametrize("symbol", ["SPY", "brk.b", "^GSPC", "ES=F", "BTC-USD"])
    def test_valid_symbols(self, symbol):
        assert normalize_symbol(symbol) == symbol.upper()

    @pytest.mark.parametrize(
        "symbol", ["", "   ", "SPY;rm -rf", "=CMD()", "-SPY", "../etc", "A" * 16, "SP Y"]
    )
    def test_invalid_symbols(self, symbol):
        with pytest.raises(ValueError):
            normalize_symbol(symbol)

    def test_symbols_deduped_and_blank_dropped(self):
        req = MultiSymbolAnalysisRequest(symbols=["spy", " SPY ", "", "qqq"], timeframe="1D")
        assert req.symbols == ["SPY", "QQQ"]

    def test_symbols_capped(self):
        with pytest.raises(ValidationError):
            MultiSymbolAnalysisRequest(symbols=[f"S{i}" for i in range(21)], timeframe="1D")

    def test_all_blank_symbols_rejected(self):
        with pytest.raises(ValidationError):
            MultiSymbolAnalysisRequest(symbols=["", "  "], timeframe="1D")

    @pytest.mark.parametrize(
        "name", ["../../etc/passwd", "/tmp/x.csv", "a/b.csv", ".hidden", "a b.csv", "x" * 120]
    )
    def test_export_filename_rejects_paths(self, name):
        with pytest.raises(ValidationError):
            ExportCSVRequest(symbol="SPY", filename=name)

    def test_export_filename_adds_extension(self):
        assert ExportCSVRequest(symbol="SPY", filename="report").filename == "report.csv"

    def test_invalid_symbol_returns_422(self, client, auth_headers):
        resp = client.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY;DROP", "timeframe": "1D", "provider": "yfinance"},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ── Endpoint behaviour with a stubbed analyzer ──


class _StubAnalyzer:
    error: Exception | None = None

    def __init__(self, symbol, provider_flag, api_key=None, periods=None):
        if _StubAnalyzer.error is not None:
            raise _StubAnalyzer.error
        self.symbol = symbol

    def build_export_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"symbol": self.symbol, "timeframe": "1D", "regime": "Bull Trending"},
                {"symbol": self.symbol, "timeframe": "1H", "regime": "Mean Reverting"},
            ]
        )

    def render_regime_chart_png(self, timeframe, days=60):
        return b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def stub_analyzer(monkeypatch):
    _StubAnalyzer.error = None
    monkeypatch.setattr(endpoints, "MarketRegimeAnalyzer", _StubAnalyzer)
    yield _StubAnalyzer
    _StubAnalyzer.error = None


class TestExportAndCharts:
    def test_csv_export_streams_csv(
        self, client, auth_headers, stub_analyzer, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        resp = client.post(
            "/api/v1/export/csv",
            json={"symbol": "spy", "provider": "yfinance", "filename": "my_export"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert resp.headers["x-record-count"] == "2"
        assert 'filename="my_export.csv"' in resp.headers["content-disposition"]
        df = pd.read_csv(io.StringIO(resp.text))
        assert list(df["symbol"]) == ["SPY", "SPY"]
        assert list(tmp_path.iterdir()) == []  # nothing written server-side

    def test_csv_export_path_traversal_rejected(self, client, auth_headers, stub_analyzer):
        resp = client.post(
            "/api/v1/export/csv",
            json={"symbol": "SPY", "provider": "yfinance", "filename": "../../pwn.csv"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_chart_returns_png(self, client, auth_headers, stub_analyzer):
        resp = client.post(
            "/api/v1/charts/generate",
            json={"symbol": "SPY", "timeframe": "1D", "days": 30, "provider": "yfinance"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(b"\x89PNG")

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (ConnectionError("GET https://x/query?apikey=LEAKED123456 failed"), 503),
            (ValueError("Data loading failed: url?apikey=LEAKED123456"), 400),
            (RuntimeError("boom apikey=LEAKED123456"), 500),
        ],
    )
    def test_errors_do_not_leak_details(self, client, auth_headers, stub_analyzer, error, code):
        stub_analyzer.error = error
        resp = client.post(
            "/api/v1/analysis/detailed",
            json={"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"},
            headers=auth_headers,
        )
        assert resp.status_code == code
        assert "LEAKED" not in resp.text
        assert "apikey" not in resp.text

    @pytest.mark.parametrize("path", ["/api/v1/export/csv", "/api/v1/charts/generate"])
    def test_export_errors_do_not_leak(self, client, auth_headers, stub_analyzer, path):
        stub_analyzer.error = RuntimeError("token=LEAKED123456")
        body = {"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"}
        resp = client.post(path, json=body, headers=auth_headers)
        assert resp.status_code == 500
        assert "LEAKED" not in resp.text


# ── Analyzer rendering (real analyzer on mock data) ──


@pytest.fixture
def mock_provider():
    added = "mock" not in MarketDataProvider._providers
    MarketDataProvider.register(MockDataProvider)
    yield
    if added:
        MarketDataProvider._providers.pop("mock", None)


class TestAnalyzerOutputs:
    def test_png_and_export_dataframe(self, mock_provider):
        from mra_lib import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer("TEST", periods={"1D": "2y"}, provider_flag="mock")
        png = analyzer.render_regime_chart_png("1D", 60)
        assert png.startswith(b"\x89PNG")
        df = analyzer.build_export_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "TEST"

    def test_insufficient_data_raises(self, mock_provider):
        from mra_lib import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer("TEST", periods={"1D": "2y"}, provider_flag="mock")
        with pytest.raises(ValueError):
            analyzer.render_regime_chart_png("1D", 5)

    def test_cli_export_still_writes_file(self, mock_provider, tmp_path):
        from mra_lib import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer("TEST", periods={"1D": "2y"}, provider_flag="mock")
        target = tmp_path / "out.csv"
        analyzer.export_analysis_to_csv(str(target))
        assert len(pd.read_csv(target)) == 1


# ── Secret scrubbing ──


class TestScrubbing:
    @pytest.mark.parametrize(
        "text",
        [
            "https://www.alphavantage.co/query?function=X&apikey=ABCDEF123456&symbol=SPY",
            "api_key='ABCDEF123456'",
            "Authorization: Bearer ABCDEF123456.abc.def",
            "token=ABCDEF123456",
        ],
    )
    def test_scrub(self, text):
        assert "ABCDEF123456" not in scrub_secrets(text)

    def test_scrubs_configured_secret_values(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "polygon-secret-value")
        assert "polygon-secret-value" not in scrub_secrets("error polygon-secret-value here")
        assert JWT_SECRET not in scrub_secrets(f"x {JWT_SECRET} y")

    def test_filter_scrubs_message_and_traceback(self):
        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "failed %s", ("apikey=ABCDEF123456",), None
        )
        try:
            raise ConnectionError("https://x?apikey=ABCDEF123456")
        except ConnectionError:
            import sys

            record.exc_info = sys.exc_info()
        SecretScrubFilter().filter(record)
        assert "ABCDEF123456" not in record.getMessage()
        assert "ABCDEF123456" not in (record.exc_text or "")


# ── Token minting CLI ──


class TestMintToken:
    def test_mints_verifiable_token(self, capsys):
        assert mint_token_main(["--sub", "alice", "--hours", "1"]) == 0
        token = capsys.readouterr().out.strip()
        claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        assert claims["sub"] == "alice"
        assert "exp" in claims

    def test_refuses_without_real_secret(self, monkeypatch, capsys):
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("JWT_SECRET", "")
        assert mint_token_main(["--sub", "alice"]) == 2

    def test_refuses_weak_secret_in_production(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "weak")
        assert mint_token_main(["--sub", "alice"]) == 2


# ── WebSocket ──


@pytest.fixture
def fake_loop(monkeypatch):
    """Replace the monitoring loop: send one message, then wait for disconnect."""

    async def _loop(websocket, symbol, provider, api_key, interval):
        await websocket.send_json({"message_type": "update", "symbol": symbol})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    monkeypatch.setattr(ws_module, "monitoring_loop", _loop)


WS_URL = "/ws/monitoring/SPY?provider=yfinance&interval=60"


class TestWebSocketSecurity:
    def test_rejects_without_credentials(self, client, fake_loop, monkeypatch):
        monkeypatch.setattr(ws_module, "AUTH_MESSAGE_TIMEOUT", 0.2)
        with client.websocket_connect(WS_URL) as ws, pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008

    def test_rejects_invalid_token_before_accept(self, client, fake_loop):
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(f"{WS_URL}&token=garbage"),
        ):
            pass
        assert exc.value.code == 1008

    def test_accepts_query_token(self, client, fake_loop):
        token = create_access_token({"sub": "alice"})
        with client.websocket_connect(f"{WS_URL}&token={token}") as ws:
            assert ws.receive_json()["message_type"] == "connection"
            assert ws.receive_json()["message_type"] == "update"

    def test_accepts_api_key_header(self, client, fake_loop):
        with client.websocket_connect(WS_URL, headers={"X-API-Key": API_KEY}) as ws:
            assert ws.receive_json()["message_type"] == "connection"

    def test_first_message_auth(self, client, fake_loop):
        token = create_access_token({"sub": "alice"})
        with client.websocket_connect(WS_URL) as ws:
            ws.send_json({"token": token})
            assert ws.receive_json()["message_type"] == "connection"

    def test_first_message_auth_invalid(self, client, fake_loop):
        with client.websocket_connect(WS_URL) as ws:
            ws.send_json({"token": "nope"})
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 1008

    def test_rejects_disallowed_origin(self, client, fake_loop):
        headers = {"X-API-Key": API_KEY, "Origin": "https://evil.example"}
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(WS_URL, headers=headers),
        ):
            pass
        assert exc.value.code == 1008

    def test_allows_configured_origin(self, app_factory, fake_loop):
        c = TestClient(app_factory(cors_origins=["https://app.example"]))
        headers = {"X-API-Key": API_KEY, "Origin": "https://app.example"}
        with c.websocket_connect(WS_URL, headers=headers) as ws:
            assert ws.receive_json()["message_type"] == "connection"

    def test_rejects_invalid_symbol(self, client, fake_loop):
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(
                "/ws/monitoring/BAD;SYM?provider=yfinance", headers={"X-API-Key": API_KEY}
            ),
        ):
            pass
        assert exc.value.code == 1008

    def test_per_ip_connection_cap(self, app_factory, fake_loop):
        c = TestClient(app_factory(ws_max_connections_per_ip=1))
        headers = {"X-API-Key": API_KEY}
        with c.websocket_connect(WS_URL, headers=headers) as first:
            assert first.receive_json()["message_type"] == "connection"
            with (
                pytest.raises(WebSocketDisconnect) as exc,
                c.websocket_connect(WS_URL, headers=headers),
            ):
                pass
            assert exc.value.code == 1013
        # Slot released after disconnect
        with c.websocket_connect(WS_URL, headers=headers) as again:
            assert again.receive_json()["message_type"] == "connection"
        assert ws_module.manager.reserved == 0

    def test_test_endpoint_removed(self, client):
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/test"):
            pass


def test_default_app_is_production():
    assert app.state.config.environment == "production"
