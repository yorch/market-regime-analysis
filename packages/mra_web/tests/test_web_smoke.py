"""Tests for mra_web package — FastAPI app, models, utils, auth, config."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from mra_web.app import app
from mra_web.auth import (
    AuthConfig,
    TokenData,
    User,
    create_access_token,
    create_user_token,
    get_api_key_user,
    verify_api_key,
    verify_token,
)
from mra_web.config import APIConfig, config
from mra_web.models import (
    AnalysisResponse,
    BaseRequest,
    ChartResponse,
    CurrentAnalysisRequest,
    DetailedAnalysisRequest,
    ErrorResponse,
    ExportCSVRequest,
    ExportResponse,
    GenerateChartsRequest,
    MonitoringMessage,
    MonitoringUpdate,
    MultiAnalysisResponse,
    MultiSymbolAnalysisRequest,
    PortfolioAnalysisResponse,
    PositionSizingRequest,
    PositionSizingResponse,
    ProviderInfo,
    ProvidersResponse,
)
from mra_web.utils import (
    APIMetrics,
    NumpyJSONEncoder,
    convert_numpy_types,
    create_error_response,
    get_regime_from_string,
    get_strategy_from_string,
    sanitize_log_data,
)

# ── Config tests ──


class TestAPIConfig:
    def test_defaults(self):
        c = APIConfig(jwt_secret="test-secret")
        assert c.host == "0.0.0.0"
        assert c.port == 8000
        assert c.environment == "development"
        assert c.jwt_algorithm == "HS256"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("JWT_SECRET", "my-secret")
        monkeypatch.setenv("ENVIRONMENT", "production")
        c = APIConfig.from_env()
        assert c.port == 9000
        assert c.jwt_secret == "my-secret"
        assert c.environment == "production"

    def test_global_config_exists(self):
        assert config is not None
        assert isinstance(config, APIConfig)


# ── Auth tests ──


class TestAuth:
    def test_create_access_token(self):
        token = create_access_token({"sub": "testuser"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_and_verify_token(self):
        token = create_access_token({"sub": "alice"})
        data = verify_token(token)
        assert data.username == "alice"

    def test_create_user_token(self):
        token = create_user_token("bob")
        data = verify_token(token)
        assert data.username == "bob"

    def test_verify_invalid_token(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            verify_token("invalid.token.here")
        assert exc_info.value.status_code == 401

    def test_token_with_expiry(self):
        token = create_access_token({"sub": "user"}, expires_delta=timedelta(hours=1))
        data = verify_token(token)
        assert data.exp is not None

    def test_verify_api_key_valid(self):
        assert verify_api_key("demo-api-key-12345") is True
        assert verify_api_key("admin-api-key-67890") is True

    def test_verify_api_key_invalid(self):
        assert verify_api_key("invalid-key") is False

    def test_get_api_key_user(self):
        assert get_api_key_user("demo-api-key-12345") == "demo"
        assert get_api_key_user("admin-api-key-67890") == "admin"
        assert get_api_key_user("nonexistent") is None

    def test_token_data_model(self):
        td = TokenData(username="test")
        assert td.username == "test"
        assert td.exp is None

    def test_user_model(self):
        u = User(username="test", email="test@example.com")
        assert u.is_active is True

    def test_auth_config_api_keys(self):
        assert "demo" in AuthConfig.API_KEYS
        assert "admin" in AuthConfig.API_KEYS


# ── Models tests ──


class TestBaseRequest:
    def test_default_provider(self):
        req = BaseRequest()
        assert req.provider == "alphavantage"

    def test_valid_provider(self):
        req = BaseRequest(provider="yfinance")
        assert req.provider == "yfinance"

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            BaseRequest(provider="invalid")


class TestDetailedAnalysisRequest:
    def test_valid(self):
        req = DetailedAnalysisRequest(symbol="spy", timeframe="1D")
        assert req.symbol == "SPY"
        assert req.timeframe == "1D"

    def test_empty_symbol_raises(self):
        with pytest.raises(ValidationError):
            DetailedAnalysisRequest(symbol="", timeframe="1D")

    def test_invalid_timeframe_raises(self):
        with pytest.raises(ValidationError):
            DetailedAnalysisRequest(symbol="SPY", timeframe="5m")


class TestCurrentAnalysisRequest:
    def test_symbol_uppercased(self):
        req = CurrentAnalysisRequest(symbol="aapl")
        assert req.symbol == "AAPL"


class TestMultiSymbolAnalysisRequest:
    def test_valid(self):
        req = MultiSymbolAnalysisRequest(symbols=["spy", "qqq"], timeframe="1D")
        assert req.symbols == ["SPY", "QQQ"]

    def test_empty_symbols_raises(self):
        with pytest.raises(ValidationError):
            MultiSymbolAnalysisRequest(symbols=[], timeframe="1D")


class TestGenerateChartsRequest:
    def test_valid(self):
        req = GenerateChartsRequest(symbol="SPY", timeframe="1D", days=30)
        assert req.days == 30

    def test_zero_days_raises(self):
        with pytest.raises(ValidationError):
            GenerateChartsRequest(symbol="SPY", timeframe="1D", days=0)

    def test_too_many_days_raises(self):
        with pytest.raises(ValidationError):
            GenerateChartsRequest(symbol="SPY", timeframe="1D", days=400)


class TestPositionSizingRequest:
    def test_valid(self):
        req = PositionSizingRequest(
            base_size=0.05,
            regime="Bull Trending",
            confidence=0.8,
            persistence=0.7,
        )
        assert req.base_size == 0.05

    def test_invalid_base_size(self):
        with pytest.raises(ValidationError):
            PositionSizingRequest(
                base_size=1.5,
                regime="Bull Trending",
                confidence=0.8,
                persistence=0.7,
            )

    def test_invalid_regime(self):
        with pytest.raises(ValidationError):
            PositionSizingRequest(
                base_size=0.05,
                regime="Invalid",
                confidence=0.8,
                persistence=0.7,
            )

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            PositionSizingRequest(
                base_size=0.05,
                regime="Bull Trending",
                confidence=1.5,
                persistence=0.7,
            )

    def test_invalid_correlation(self):
        with pytest.raises(ValidationError):
            PositionSizingRequest(
                base_size=0.05,
                regime="Bull Trending",
                confidence=0.8,
                persistence=0.7,
                correlation=2.0,
            )


class TestExportCSVRequest:
    def test_valid(self):
        req = ExportCSVRequest(symbol="spy")
        assert req.symbol == "SPY"
        assert req.filename is None


class TestResponseModels:
    def test_analysis_response(self):
        resp = AnalysisResponse(
            symbol="SPY",
            timeframe="1D",
            current_regime="Bull Trending",
            regime_confidence=0.9,
            regime_persistence=0.8,
            transition_probability=0.5,
            hmm_state=0,
            risk_level="Low",
            position_sizing_multiplier=0.1,
            recommended_strategy="Trend Following",
        )
        assert resp.symbol == "SPY"

    def test_error_response(self):
        resp = ErrorResponse(error_code="TEST", message="test error")
        assert resp.error_code == "TEST"

    def test_position_sizing_response(self):
        resp = PositionSizingResponse(
            base_size=0.02,
            regime="Bull Trending",
            regime_adjusted_size=0.03,
            correlation_adjusted_size=0.025,
            final_recommendation=0.025,
        )
        assert resp.final_recommendation == 0.025

    def test_provider_info(self):
        pi = ProviderInfo(
            name="yfinance",
            description="Yahoo Finance",
            requires_api_key=False,
            rate_limit_per_minute=100,
            supported_intervals=["1d"],
            supported_periods=["1y"],
        )
        assert pi.name == "yfinance"

    def test_chart_response(self):
        resp = ChartResponse(symbol="SPY", timeframe="1D", days=60)
        assert resp.chart_data is None

    def test_export_response(self):
        resp = ExportResponse(
            symbol="SPY",
            file_path="/tmp/test.csv",
            filename="test.csv",
            records_count=10,
        )
        assert resp.records_count == 10

    def test_monitoring_message(self):
        msg = MonitoringMessage(
            message_type="update",
            symbol="SPY",
        )
        assert msg.message_type == "update"

    def test_monitoring_update(self):
        upd = MonitoringUpdate(
            symbol="SPY",
            current_regime="Bull Trending",
            regime_confidence=0.9,
            regime_change=False,
            alert_level="low",
        )
        assert upd.regime_change is False

    def test_multi_analysis_response(self):
        resp = MultiAnalysisResponse(
            symbol="SPY",
            analyses=[],
        )
        assert resp.symbol == "SPY"

    def test_portfolio_analysis_response(self):
        resp = PortfolioAnalysisResponse(
            symbols=["SPY", "QQQ"],
            timeframe="1D",
            analyses=[],
        )
        assert len(resp.symbols) == 2

    def test_providers_response(self):
        resp = ProvidersResponse(providers={})
        assert isinstance(resp.providers, dict)


# ── Utils tests ──


class TestNumpyJSONEncoder:
    def test_numpy_int(self):
        result = json.dumps({"val": np.int64(42)}, cls=NumpyJSONEncoder)
        assert "42" in result

    def test_numpy_float(self):
        result = json.dumps({"val": np.float64(3.14)}, cls=NumpyJSONEncoder)
        assert "3.14" in result

    def test_numpy_array(self):
        result = json.dumps({"val": np.array([1, 2, 3])}, cls=NumpyJSONEncoder)
        assert "[1, 2, 3]" in result

    def test_datetime(self):
        dt = datetime(2023, 1, 1, tzinfo=UTC)
        result = json.dumps({"val": dt}, cls=NumpyJSONEncoder)
        assert "2023" in result


class TestConvertNumpyTypes:
    def test_int(self):
        assert convert_numpy_types(np.int64(5)) == 5

    def test_float(self):
        assert convert_numpy_types(np.float64(1.5)) == 1.5

    def test_array(self):
        assert convert_numpy_types(np.array([1, 2])) == [1, 2]

    def test_dict(self):
        result = convert_numpy_types({"a": np.int64(1)})
        assert result == {"a": 1}

    def test_list(self):
        result = convert_numpy_types([np.float64(1.0)])
        assert result == [1.0]

    def test_tuple(self):
        result = convert_numpy_types((np.int64(1),))
        assert result == (1,)

    def test_passthrough(self):
        assert convert_numpy_types("hello") == "hello"


class TestCreateErrorResponse:
    def test_basic(self):
        resp = create_error_response("TEST", "test message")
        assert resp.error_code == "TEST"
        assert resp.message == "test message"

    def test_with_details(self):
        resp = create_error_response("ERR", "msg", {"key": "val"})
        assert resp.details == {"key": "val"}


class TestSanitizeLogData:
    def test_redacts_api_key(self):
        data = {"api_key": "secret123", "symbol": "SPY"}
        result = sanitize_log_data(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["symbol"] == "SPY"

    def test_redacts_password(self):
        data = {"password": "mypass"}
        result = sanitize_log_data(data)
        assert result["password"] == "***REDACTED***"

    def test_no_mutation(self):
        data = {"api_key": "secret"}
        sanitize_log_data(data)
        assert data["api_key"] == "secret"


class TestGetRegimeFromString:
    def test_valid(self):
        from mra_lib.config.enums import MarketRegime

        result = get_regime_from_string("Bull Trending")
        assert result == MarketRegime.BULL_TRENDING

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_regime_from_string("Nonexistent")


class TestGetStrategyFromString:
    def test_valid(self):
        from mra_lib.config.enums import TradingStrategy

        result = get_strategy_from_string("Trend Following")
        assert result == TradingStrategy.TREND_FOLLOWING

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_strategy_from_string("Nonexistent")


class TestAPIMetrics:
    def test_record_request(self):
        m = APIMetrics()
        m.record_request("/test")
        m.record_request("/test")
        assert m.request_counts["/test"] == 2

    def test_record_error(self):
        m = APIMetrics()
        m.record_error("/test")
        assert m.error_counts["/test"] == 1

    def test_record_response_time(self):
        m = APIMetrics()
        m.record_response_time("/test", 0.5)
        m.record_response_time("/test", 1.0)
        assert len(m.response_times["/test"]) == 2

    def test_get_metrics(self):
        m = APIMetrics()
        m.record_request("/a")
        m.record_request("/b")
        m.record_error("/a")
        m.record_response_time("/a", 0.1)
        metrics = m.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["total_errors"] == 1
        assert "uptime_seconds" in metrics
        assert "average_response_times" in metrics


# ── App tests ──


class TestApp:
    def test_app_exists(self):
        assert app is not None

    def test_app_title(self):
        assert app.title == "Market Regime Analysis API"

    def test_app_version(self):
        assert app.version == "1.0.0"


class TestAppEndpoints:
    """Test FastAPI endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_ready(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Market Regime Analysis API"

    def test_auth_token(self, client):
        resp = client.post("/auth/token?username=testuser")
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["username"] == "testuser"

    def test_auth_token_empty_username(self, client):
        resp = client.post("/auth/token?username=")
        assert resp.status_code == 400
