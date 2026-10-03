"""
WebSocket handlers for real-time monitoring in the Market Regime Analysis API.

Connections must authenticate. Credentials are checked before the handshake is
accepted, from (in order) the ``X-API-Key`` header, an ``Authorization: Bearer``
header, or a ``token`` query parameter (JWT or API key). Browsers that cannot set
headers and do not want the token in the URL may instead send
``{"token": "<JWT or API key>"}`` as the first message within
``AUTH_MESSAGE_TIMEOUT`` seconds. Browser ``Origin`` headers must be in
``CORS_ORIGINS``, and concurrent connections are capped in total and per client IP.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.routing import APIRouter

from mra_lib import MarketRegimeAnalyzer

from .auth import (
    User,
    authenticate_credentials,
    authenticate_request,
    get_api_key_user,
    get_app_config,
)
from .config import APIConfig
from .models import MonitoringMessage, MonitoringUpdate, normalize_symbol
from .utils import validate_api_key

# Setup logging
logger = logging.getLogger(__name__)

# WebSocket router
ws_router = APIRouter()

# Close codes (RFC 6455 / IANA registry)
WS_POLICY_VIOLATION = status.WS_1008_POLICY_VIOLATION
WS_TRY_AGAIN_LATER = status.WS_1013_TRY_AGAIN_LATER
WS_INTERNAL_ERROR = status.WS_1011_INTERNAL_ERROR

AUTH_MESSAGE_TIMEOUT = 10.0


def _client_ip(websocket: WebSocket) -> str:
    return websocket.client.host if websocket.client else "unknown"


# Active connections manager
class ConnectionManager:
    """Manages WebSocket connections and enforces connection caps."""

    def __init__(self) -> None:
        self.active_connections: dict[str, set[WebSocket]] = {}
        self.connection_data: dict[WebSocket, dict] = {}
        self.connections_per_ip: dict[str, int] = {}
        self.reserved = 0

    def try_reserve(self, client_ip: str, max_total: int, max_per_ip: int) -> bool:
        """Reserve a connection slot; False if a cap is reached.

        Runs without awaiting, so check-and-increment is atomic on the event loop.
        """
        if self.reserved >= max_total:
            return False
        if self.connections_per_ip.get(client_ip, 0) >= max_per_ip:
            return False
        self.reserved += 1
        self.connections_per_ip[client_ip] = self.connections_per_ip.get(client_ip, 0) + 1
        return True

    def release(self, client_ip: str) -> None:
        """Release a slot taken by :meth:`try_reserve`."""
        self.reserved = max(0, self.reserved - 1)
        remaining = self.connections_per_ip.get(client_ip, 0) - 1
        if remaining > 0:
            self.connections_per_ip[client_ip] = remaining
        else:
            self.connections_per_ip.pop(client_ip, None)

    def register(self, websocket: WebSocket, symbol: str, connection_data: dict) -> None:
        """Track an accepted connection for a symbol."""
        self.active_connections.setdefault(symbol, set()).add(websocket)
        self.connection_data[websocket] = {
            "symbol": symbol,
            "connected_at": datetime.now(UTC),
            **connection_data,
        }
        logger.info(
            "New WebSocket connection for %s: %d total",
            symbol,
            len(self.active_connections[symbol]),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket not in self.connection_data:
            return

        symbol = self.connection_data[websocket]["symbol"]

        if symbol in self.active_connections:
            self.active_connections[symbol].discard(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

        del self.connection_data[websocket]
        logger.info("WebSocket disconnected for %s", symbol)

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception:
            logger.info("Failed to send message to WebSocket", exc_info=True)
            self.disconnect(websocket)

    async def broadcast_to_symbol(self, symbol: str, message: str) -> None:
        """Broadcast a message to all connections monitoring a specific symbol."""
        if symbol not in self.active_connections:
            return

        disconnected = set()

        for websocket in self.active_connections[symbol].copy():
            try:
                await websocket.send_text(message)
            except Exception:
                logger.info("Failed to broadcast to WebSocket", exc_info=True)
                disconnected.add(websocket)

        # Clean up disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)

    def get_connection_count(self, symbol: str | None = None) -> int:
        """Get the number of active connections."""
        if symbol:
            return len(self.active_connections.get(symbol, set()))
        return sum(len(connections) for connections in self.active_connections.values())

    def get_active_symbols(self) -> list[str]:
        """Get list of symbols with active connections."""
        return list(self.active_connections.keys())


# Global connection manager
manager = ConnectionManager()


def origin_allowed(origin: str | None, cfg: APIConfig) -> bool:
    """Return True if a browser Origin may open a WebSocket.

    Non-browser clients send no Origin and are allowed (they still need credentials).
    """
    if origin is None:
        return True
    return "*" in cfg.cors_origins or origin in cfg.cors_origins


def handshake_credentials(websocket: WebSocket) -> tuple[str | None, str | None]:
    """Extract (bearer_token, api_key) from handshake headers or the ``token`` param."""
    api_key = websocket.headers.get("x-api-key") or None
    bearer = None
    authorization = websocket.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        bearer = value.strip()
    if not api_key and not bearer:
        bearer = websocket.query_params.get("token") or None
    return bearer, api_key


def authenticate_token(cfg: APIConfig, token: str) -> User:
    """Authenticate a single opaque token that may be an API key or a JWT."""
    username = get_api_key_user(token, cfg)  # constant-time comparison
    if username is not None:
        return User(username=username, is_active=True)
    return authenticate_credentials(cfg, bearer_token=token)


def _authenticate_handshake(websocket: WebSocket, cfg: APIConfig) -> User | None:
    """Authenticate from handshake credentials.

    Returns None when no credentials were presented outside development (the
    client may still authenticate with a first message).

    Raises:
        HTTPException: If presented credentials are invalid.
    """
    bearer, api_key = handshake_credentials(websocket)
    if api_key:
        return authenticate_credentials(cfg, api_key=api_key)
    if bearer:
        return authenticate_token(cfg, bearer)
    if cfg.is_development:
        return authenticate_credentials(cfg)
    return None


async def _authenticate_first_message(websocket: WebSocket, cfg: APIConfig) -> User | None:
    """Wait for ``{"token": ...}`` as the first message; None if invalid or late."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_MESSAGE_TIMEOUT)
        payload = json.loads(raw)
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            return None
        return authenticate_token(cfg, token)
    except (TimeoutError, ValueError, HTTPException):
        return None


class _Rejected(Exception):
    """Pre-accept validation failure; the handshake is refused with ``code``."""

    def __init__(self, code: int = WS_POLICY_VIOLATION) -> None:
        super().__init__(code)
        self.code = code


def _check_handshake(
    websocket: WebSocket, symbol: str, cfg: APIConfig
) -> tuple[User | None, str, str, str | None, int]:
    """Validate origin, credentials and parameters before accepting.

    Returns:
        (user or None if first-message auth is needed, symbol, provider,
        provider API key, interval)

    Raises:
        _Rejected: If the handshake must be refused.
    """
    # Origin check (cross-site WebSocket hijacking protection)
    if not origin_allowed(websocket.headers.get("origin"), cfg):
        logger.warning("Rejected WebSocket from disallowed origin")
        raise _Rejected()

    try:
        user = _authenticate_handshake(websocket, cfg)
        symbol = normalize_symbol(symbol)
        interval = int(websocket.query_params.get("interval", "300"))  # Default 5 minutes
    except (HTTPException, ValueError) as e:
        raise _Rejected() from e
    if interval < 60 or interval > 3600:
        raise _Rejected()

    provider = websocket.query_params.get("provider", "alphavantage")
    provider_api_key = websocket.query_params.get("api_key")
    return user, symbol, provider, provider_api_key, interval


@ws_router.websocket("/monitoring/{symbol}")
async def websocket_monitoring_endpoint(websocket: WebSocket, symbol: str) -> None:
    """
    WebSocket endpoint for real-time regime monitoring.

    Query parameters: ``provider``, ``api_key`` (data provider key), ``interval``
    (seconds, 60-3600) and optionally ``token`` (API credential).
    """
    cfg = get_app_config(websocket)
    try:
        user, symbol, provider, provider_api_key, interval = _check_handshake(
            websocket, symbol, cfg
        )
    except _Rejected as rejected:
        await websocket.close(code=rejected.code)
        return

    # Connection caps
    client_ip = _client_ip(websocket)
    if not manager.try_reserve(client_ip, cfg.ws_max_connections, cfg.ws_max_connections_per_ip):
        logger.warning("Rejected WebSocket: connection limit reached")
        await websocket.close(code=WS_TRY_AGAIN_LATER)
        return

    try:
        await websocket.accept()

        if user is None:
            user = await _authenticate_first_message(websocket, cfg)
            if user is None:
                await websocket.close(code=WS_POLICY_VIOLATION, reason="Authentication required")
                return

        try:
            validated_api_key = validate_api_key(provider, provider_api_key)
        except HTTPException:
            await websocket.close(code=WS_POLICY_VIOLATION, reason="Provider API key required")
            return

        manager.register(
            websocket,
            symbol,
            {"provider": provider, "interval": interval, "user": user.username},
        )

        # Send initial connection confirmation
        welcome_message = MonitoringMessage(
            message_type="connection",
            symbol=symbol,
            data={
                "status": "connected",
                "provider": provider,
                "interval": interval,
                "message": f"Started monitoring {symbol} with {interval}s intervals",
            },
        )
        await manager.send_personal_message(welcome_message.model_dump_json(), websocket)

        # Start monitoring loop
        await monitoring_loop(websocket, symbol, provider, validated_api_key, interval)

    except WebSocketDisconnect:
        logger.info("Client disconnected from %s monitoring", symbol)
    except Exception:
        logger.exception("WebSocket connection error")
        try:
            await websocket.close(code=WS_INTERNAL_ERROR, reason="Internal server error")
        except Exception:
            logger.debug("WebSocket already closed", exc_info=True)
    finally:
        manager.disconnect(websocket)
        manager.release(client_ip)


async def monitoring_loop(  # noqa: PLR0912, PLR0915
    websocket: WebSocket, symbol: str, provider: str, api_key: str, interval: int
) -> None:
    """
    Main monitoring loop for WebSocket connections.

    Args:
        websocket: WebSocket connection
        symbol: Trading symbol to monitor
        provider: Data provider
        api_key: Provider API key
        interval: Update interval in seconds
    """
    previous_regime = None
    error_count = 0
    max_errors = 5

    try:
        while True:
            try:
                # Create analyzer
                analyzer = MarketRegimeAnalyzer(
                    symbol=symbol, provider_flag=provider, api_key=api_key
                )

                # Analyze current regime (using 1D timeframe for monitoring)
                analysis = analyzer.analyze_current_regime("1D")

                # Check for regime change
                current_regime = analysis.current_regime.value
                regime_changed = previous_regime is not None and current_regime != previous_regime

                # Determine alert level
                alert_level = "low"
                if regime_changed:
                    alert_level = "high"
                elif analysis.regime_confidence < 0.6:
                    alert_level = "medium"

                # Create monitoring update
                update = MonitoringUpdate(
                    symbol=symbol,
                    current_regime=current_regime,
                    regime_confidence=analysis.regime_confidence,
                    regime_change=regime_changed,
                    previous_regime=previous_regime,
                    alert_level=alert_level,
                )

                # Send update message
                message = MonitoringMessage(
                    message_type="update", symbol=symbol, data=update.model_dump(mode="json")
                )

                await manager.send_personal_message(message.model_dump_json(), websocket)

                # Send alert if regime changed
                if regime_changed:
                    alert_message = MonitoringMessage(
                        message_type="alert",
                        symbol=symbol,
                        data={
                            "alert_type": "regime_change",
                            "previous_regime": previous_regime,
                            "new_regime": current_regime,
                            "confidence": analysis.regime_confidence,
                            "message": f"Regime changed from {previous_regime} to {current_regime}",
                        },
                    )
                    await manager.send_personal_message(alert_message.model_dump_json(), websocket)

                # Update previous regime
                previous_regime = current_regime

                # Reset error count on successful analysis
                error_count = 0

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected for %s", symbol)
                break

            except Exception:
                error_count += 1
                logger.exception("Monitoring error for %s", symbol)

                # Send a generic error message (never exception text: it can carry keys)
                error_message = MonitoringMessage(
                    message_type="error",
                    symbol=symbol,
                    data={
                        "error": "Analysis failed",
                        "error_count": error_count,
                        "max_errors": max_errors,
                    },
                )

                try:
                    await manager.send_personal_message(error_message.model_dump_json(), websocket)
                except Exception:
                    break

                # Stop monitoring if too many errors
                if error_count >= max_errors:
                    logger.error("Too many errors for %s monitoring, stopping", symbol)
                    break

            # Stop if the client went away (send failure unregisters the socket)
            if websocket not in manager.connection_data:
                break

            # Wait for next interval
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from %s monitoring", symbol)
    except Exception:
        logger.exception("Monitoring loop error for %s", symbol)
    finally:
        manager.disconnect(websocket)


@ws_router.get("/monitoring/status")
async def monitoring_status(
    current_user: User = Depends(authenticate_request),  # noqa: B008
) -> dict:
    """Get status of all active monitoring connections (authenticated)."""
    return {
        "active_connections": manager.get_connection_count(),
        "monitored_symbols": manager.get_active_symbols(),
        "connections_by_symbol": {
            symbol: manager.get_connection_count(symbol) for symbol in manager.get_active_symbols()
        },
    }
