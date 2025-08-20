"""
WebSocket handlers for real-time monitoring in the Market Regime Analysis API.

This module provides WebSocket endpoints for continuous regime monitoring
and real-time updates.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from .. import MarketRegimeAnalyzer
from .models import MonitoringMessage, MonitoringUpdate
from .utils import validate_api_key

# Setup logging
logger = logging.getLogger(__name__)

# WebSocket router
ws_router = APIRouter()


# Active connections manager
class ConnectionManager:
    """Manages WebSocket connections for real-time monitoring."""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
        self.connection_data: dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, symbol: str, connection_data: dict):
        """Accept a new WebSocket connection."""
        await websocket.accept()

        if symbol not in self.active_connections:
            self.active_connections[symbol] = set()

        self.active_connections[symbol].add(websocket)
        self.connection_data[websocket] = {
            "symbol": symbol,
            "connected_at": datetime.utcnow(),
            **connection_data,
        }

        logger.info(
            f"New WebSocket connection for {symbol}: {len(self.active_connections[symbol])} total"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket not in self.connection_data:
            return

        symbol = self.connection_data[websocket]["symbol"]

        if symbol in self.active_connections:
            self.active_connections[symbol].discard(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

        del self.connection_data[websocket]
        logger.info(f"WebSocket disconnected for {symbol}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast_to_symbol(self, symbol: str, message: str):
        """Broadcast a message to all connections monitoring a specific symbol."""
        if symbol not in self.active_connections:
            return

        disconnected = set()

        for websocket in self.active_connections[symbol].copy():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to WebSocket: {e}")
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


@ws_router.websocket("/monitoring/{symbol}")
async def websocket_monitoring_endpoint(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time regime monitoring.

    Args:
        websocket: WebSocket connection
        symbol: Trading symbol to monitor
    """
    # Get query parameters
    query_params = dict(websocket.query_params)
    provider = query_params.get("provider", "alphavantage")
    api_key = query_params.get("api_key")
    interval = int(query_params.get("interval", "300"))  # Default 5 minutes

    # Validate parameters
    if not symbol or not symbol.strip():
        await websocket.close(code=1008, reason="Symbol is required")
        return

    symbol = symbol.strip().upper()

    # Validate interval
    if interval < 60 or interval > 3600:
        await websocket.close(code=1008, reason="Interval must be between 60 and 3600 seconds")
        return

    try:
        # Validate API key
        validated_api_key = validate_api_key(provider, api_key)

        # Accept connection
        connection_data = {
            "provider": provider,
            "api_key": validated_api_key[:10] + "..." if validated_api_key else None,
            "interval": interval,
        }
        await manager.connect(websocket, symbol, connection_data)

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
        await manager.send_personal_message(welcome_message.json(), websocket)

        # Start monitoring loop
        await monitoring_loop(websocket, symbol, provider, validated_api_key, interval)

    except ValueError as e:
        await websocket.close(code=1008, reason=f"Invalid input: {e!s}")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        manager.disconnect(websocket)


async def monitoring_loop(  # noqa: PLR0915
    websocket: WebSocket, symbol: str, provider: str, api_key: str, interval: int
):
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
                    message_type="update", symbol=symbol, data=update.dict()
                )

                await manager.send_personal_message(message.json(), websocket)

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
                    await manager.send_personal_message(alert_message.json(), websocket)

                # Update previous regime
                previous_regime = current_regime

                # Reset error count on successful analysis
                error_count = 0

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for {symbol}")
                break

            except Exception as e:
                error_count += 1
                logger.error(f"Monitoring error for {symbol}: {e}")

                # Send error message
                error_message = MonitoringMessage(
                    message_type="error",
                    symbol=symbol,
                    data={"error": str(e), "error_count": error_count, "max_errors": max_errors},
                )

                try:
                    await manager.send_personal_message(error_message.json(), websocket)
                except Exception:
                    break

                # Stop monitoring if too many errors
                if error_count >= max_errors:
                    logger.error(f"Too many errors for {symbol} monitoring, stopping")
                    break

            # Wait for next interval
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {symbol} monitoring")
    except Exception as e:
        logger.error(f"Monitoring loop error for {symbol}: {e}")
    finally:
        manager.disconnect(websocket)


@ws_router.get("/monitoring/status")
async def monitoring_status():
    """Get status of all active monitoring connections."""
    return {
        "active_connections": manager.get_connection_count(),
        "monitored_symbols": manager.get_active_symbols(),
        "connections_by_symbol": {
            symbol: manager.get_connection_count(symbol) for symbol in manager.get_active_symbols()
        },
    }


# WebSocket testing endpoint
@ws_router.websocket("/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """Simple WebSocket test endpoint."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = f"Echo: {data} at {datetime.utcnow().isoformat()}"
            await websocket.send_text(message)
    except WebSocketDisconnect:
        logger.info("Test WebSocket disconnected")
