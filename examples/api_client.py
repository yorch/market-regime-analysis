#!/usr/bin/env python3
"""
API client examples consolidated in a single script.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import requests
import websockets

from examples.common import Banner

banner = Banner(50)


class MarketRegimeAPIClient:
    """Minimal client for the Market Regime Analysis API.

    Args:
        base_url: Server URL
        api_key: Data provider API key forwarded in request bodies (optional)
        token: JWT for ``Authorization: Bearer`` (mint with ``uv run mra-token --sub <name>``)
        service_key: Server API key sent as ``X-API-Key`` (one of the server's ``API_KEYS``)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        token: str | None = None,
        service_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.token = token
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        if service_key:
            self.session.headers.update({"X-API-Key": service_key})

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        res = self.session.post(f"{self.base_url}{path}", json=payload)
        res.raise_for_status()
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = self.session.get(f"{self.base_url}{path}")
        res.raise_for_status()
        return res.json()

    # REST endpoints
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def metrics(self) -> dict[str, Any]:
        return self._get("/metrics")

    def list_providers(self) -> dict[str, Any]:
        return self._get("/api/v1/providers")

    def detailed_analysis(
        self, symbol: str, timeframe: str, provider: str = "yfinance"
    ) -> dict[str, Any]:
        return self._post(
            "/api/v1/analysis/detailed",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "provider": provider,
                "api_key": self.api_key,
            },
        )

    def current_analysis(self, symbol: str, provider: str = "yfinance") -> dict[str, Any]:
        return self._post(
            "/api/v1/analysis/current",
            {"symbol": symbol, "provider": provider, "api_key": self.api_key},
        )

    def multi_symbol_analysis(
        self, symbols: list[str], timeframe: str, provider: str = "yfinance"
    ) -> dict[str, Any]:
        return self._post(
            "/api/v1/analysis/multi-symbol",
            {
                "symbols": symbols,
                "timeframe": timeframe,
                "provider": provider,
                "api_key": self.api_key,
            },
        )

    def position_sizing(
        self,
        base_size: float,
        regime: str,
        confidence: float,
        persistence: float,
        correlation: float = 0.0,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v1/position-sizing",
            {
                "base_size": base_size,
                "regime": regime,
                "confidence": confidence,
                "persistence": persistence,
                "correlation": correlation,
            },
        )

    def generate_charts(
        self, symbol: str, timeframe: str, days: int = 60, provider: str = "yfinance"
    ) -> bytes:
        """Return the regime chart as PNG bytes."""
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "provider": provider,
            "api_key": self.api_key,
        }
        res = self.session.post(f"{self.base_url}/api/v1/charts/generate", json=payload)
        res.raise_for_status()
        return res.content

    def export_csv(
        self, symbol: str, provider: str = "yfinance", filename: str | None = None
    ) -> str:
        """Return the analysis export as CSV text."""
        payload: dict[str, Any] = {"symbol": symbol, "provider": provider, "api_key": self.api_key}
        if filename:
            payload["filename"] = filename
        res = self.session.post(f"{self.base_url}/api/v1/export/csv", json=payload)
        res.raise_for_status()
        return res.text


async def websocket_monitor(symbol: str = "SPY", provider: str = "yfinance") -> None:
    uri = f"ws://localhost:8000/ws/monitoring/{symbol}?provider={provider}&interval=60"
    try:
        async with websockets.connect(uri) as ws:  # type: ignore[call-arg]
            # Authenticate with the first message (keeps the token out of the URL).
            # Not needed when the server runs with ENVIRONMENT=development.
            token = os.getenv("MRA_TOKEN") or os.getenv("MRA_API_KEY")
            if token:
                await ws.send(json.dumps({"token": token}))
            print(f"Connected to WebSocket: {symbol}")
            timeout = time.time() + 30  # 30 seconds demo
            while time.time() < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)
                    print(f"{data['message_type']} @ {data['timestamp']}")
                except TimeoutError:
                    print("No message in 10s...")
                    continue
    except Exception as e:
        print(f"WebSocket error: {e}")


def main() -> None:  # noqa: PLR0915 demo script
    print("🚀 Market Regime Analysis - API Examples")
    print("=" * 50)

    # Credentials: a JWT (MRA_TOKEN, from `uv run mra-token --sub demo`) or a server
    # API key (MRA_API_KEY). Neither is needed against a development server.
    client = MarketRegimeAPIClient(
        token=os.getenv("MRA_TOKEN"), service_key=os.getenv("MRA_API_KEY")
    )

    banner.title("1) Health Check")
    try:
        health = client.health()
        print(f"Status: {health['status']}  Version: {health['version']}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    banner.title("2) Authentication")
    if client.token or "X-API-Key" in client.session.headers:
        print("Using credentials from MRA_TOKEN / MRA_API_KEY")
    else:
        print("No credentials set; only works against a development server")

    banner.title("3) Providers")
    try:
        providers = client.list_providers()
        for n, info in providers.get("providers", {}).items():
            print(f"• {n}: {info['description']}")
    except Exception as e:
        print(f"Providers error: {e}")

    banner.title("4) Detailed Analysis")
    try:
        a = client.detailed_analysis("SPY", "1D")
        print(f"Regime: {a['current_regime']}  Confidence: {a['regime_confidence']:.3f}")
    except Exception as e:
        print(f"Detailed error: {e}")

    banner.title("5) Current Analysis (All TF)")
    try:
        c = client.current_analysis("QQQ")
        for an in c.get("analyses", []):
            print(f"{an['timeframe']}: {an['current_regime']} ({an['regime_confidence']:.3f})")
    except Exception as e:
        print(f"Current error: {e}")

    banner.title("6) Position Sizing")
    try:
        s = client.position_sizing(0.02, "Bull Trending", 0.8, 0.75, 0.1)
        print(f"Final Recommendation: {s['final_recommendation']:.1%}")
    except Exception as e:
        print(f"Sizing error: {e}")

    banner.title("7) Portfolio Analysis")
    try:
        p = client.multi_symbol_analysis(["SPY", "QQQ", "IWM"], "1D")
        print(f"Dominant Regime: {p['portfolio_metrics']['dominant_regime']}")
    except Exception as e:
        print(f"Portfolio error: {e}")

    banner.title("8) Metrics")
    try:
        m = client.metrics()
        print(f"Requests: {m['total_requests']} Errors: {m['total_errors']}")
    except Exception as e:
        print(f"Metrics error: {e}")

    print(
        "\nAPI examples completed. For WebSocket demo run: \n  uv run examples/api_client.py websocket"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "websocket":
        asyncio.run(websocket_monitor("SPY"))
    else:
        main()
