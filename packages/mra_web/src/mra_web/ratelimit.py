"""
Per-client rate limiting middleware.

slowapi's ``SlowAPIMiddleware`` resolves route handlers by walking ``app.routes``;
since FastAPI 0.140 included routers are nested (``_IncludedRouter``), so it finds no
handler and silently exempts every ``/api/v1`` route. This small ASGI middleware
limits by request path instead and needs no route lookup.

Limits are kept in process memory: with several workers each enforces its own window.
"""

import ipaddress
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mra_web.models import ErrorResponse

# Hard cap on tracked clients; beyond it the oldest windows are evicted.
MAX_TRACKED_CLIENTS = 50_000


class FixedWindowCounter:
    """Fixed-window request counter keyed by client."""

    def __init__(
        self, limit: int, window_seconds: float = 60.0, max_clients: int = MAX_TRACKED_CLIENTS
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self.max_clients = max_clients
        # Ordered by window start (a reset window moves to the end), so the
        # oldest entries are always first: expiry and eviction pop from the front.
        self._buckets: OrderedDict[str, tuple[float, int]] = OrderedDict()

    def hit(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Count a request. Returns (allowed, seconds until the window resets)."""
        now = time.monotonic() if now is None else now
        self._expire(now)
        start, count = self._buckets.get(key, (now, 0))
        if count == 0:
            start = now
        count += 1
        self._buckets[key] = (start, count)
        if count == 1:
            self._buckets.move_to_end(key)
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        return count <= self.limit, max(0.0, start + self.window - now)

    def _expire(self, now: float) -> None:
        """Drop windows that have ended (amortized O(1): oldest entries first)."""
        while self._buckets:
            key, (start, _) = next(iter(self._buckets.items()))
            if now - start < self.window:
                break
            del self._buckets[key]

    def __len__(self) -> int:
        return len(self._buckets)


def client_ip(scope: Scope) -> str:
    """Return the rate-limit key for the direct peer.

    IPv6 clients are keyed by their /64 prefix, since a single host typically
    controls a whole /64. Behind a proxy, configure uvicorn's proxy headers.
    """
    client = scope.get("client")
    if not client:
        return "unknown"
    host = client[0]
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is None:
        return str(ipaddress.IPv6Network(f"{address}/64", strict=False))
    return host


class RateLimitMiddleware:
    """Apply a per-client limit to every HTTP request except exempt paths."""

    def __init__(
        self,
        app: ASGIApp,
        per_minute: int,
        exempt_paths: Iterable[str] = (),
        key_func: Callable[[Scope], str] = client_ip,
    ) -> None:
        self.app = app
        self.counter = FixedWindowCounter(per_minute, 60.0)
        self.exempt_paths = frozenset(exempt_paths)
        self.key_func = key_func

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        allowed, reset_in = self.counter.hit(self.key_func(scope))
        if allowed:
            await self.app(scope, receive, send)
            return

        retry_after = max(1, math.ceil(reset_in))
        body = ErrorResponse(
            error_code="RATE_LIMITED",
            message=f"Rate limit exceeded: {self.counter.limit} per minute",
            details={"retry_after": retry_after},
        )
        response = JSONResponse(
            status_code=429,
            content=body.model_dump(mode="json"),
            headers={"Retry-After": str(retry_after)},
        )
        await response(scope, receive, send)
