"""
Per-client rate limiting middleware.

slowapi's ``SlowAPIMiddleware`` resolves route handlers by walking ``app.routes``;
since FastAPI 0.140 included routers are nested (``_IncludedRouter``), so it finds no
handler and silently exempts every ``/api/v1`` route. This small ASGI middleware
limits by request path instead and needs no route lookup.

Limits are kept in process memory: with several workers each enforces its own window.
"""

import math
import time
from collections.abc import Callable, Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mra_web.models import ErrorResponse

# Prune expired buckets once the table grows past this many clients.
_PRUNE_THRESHOLD = 10_000


class FixedWindowCounter:
    """Fixed-window request counter keyed by client."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._buckets: dict[str, tuple[float, int]] = {}

    def hit(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Count a request. Returns (allowed, seconds until the window resets)."""
        now = time.monotonic() if now is None else now
        start, count = self._buckets.get(key, (now, 0))
        if now - start >= self.window:
            start, count = now, 0
        count += 1
        self._buckets[key] = (start, count)
        if len(self._buckets) > _PRUNE_THRESHOLD:
            self._prune(now)
        return count <= self.limit, max(0.0, start + self.window - now)

    def _prune(self, now: float) -> None:
        expired = [k for k, (start, _) in self._buckets.items() if now - start >= self.window]
        for key in expired:
            del self._buckets[key]


def client_ip(scope: Scope) -> str:
    """Return the direct peer address (configure the proxy to set it correctly)."""
    client = scope.get("client")
    return client[0] if client else "unknown"


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
