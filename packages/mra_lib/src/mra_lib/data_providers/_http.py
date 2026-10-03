"""
HTTP helpers shared by REST-based data providers.
"""

import re
import time
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import requests

from .base import AuthError, InvalidSymbolError, ProviderConfig, RateLimitError

# Status codes worth retrying: rate limiting and transient server errors
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Base delay in seconds before the first retry; doubles after each attempt
_BACKOFF_SECONDS = 1.0

# Longest server-requested wait we honor; beyond this we fail fast instead of blocking
_MAX_RETRY_AFTER_SECONDS = 60.0

# Query parameters that carry credentials and must never appear in error messages
_SECRET_PARAM = re.compile(r"((?:api_?key|token|secret)=)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Mask credential query parameters (``apikey=...``, ``token=...``) in ``text``."""
    return _SECRET_PARAM.sub(r"\1***", text)


def get_json(  # noqa: PLR0913
    url: str,
    *,
    provider: str,
    config: ProviderConfig,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    throttle: Callable[[], None] | None = None,
) -> Any:
    """
    GET a JSON document, retrying on rate limits and transient server errors.

    Args:
        url: Request URL
        provider: Provider display name used in error messages
        config: Provider config supplying ``timeout`` (seconds) and ``retries``
        params: Query string parameters
        headers: Request headers (credentials belong here, never in ``params``
            when the provider supports header auth, so they stay out of logs)
        throttle: Called before every attempt (client-side rate limiter)

    Returns:
        Decoded JSON body

    Raises:
        AuthError: HTTP 401/403
        RateLimitError: HTTP 429 after retries, or a server-requested wait that is too long
        InvalidSymbolError: HTTP 404 (these endpoints put the symbol in the path)
        ConnectionError: On network failure or any other non-2xx response
    """
    retries = config.retries
    delay = _BACKOFF_SECONDS
    for attempt in range(retries + 1):
        if throttle is not None:
            throttle()
        try:
            response = requests.get(url, params=params, headers=headers, timeout=config.timeout)
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            # ``from None``: the chained exception's repr would carry the full URL
            raise ConnectionError(redact(f"{provider} request failed: {e}")) from None

        if response.status_code in _RETRYABLE_STATUS and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            if wait > _MAX_RETRY_AFTER_SECONDS:
                raise RateLimitError(
                    f"{provider} is rate limiting requests (retry after {wait:.0f}s)"
                )
            time.sleep(wait)
            delay *= 2
            continue

        if response.status_code in (401, 403):
            raise AuthError(
                f"{provider} rejected the credentials (HTTP {response.status_code}). "
                "Check your API key and plan entitlements."
            )
        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            raise RateLimitError(f"{provider} is rate limiting requests (HTTP 429)")
        if response.status_code == HTTPStatus.NOT_FOUND:
            raise InvalidSymbolError(
                redact(f"{provider} returned HTTP 404 (unknown symbol?): {response.text[:200]}")
            )
        if not response.ok:
            raise ConnectionError(
                redact(f"{provider} returned HTTP {response.status_code}: {response.text[:200]}")
            )

        try:
            return response.json()
        except ValueError as e:
            raise ConnectionError(f"{provider} returned invalid JSON: {e}") from e

    # Unreachable: the loop either returns or raises on the final attempt
    raise ConnectionError(f"{provider} request failed after {retries} retries")
