"""
HTTP helpers shared by REST-based data providers.
"""

import time
from typing import Any

import requests

from .base import ProviderConfig

# Status codes worth retrying: rate limiting and transient server errors
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Base delay in seconds before the first retry; doubles after each attempt
_BACKOFF_SECONDS = 1.0

# Longest server-requested wait we honor; beyond this we fail fast instead of blocking
_MAX_RETRY_AFTER_SECONDS = 60.0


def get_json(
    url: str,
    *,
    provider: str,
    config: ProviderConfig,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
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

    Returns:
        Decoded JSON body

    Raises:
        ConnectionError: On network failure, auth failure, or non-2xx response
    """
    retries = config.retries
    delay = _BACKOFF_SECONDS
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=config.timeout)
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise ConnectionError(f"{provider} request failed: {e}") from e

        if response.status_code in _RETRYABLE_STATUS and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            if wait > _MAX_RETRY_AFTER_SECONDS:
                raise ConnectionError(
                    f"{provider} is rate limiting requests (retry after {wait:.0f}s)"
                )
            time.sleep(wait)
            delay *= 2
            continue

        if response.status_code in (401, 403):
            raise ConnectionError(
                f"{provider} rejected the credentials (HTTP {response.status_code}). "
                "Check your API key and plan entitlements."
            )
        if not response.ok:
            raise ConnectionError(
                f"{provider} returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()
        except ValueError as e:
            raise ConnectionError(f"{provider} returned invalid JSON: {e}") from e

    # Unreachable: the loop either returns or raises on the final attempt
    raise ConnectionError(f"{provider} request failed after {retries} retries")
