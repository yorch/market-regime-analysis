"""
Credential lookup for data providers.

Shared by the CLI and web frontends so both resolve provider API keys from the
same environment variables.
"""

import os

# Environment variables checked (in order) when no API key is given explicitly.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "alphavantage": ["ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"],
    "polygon": ["POLYGON_API_KEY"],
    "tiingo": ["TIINGO_API_KEY"],
}

# Providers that need a key ID *and* secret; both variables must be set.
PROVIDER_ENV_PAIRS: dict[str, tuple[str, str]] = {
    "alpaca": ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
}


def required_env_vars(provider: str) -> list[str]:
    """Return the environment variables a provider's credentials can come from."""
    if provider in PROVIDER_ENV_PAIRS:
        return list(PROVIDER_ENV_PAIRS[provider])
    return PROVIDER_ENV_VARS.get(provider, [])


def requires_credentials(provider: str) -> bool:
    """Return True if the provider cannot be used without credentials."""
    return provider in PROVIDER_ENV_VARS or provider in PROVIDER_ENV_PAIRS


def resolve_api_key(provider: str, api_key: str | None) -> str | None:
    """
    Resolve a provider API key from an explicit value or the environment.

    Key-pair providers (Alpaca) resolve to a combined ``"KEY_ID:SECRET"`` string,
    which their provider class splits back apart.

    Args:
        provider: Provider name
        api_key: Explicit key (e.g. from ``--api-key``); returned as-is when given

    Returns:
        The key, ``""`` for providers that need none, or ``None`` if a required key
        is missing
    """
    if api_key:
        return api_key

    if provider in PROVIDER_ENV_PAIRS:
        key_var, secret_var = PROVIDER_ENV_PAIRS[provider]
        key_id, secret = os.getenv(key_var), os.getenv(secret_var)
        return f"{key_id}:{secret}" if key_id and secret else None

    if provider not in PROVIDER_ENV_VARS:
        return ""

    for env_var in PROVIDER_ENV_VARS[provider]:
        value = os.getenv(env_var)
        if value:
            return value
    return None
