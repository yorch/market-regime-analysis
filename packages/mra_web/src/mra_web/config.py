"""
Configuration management for the Market Regime Analysis API server.

Settings are read from environment variables. The server fails closed: outside
``ENVIRONMENT=development`` it refuses to start without a strong ``JWT_SECRET``.

Environment variables
---------------------
ENVIRONMENT            ``production`` (default) or ``development``
JWT_SECRET             HS256 signing secret, at least 32 characters (required outside development)
JWT_EXPIRATION_HOURS   Lifetime of minted tokens (default 24)
API_KEYS               Comma-separated static API keys accepted in the ``X-API-Key`` header
CORS_ORIGINS           Comma-separated allowed browser origins (default: none)
RATE_LIMIT_PER_MINUTE  Default per-client rate limit for all API routes (default 60)
API_HOST / API_PORT    Bind address (default 127.0.0.1:8000)
ENABLE_DOCS            Serve /docs, /redoc and /openapi.json (default: true only in development)
"""

import logging
import os
import secrets
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

DEVELOPMENT = "development"
PRODUCTION = "production"

# The only signing algorithm accepted for JWTs. Not configurable on purpose:
# letting the environment choose the algorithm invites alg-confusion attacks.
JWT_ALGORITHM = "HS256"

MIN_JWT_SECRET_LENGTH = 32
MIN_API_KEY_LENGTH = 16

# Well-known placeholder secrets that must never be used to sign tokens.
PLACEHOLDER_SECRETS = frozenset(
    {
        "your-secret-key-change-in-production",
        "your-super-secure-secret-key",
        "change-me",
        "changeme",
        "secret",
    }
)


class ConfigError(RuntimeError):
    """Raised when the server configuration is unsafe or invalid."""


def _split_csv(value: str | None) -> list[str]:
    """Split a comma-separated env value, dropping blanks."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _weak_secret_reason(secret: str) -> str | None:
    """Return why a JWT secret is unacceptable, or None if it is fine."""
    if not secret:
        return "unset or empty"
    if secret.strip().lower() in PLACEHOLDER_SECRETS:
        return "a well-known placeholder value"
    if len(secret) < MIN_JWT_SECRET_LENGTH:
        return f"shorter than {MIN_JWT_SECRET_LENGTH} characters"
    return None


class APIConfig(BaseModel):
    """API server configuration settings."""

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    reload: bool = Field(default=False, description="Enable hot reload for development")

    # Authentication settings
    jwt_secret: str = Field(default="", description="JWT secret key", repr=False)
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")
    jwt_secret_ephemeral: bool = Field(
        default=False,
        description="True when the secret was generated for this process (development only)",
    )
    api_keys: list[str] = Field(
        default_factory=list, description="Static API keys (X-API-Key)", repr=False
    )

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute per client")
    rate_limit_burst: int = Field(default=10, description="Burst limit for rate limiting")

    # CORS settings (no origins allowed unless configured)
    cors_origins: list[str] = Field(default_factory=list, description="Allowed CORS origins")
    cors_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "OPTIONS"], description="Allowed CORS methods"
    )
    cors_headers: list[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type", "X-API-Key"],
        description="Allowed CORS headers",
    )

    # WebSocket limits
    ws_max_connections: int = Field(default=100, description="Max concurrent WebSockets")
    ws_max_connections_per_ip: int = Field(default=5, description="Max WebSockets per client IP")

    # Blocking analyses running at once (HTTP and WebSocket); excess requests get 503
    max_concurrent_analyses: int = Field(
        default=4, ge=1, description="Max concurrent analysis worker threads"
    )

    # Environment
    environment: str = Field(default=PRODUCTION, description="Environment (development/production)")
    debug: bool = Field(default=False, description="Debug mode")
    enable_docs: bool | None = Field(
        default=None, description="Serve OpenAPI docs (default: development only)"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if isinstance(data.get("environment"), str):
                data["environment"] = data["environment"].strip().lower() or PRODUCTION
            if isinstance(data.get("log_level"), str):
                data["log_level"] = data["log_level"].strip().upper() or "INFO"
        return data

    @model_validator(mode="after")
    def _enforce_secrets(self) -> "APIConfig":
        reason = _weak_secret_reason(self.jwt_secret)
        if reason is not None:
            if not self.is_development:
                raise ConfigError(
                    f"JWT_SECRET is {reason}. Set JWT_SECRET to a random value of at least "
                    f"{MIN_JWT_SECRET_LENGTH} characters, e.g. "
                    "`python -c 'import secrets; print(secrets.token_urlsafe(48))'`, "
                    "or set ENVIRONMENT=development for local use."
                )
            logger.warning(
                "!!! JWT_SECRET is %s; generated a random per-process secret because "
                "ENVIRONMENT=development. Tokens will not survive a restart and will not be "
                "accepted by other workers. Never run like this in production. !!!",
                reason,
            )
            self.jwt_secret = secrets.token_urlsafe(48)
            self.jwt_secret_ephemeral = True

        short_keys = [k for k in self.api_keys if len(k) < MIN_API_KEY_LENGTH]
        if short_keys:
            if not self.is_development:
                raise ConfigError(
                    f"API_KEYS contains {len(short_keys)} key(s) shorter than "
                    f"{MIN_API_KEY_LENGTH} characters; use long random keys."
                )
            logger.warning("API_KEYS contains short keys; acceptable only in development.")

        if "*" in self.cors_origins:
            logger.warning(
                "CORS_ORIGINS contains '*': any site may call the API from a browser "
                "(credentials are never allowed with a wildcard)."
            )
        return self

    @property
    def is_development(self) -> bool:
        """True when running with ENVIRONMENT=development."""
        return self.environment == DEVELOPMENT

    @property
    def docs_enabled(self) -> bool:
        """Whether /docs, /redoc and /openapi.json are served."""
        if self.enable_docs is None:
            return self.is_development
        return self.enable_docs

    @property
    def jwt_algorithm(self) -> str:
        """The (fixed) JWT signing algorithm."""
        return JWT_ALGORITHM

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create configuration from environment variables.

        Raises:
            ConfigError: If the configuration is unsafe for the selected environment.
        """
        enable_docs_raw = os.getenv("ENABLE_DOCS")
        return cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            timeout=int(os.getenv("API_TIMEOUT", "300")),
            reload=_env_bool("API_RELOAD", False),
            jwt_secret=os.getenv("JWT_SECRET", ""),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            api_keys=_split_csv(os.getenv("API_KEYS")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "10")),
            cors_origins=_split_csv(os.getenv("CORS_ORIGINS")),
            cors_methods=_split_csv(os.getenv("CORS_METHODS")) or ["GET", "POST", "OPTIONS"],
            cors_headers=_split_csv(os.getenv("CORS_HEADERS"))
            or ["Authorization", "Content-Type", "X-API-Key"],
            ws_max_connections=int(os.getenv("WS_MAX_CONNECTIONS", "100")),
            ws_max_connections_per_ip=int(os.getenv("WS_MAX_CONNECTIONS_PER_IP", "5")),
            max_concurrent_analyses=int(os.getenv("API_MAX_CONCURRENT_ANALYSES", "4")),
            environment=os.getenv("ENVIRONMENT", PRODUCTION),
            debug=_env_bool("DEBUG", False),
            enable_docs=None
            if enable_docs_raw is None or not enable_docs_raw.strip()
            else _env_bool("ENABLE_DOCS", False),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


_config: APIConfig | None = None


def get_config() -> APIConfig:
    """Return the process-wide configuration, loading it from the environment once.

    Raises:
        ConfigError: If the configuration is unsafe; the app then refuses to start.
    """
    global _config  # noqa: PLW0603
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def __getattr__(name: str) -> Any:
    # ``from mra_web.config import config`` loads (and validates) lazily, so the
    # config classes can be imported (e.g. by ``mra-api`` to report errors cleanly)
    # without an environment that is valid yet.
    if name == "config":
        return get_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
