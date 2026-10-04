"""
Authentication and authorization for the Market Regime Analysis API.

Two credential types are accepted:

* ``Authorization: Bearer <JWT>`` — HS256 tokens signed with ``JWT_SECRET`` that
  must carry ``sub`` and ``exp`` claims. Mint them with the ``mra-token`` command.
* ``X-API-Key: <key>`` — static keys listed in the ``API_KEYS`` environment variable.

Unauthenticated access is allowed only when ``ENVIRONMENT=development``.
"""

import argparse
import hashlib
import hmac
import logging
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from mra_lib.config.env_file import EnvFileError, load_env_file
from mra_web.config import JWT_ALGORITHM, APIConfig, ConfigError, get_config

# Setup logging
logger = logging.getLogger(__name__)

# Security schemes. auto_error=False so missing credentials reach our own logic
# (development bypass, API key fallback) and produce a 401 instead of a 403.
security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

DEV_USERNAME = "dev_user"


class TokenData(BaseModel):
    """Token payload data model."""

    username: str | None = None
    exp: datetime | None = None


class User(BaseModel):
    """User model for authentication."""

    username: str
    email: str | None = None
    is_active: bool = True


def _unauthorized(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_app_config(conn: HTTPConnection) -> APIConfig:
    """Return the configuration of the app serving this request or WebSocket."""
    cfg = getattr(conn.app.state, "config", None)
    return cfg if isinstance(cfg, APIConfig) else get_config()


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
    cfg: APIConfig | None = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        data: Claims to include; must contain a non-empty ``sub``.
        expires_delta: Token lifetime (default ``JWT_EXPIRATION_HOURS``).
        cfg: Configuration to sign with (default: global config).
    """
    cfg = cfg or get_config()
    subject = data.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("Token data must include a non-empty 'sub' claim")

    now = datetime.now(UTC)
    lifetime = expires_delta or timedelta(hours=cfg.jwt_expiration_hours)
    to_encode = {**data, "iat": now, "exp": now + lifetime}
    token: str = jwt.encode(to_encode, cfg.jwt_secret, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str, cfg: APIConfig | None = None) -> TokenData:
    """Verify and decode a JWT token.

    Only HS256 is accepted, and the ``exp`` and ``sub`` claims are mandatory.

    Raises:
        HTTPException: 401 if the token is invalid, expired, or missing claims.
    """
    cfg = cfg or get_config()
    try:
        payload = jwt.decode(
            token,
            cfg.jwt_secret,
            algorithms=[JWT_ALGORITHM],
            options={"require_exp": True, "require_sub": True, "verify_exp": True},
        )
    except JWTError as e:
        logger.info("JWT validation failed: %s", type(e).__name__)
        raise _unauthorized() from e
    except Exception as e:  # malformed input that the library did not wrap
        logger.info("JWT verification error: %s", type(e).__name__)
        raise _unauthorized() from e

    username = payload.get("sub")
    exp_timestamp = payload.get("exp")
    if not isinstance(username, str) or not username.strip():
        raise _unauthorized()
    if not isinstance(exp_timestamp, int | float):
        raise _unauthorized()

    exp = datetime.fromtimestamp(exp_timestamp, tz=UTC)
    if exp <= datetime.now(UTC):
        raise _unauthorized("Token has expired")
    return TokenData(username=username, exp=exp)


def create_user_token(username: str, cfg: APIConfig | None = None) -> str:
    """Create a token for a given username with the configured lifetime."""
    cfg = cfg or get_config()
    return create_access_token(
        data={"sub": username},
        expires_delta=timedelta(hours=cfg.jwt_expiration_hours),
        cfg=cfg,
    )


def get_api_key_user(api_key: str, cfg: APIConfig | None = None) -> str | None:
    """Return a stable principal name for a valid API key, or None.

    Every configured key is compared with ``hmac.compare_digest`` so timing does
    not reveal which key (or how much of it) matched.
    """
    cfg = cfg or get_config()
    if not api_key:
        return None
    candidate = api_key.encode()
    matched: str | None = None
    for key in cfg.api_keys:
        if hmac.compare_digest(candidate, key.encode()):
            matched = key
    if matched is None:
        return None
    digest = hashlib.sha256(matched.encode()).hexdigest()[:12]
    return f"apikey-{digest}"


def verify_api_key(api_key: str, cfg: APIConfig | None = None) -> bool:
    """Verify an API key against ``API_KEYS``."""
    return get_api_key_user(api_key, cfg) is not None


def authenticate_credentials(
    cfg: APIConfig,
    bearer_token: str | None = None,
    api_key: str | None = None,
) -> User:
    """Authenticate a JWT and/or API key; fall back to the dev user in development.

    Credentials that are presented are always validated, even in development.

    Raises:
        HTTPException: 401 when credentials are invalid or missing.
    """
    if api_key:
        username = get_api_key_user(api_key, cfg)
        if username is None:
            raise _unauthorized("Invalid API key")
        return User(username=username, is_active=True)

    if bearer_token:
        token_data = verify_token(bearer_token, cfg)
        assert token_data.username is not None  # guaranteed by verify_token
        return User(username=token_data.username, is_active=True)

    if cfg.is_development:
        return User(username=DEV_USERNAME, is_active=True)

    raise _unauthorized("Authentication required (Bearer token or X-API-Key header)")


async def authenticate_request(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    api_key: str | None = Depends(api_key_header),
) -> User:
    """FastAPI dependency: authenticate via ``X-API-Key`` or a Bearer JWT."""
    return authenticate_credentials(
        get_app_config(request),
        bearer_token=credentials.credentials if credentials else None,
        api_key=api_key,
    )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
) -> User:
    """Get the current user from a mandatory Bearer JWT."""
    if credentials is None:
        raise _unauthorized("Bearer token required")
    token_data = verify_token(credentials.credentials, get_app_config(request))
    assert token_data.username is not None
    return User(username=token_data.username, is_active=True)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:  # noqa: B008
    """Get current active authenticated user."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
) -> User | None:
    """Get current user if a valid token is provided, otherwise return None."""
    if credentials is None:
        return None
    try:
        token_data = verify_token(credentials.credentials, get_app_config(request))
    except HTTPException:
        return None
    return User(username=token_data.username or "", is_active=True)


def mint_token_main(argv: list[str] | None = None) -> int:
    """Entry point for ``mra-token``: print a JWT signed with ``JWT_SECRET``.

    Run it where the server's environment is available, e.g.
    ``JWT_SECRET=... uv run mra-token --sub alice --hours 12``.
    """
    parser = argparse.ArgumentParser(
        prog="mra-token", description="Mint a JWT for the Market Regime Analysis API."
    )
    parser.add_argument("--sub", required=True, help="Subject (user or client name)")
    parser.add_argument(
        "--hours", type=float, default=None, help="Lifetime in hours (default JWT_EXPIRATION_HOURS)"
    )
    args = parser.parse_args(argv)

    try:
        cfg = APIConfig.from_env()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if cfg.jwt_secret_ephemeral:
        print(
            "error: JWT_SECRET is not set to a strong value; a token signed with a random "
            "per-process secret would be rejected by the server. Set JWT_SECRET first.",
            file=sys.stderr,
        )
        return 2
    if not args.sub.strip():
        print("error: --sub must not be empty", file=sys.stderr)
        return 2
    if args.hours is not None and args.hours <= 0:
        print("error: --hours must be positive", file=sys.stderr)
        return 2

    delta = timedelta(hours=args.hours) if args.hours is not None else None
    print(create_access_token({"sub": args.sub.strip()}, expires_delta=delta, cfg=cfg))
    return 0


def main() -> None:
    """Console-script wrapper for :func:`mint_token_main`; loads ``.env`` first."""
    try:
        load_env_file()
    except EnvFileError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(mint_token_main())
