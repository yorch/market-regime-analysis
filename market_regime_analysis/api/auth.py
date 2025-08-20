"""
Authentication and authorization for the Market Regime Analysis API.

This module provides JWT token authentication and API key validation.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from config import config

# Setup logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


class TokenData(BaseModel):
    """Token payload data model."""

    username: str | None = None
    exp: datetime | None = None


class User(BaseModel):
    """User model for authentication."""

    username: str
    email: str | None = None
    is_active: bool = True


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(hours=config.jwt_expiration_hours)

    to_encode.update({"exp": expire})

    try:
        encoded_jwt = jwt.encode(to_encode, config.jwt_secret, algorithm=config.jwt_algorithm)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token",
        ) from e


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, config.jwt_secret, algorithms=[config.jwt_algorithm])
        username: str | None = payload.get("sub")
        exp_timestamp = payload.get("exp")
        exp: datetime | None = (
            datetime.fromtimestamp(exp_timestamp, tz=UTC) if exp_timestamp else None
        )

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username, exp=exp)

        # Check if token is expired
        if exp and exp < datetime.now(UTC):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return token_data

    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise credentials_exception from e
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise credentials_exception from e


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:  # noqa: B008
    """Get current authenticated user from JWT token."""
    token_data = verify_token(credentials.credentials)

    # In a real application, you would fetch user from database
    # For now, we'll create a simple user from token data
    if not token_data.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token data",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = User(username=token_data.username, is_active=True)

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:  # noqa: B008
    """Get current active authenticated user."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


def create_user_token(username: str) -> str:
    """Create a token for a given username (for testing/demo purposes)."""
    access_token_expires = timedelta(hours=config.jwt_expiration_hours)
    access_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)
    return access_token


# Optional authentication dependency for endpoints that don't require auth
async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
) -> User | None:
    """Get current user if token is provided, otherwise return None."""
    if credentials is None:
        return None

    try:
        token_data = verify_token(credentials.credentials)
        if not token_data.username:
            return None
        user = User(username=token_data.username, is_active=True)
        return user if user.is_active else None
    except HTTPException:
        # If token is invalid, return None instead of raising exception
        return None


class AuthConfig:
    """Authentication configuration."""

    # For development/demo purposes, you can disable authentication
    REQUIRE_AUTH = config.environment == "production"

    # API key-based access for specific endpoints
    API_KEYS: ClassVar[dict[str, str]] = {
        "demo": "demo-api-key-12345",  # Demo key for testing
        "admin": "admin-api-key-67890",  # Admin key
    }


def verify_api_key(api_key: str) -> bool:
    """Verify API key for key-based authentication."""
    return api_key in AuthConfig.API_KEYS.values()


def get_api_key_user(api_key: str) -> str | None:
    """Get username associated with API key."""
    for username, key in AuthConfig.API_KEYS.items():
        if key == api_key:
            return username
    return None


# Authentication dependency that supports both JWT and API key
async def authenticate_request(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    api_key: str | None = None,
) -> User | None:
    """Authenticate request using JWT token or API key."""
    # Skip authentication in development mode if not required
    if not AuthConfig.REQUIRE_AUTH and config.environment == "development":
        return User(username="dev_user", is_active=True)

    # Try API key authentication first
    if api_key and verify_api_key(api_key):
        username = get_api_key_user(api_key)
        if username:
            return User(username=username, is_active=True)

    # Try JWT authentication
    if credentials:
        try:
            return await get_current_user(credentials)
        except HTTPException:
            pass

    # If neither authentication method works, raise exception
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Valid authentication required (JWT token or API key)",
        headers={"WWW-Authenticate": "Bearer"},
    )
